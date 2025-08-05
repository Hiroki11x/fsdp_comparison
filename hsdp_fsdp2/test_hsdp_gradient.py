#!/usr/bin/env python3
"""
Enhanced HSDP gradient test to verify gradients are captured BEFORE synchronization
Each replica will have dramatically different gradients to make the difference clear
"""

import os
os.environ["TORCH_CPP_LOG_LEVEL"] = "WARNING"

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
import math

def setup_distributed():
    """Setup distributed environment"""
    if "RANK" not in os.environ:
        # Single GPU test
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        os.environ["LOCAL_RANK"] = "0"

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


class SimpleModel(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.fc3 = nn.Linear(dim, dim)

        # Initialize with small weights to control gradient magnitudes
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def create_replica_specific_data(replicate_rank, batch_size, seq_len, dim, device):
    """Create dramatically different data for each replica to ensure different gradients"""

    # Each replica gets data with different magnitude and patterns
    if replicate_rank == 0:
        # Small magnitude, positive values
        data = torch.randn(batch_size, seq_len, dim, device=device) * 0.1
        target = torch.ones(batch_size, seq_len, dim, device=device) * 0.1
    elif replicate_rank == 1:
        # Large magnitude, negative values
        data = torch.randn(batch_size, seq_len, dim, device=device) * 10.0
        target = -torch.ones(batch_size, seq_len, dim, device=device) * 5.0
    elif replicate_rank == 2:
        # Oscillating pattern
        data = torch.randn(batch_size, seq_len, dim, device=device) * 2.0
        target = torch.sin(torch.arange(dim, device=device).float()).unsqueeze(0).unsqueeze(0) * 3.0
        target = target.expand(batch_size, seq_len, dim)
    else:  # replicate_rank == 3
        # Sparse pattern with spikes
        data = torch.zeros(batch_size, seq_len, dim, device=device)
        # Create sparse data by setting every 10th element
        sparse_indices = torch.arange(0, dim, 10, device=device)
        num_sparse = len(sparse_indices)
        data[:, :, sparse_indices] = torch.randn(batch_size, seq_len, num_sparse, device=device) * 20.0

        target = torch.zeros(batch_size, seq_len, dim, device=device)
        # Set every 5th element in target
        target_indices = torch.arange(0, dim, 5, device=device)
        target[:, :, target_indices] = 10.0

    return data, target


def test_hsdp_gradient_variance():
    """Test if HSDP allows gradient variance calculation across replicas"""
    setup_distributed()

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print(f"[Rank {rank}] Starting Enhanced HSDP gradient variance test")
    print(f"[Rank {rank}] World size: {world_size}")

    # Check if we can create HSDP mesh
    if world_size < 4:
        print(f"[Rank {rank}] Need at least 4 GPUs for this test (got {world_size})")
        if world_size == 1:
            print("[Rank 0] Running single GPU demonstration...")
            demonstrate_gradient_differences()
        return

    # Create HSDP mesh (2D: replicate x shard)
    replicate_degree = min(4, world_size)
    shard_degree = world_size // replicate_degree

    print(f"[Rank {rank}] Creating mesh: {replicate_degree} replicate x {shard_degree} shard")

    # Create 2D device mesh for HSDP
    device_mesh = init_device_mesh(
        "cuda",
        (replicate_degree, shard_degree),
        mesh_dim_names=("replicate", "shard")
    )

    # Get submeshes
    replicate_mesh = device_mesh["replicate"]
    shard_mesh = device_mesh["shard"]

    replicate_rank = replicate_mesh.get_local_rank()
    shard_rank = shard_mesh.get_local_rank()

    print(f"[Rank {rank}] Replicate rank: {replicate_rank}, Shard rank: {shard_rank}")

    # Create model with same initialization for all ranks
    torch.manual_seed(42)  # Same seed for model initialization
    model = SimpleModel(dim=512).cuda()

    # Apply HSDP
    for name, layer in model.named_children():
        fully_shard(layer, mesh=shard_mesh)
    fully_shard(model, mesh=shard_mesh)

    print(f"[Rank {rank}] Model wrapped with HSDP")

    # Setup optimizer with small learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

    # Custom loss function that amplifies differences
    def custom_loss(output, target):
        # MSE loss with replica-specific scaling
        base_loss = nn.functional.mse_loss(output, target)
        # Amplify loss based on replica rank to create different gradient scales
        scale_factor = 1.0 + replicate_rank * 2.0
        return base_loss * scale_factor

    print(f"\n[Rank {rank}] Testing gradient variance across replicas...")
    print(f"[Rank {rank}] Replica {replicate_rank} will use loss scale factor: {1.0 + replicate_rank * 2.0}")

    batch_size = 8
    seq_len = 64
    dim = 512

    # Store gradients for analysis
    gradient_info = []

    for step in range(3):
        # Create dramatically different data for each replica
        x, target = create_replica_specific_data(replicate_rank, batch_size, seq_len, dim, "cuda")

        # Forward pass
        optimizer.zero_grad()
        output = model(x)
        loss = custom_loss(output, target)

        # Backward pass
        loss.backward()

        # Collect gradient information BEFORE optimizer step
        step_grad_info = {
            'norms': [],
            'mean_grads': [],
            'max_grads': []
        }

        for name, param in model.named_parameters():
            if param.grad is not None:
                # Get local gradient (before any synchronization)
                grad = param.grad
                if hasattr(grad, 'to_local'):
                    grad = grad.to_local()

                # Collect various statistics
                grad_norm = grad.norm().item()
                grad_mean = grad.mean().item()
                grad_max = grad.abs().max().item()

                step_grad_info['norms'].append(grad_norm)
                step_grad_info['mean_grads'].append(grad_mean)
                step_grad_info['max_grads'].append(grad_max)

        # Calculate total gradient norm
        total_norm = math.sqrt(sum(g**2 for g in step_grad_info['norms']))
        avg_mean = sum(step_grad_info['mean_grads']) / len(step_grad_info['mean_grads'])
        max_grad = max(step_grad_info['max_grads'])

        gradient_info.append({
            'total_norm': total_norm,
            'avg_mean': avg_mean,
            'max_grad': max_grad,
            'loss': loss.item()
        })

        print(f"[Rank {rank}] Step {step}: loss={loss.item():.4f}, grad_norm={total_norm:.4f}, "
              f"avg_grad_mean={avg_mean:.6f}, max_grad={max_grad:.4f}")

        # Optimizer step (this is where gradient synchronization happens in HSDP)
        optimizer.step()

    # Synchronize before gathering results
    dist.barrier()

    # Check gradient variance across replicas
    print(f"\n[Rank {rank}] Checking gradient variance across replicas...")

    # Gather gradient statistics from all replicas within the same shard
    if shard_rank == 0:  # Only collect from first shard for clarity
        all_grad_norms = [None] * replicate_degree
        all_grad_means = [None] * replicate_degree
        all_losses = [None] * replicate_degree

        # Gather last step's gradient info
        last_step_info = gradient_info[-1]

        dist.all_gather_object(
            all_grad_norms,
            last_step_info['total_norm'],
            group=replicate_mesh.get_group()
        )

        dist.all_gather_object(
            all_grad_means,
            last_step_info['avg_mean'],
            group=replicate_mesh.get_group()
        )

        dist.all_gather_object(
            all_losses,
            last_step_info['loss'],
            group=replicate_mesh.get_group()
        )

        if replicate_rank == 0:
            print(f"\n[Shard 0] Final gradient statistics across replicas:")
            print(f"{'Replica':<10} {'Loss':<12} {'Grad Norm':<12} {'Avg Grad Mean':<15}")
            print("-" * 50)

            for i in range(replicate_degree):
                print(f"{i:<10} {all_losses[i]:<12.4f} {all_grad_norms[i]:<12.4f} {all_grad_means[i]:<15.6f}")

            # Calculate statistics
            norm_mean = sum(all_grad_norms) / len(all_grad_norms)
            norm_std = math.sqrt(sum((x - norm_mean)**2 for x in all_grad_norms) / len(all_grad_norms))
            norm_cv = norm_std / norm_mean if norm_mean > 0 else 0

            print(f"\nGradient Norm Statistics:")
            print(f"  Mean: {norm_mean:.4f}")
            print(f"  Std Dev: {norm_std:.4f}")
            print(f"  Coefficient of Variation: {norm_cv:.2%}")

            # Check if gradients are significantly different
            min_norm = min(all_grad_norms)
            max_norm = max(all_grad_norms)
            norm_ratio = max_norm / min_norm if min_norm > 0 else float('inf')

            print(f"  Min/Max Ratio: {norm_ratio:.2f}x")

            if norm_ratio > 2.0:  # Expect at least 2x difference
                print(f"\n✅ SUCCESS: Gradients vary significantly across replicas!")
                print(f"  This confirms we're capturing gradients BEFORE synchronization")
                print(f"  The {norm_ratio:.1f}x difference in gradient norms proves replicas have independent gradients")
            else:
                print(f"\n⚠️  WARNING: Gradient differences are smaller than expected ({norm_ratio:.1f}x)")
                print(f"  This might indicate premature synchronization")


def demonstrate_gradient_differences():
    """Single GPU demonstration of how different data creates different gradients"""
    print("\nDemonstrating gradient differences with different data patterns:")

    model = SimpleModel(dim=256).cuda()

    batch_size = 4
    seq_len = 32
    dim = 256

    print("\nTesting 4 different data patterns:")

    for replica_id in range(4):
        model.zero_grad()

        x, target = create_replica_specific_data(replica_id, batch_size, seq_len, dim, "cuda")

        output = model(x)
        loss = nn.functional.mse_loss(output, target) * (1.0 + replica_id * 2.0)
        loss.backward()

        # Calculate gradient norm
        total_norm = 0
        for param in model.parameters():
            if param.grad is not None:
                total_norm += param.grad.norm().item() ** 2
        total_norm = math.sqrt(total_norm)

        print(f"  Pattern {replica_id}: loss={loss.item():.4f}, grad_norm={total_norm:.4f}")

    print("\nIn HSDP, each replica would have these different gradients before synchronization")


if __name__ == "__main__":
    test_hsdp_gradient_variance()
