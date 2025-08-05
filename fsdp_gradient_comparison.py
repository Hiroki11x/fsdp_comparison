#!/usr/bin/env python3
"""
torchrun --nproc-per-node=4 fsdp_gradient_comparison.py

FSDP1 vs FSDP2 Parameter/Gradient Access Comparison

Purpose: Compare parameter and gradient accessibility between FSDP1 and FSDP2
with different sync settings to identify functionality regressions.

Key Tests:
1. Parameter access with/without sync
2. Gradient collection with/without sync
3. Individual microbatch gradient isolation
"""

import os
import sys
import contextlib
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._tensor import DTensor
from typing import List, Dict, Tuple, Optional
import traceback


def setup_distributed():
    """Setup distributed environment"""
    if "RANK" not in os.environ:
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12356"
        os.environ["LOCAL_RANK"] = "0"

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


class SimpleModel(nn.Module):
    """Simple test model with multiple layers"""
    def __init__(self, hidden_size=512, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)
        ])
        self.output = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        return self.output(x)


class ParameterAccessTester:
    """Test parameter and gradient access capabilities"""

    @staticmethod
    def test_parameter_access(model, test_name: str) -> Dict[str, any]:
        """Test parameter accessibility"""
        results = {
            'test_name': test_name,
            'accessible_params': 0,
            'total_params': 0,
            'param_details': [],
            'success': False
        }

        try:
            for name, param in model.named_parameters():
                results['total_params'] += 1

                # Test various parameter access methods
                access_methods = {
                    'data': lambda p: p.data,
                    'shape': lambda p: p.shape,
                    'requires_grad': lambda p: p.requires_grad,
                    'is_sharded': lambda p: hasattr(p, '_fsdp_flattened'),
                    'to_local': lambda p: p.to_local() if isinstance(p, DTensor) else p
                }

                param_info = {'name': name}
                accessible = True

                for method_name, method in access_methods.items():
                    try:
                        result = method(param)
                        param_info[method_name] = str(type(result)) if method_name == 'data' else result
                    except Exception as e:
                        param_info[method_name] = f"Error: {e}"
                        if method_name in ['data', 'shape']:  # Critical access methods
                            accessible = False

                if accessible:
                    results['accessible_params'] += 1

                results['param_details'].append(param_info)

            results['success'] = results['accessible_params'] > 0

        except Exception as e:
            results['error'] = str(e)

        return results

    @staticmethod
    def test_gradient_access_fsdp1(model, test_name: str, clear_grads: bool = True) -> Dict[str, any]:
        """Test gradient accessibility"""
        results = {
            'test_name': test_name,
            'accessible_grads': 0,
            'total_grads': 0,
            'grad_details': [],
            'success': False,
            'gradient_norms': []
        }

        try:
            if clear_grads:
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad = None

            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue

                results['total_grads'] += 1

                # Test gradient access methods
                grad_sources = {
                    'grad': param.grad,
                    'main_grad': getattr(param, 'main_grad', None),
                    'cuda_grad': getattr(param, 'cuda_grad', None),
                }

                grad_info = {'name': name}
                accessible_grad = None

                for source_name, grad in grad_sources.items():
                    if grad is not None:
                        try:
                            # Convert DTensor to local if needed
                            if isinstance(grad, DTensor):
                                local_grad = grad.to_local()
                                grad_info[source_name] = f"DTensor->local, shape={local_grad.shape}"
                                accessible_grad = local_grad
                            else:
                                grad_info[source_name] = f"Tensor, shape={grad.shape}"
                                accessible_grad = grad

                        except Exception as e:
                            grad_info[source_name] = f"Error: {e}"
                    else:
                        grad_info[source_name] = "None"

                if accessible_grad is not None:
                    results['accessible_grads'] += 1
                    results['gradient_norms'].append(accessible_grad.norm().item())

                results['grad_details'].append(grad_info)

            results['success'] = results['accessible_grads'] > 0

        except Exception as e:
            results['error'] = str(e)

        return results

    @staticmethod
    def test_gradient_access_fsdp2(model, test_name: str, clear_grads: bool = True) -> Dict[str, any]:
        """Test gradient accessibility"""
        results = {
            'test_name': test_name,
            'accessible_grads': 0,
            'total_grads': 0,
            'grad_details': [],
            'success': False,
            'gradient_norms': []
        }

        try:
            if clear_grads:
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad = None

            for fsdp_module in model.modules():
                from torch.distributed.fsdp import FSDPModule
                if not isinstance(fsdp_module, FSDPModule):
                    continue

                for fsdp_param in fsdp_module._get_fsdp_state()._fsdp_param_group.fsdp_params:
                    param_name = fsdp_param._param_fqn
                    unsharded_param = fsdp_param._unsharded_param

                    if not unsharded_param.requires_grad:
                        continue

                    grad_info = {'name': param_name}
                    # https://github.com/pytorch/pytorch/blob/6b414f56a4a133a428af618d8ed1553849341497/torch/distributed/fsdp/_fully_shard/_fsdp_param.py#L640-L646
                    grad_sources = {
                        'unsharded_param.grad': unsharded_param.grad,
                        'unsharded_accumulated_grad': fsdp_param.unsharded_accumulated_grad,
                    }
                    accessible_grad = None

                    for source_name, grad in grad_sources.items():
                        if grad is not None:
                            try:
                                # Convert DTensor to local if needed
                                if isinstance(grad, DTensor):
                                    local_grad = grad.to_local()
                                    grad_info[source_name] = f"DTensor->local, shape={local_grad.shape}"
                                    accessible_grad = local_grad
                                else:
                                    grad_info[source_name] = f"Tensor, shape={grad.shape}"
                                    accessible_grad = grad
                                break
                            except Exception as e:
                                grad_info[source_name] = f"Error: {e}"
                        else:
                            grad_info[source_name] = "None"

                    if accessible_grad is not None:
                        results['accessible_grads'] += 1
                        results['gradient_norms'].append(accessible_grad.norm().item())

                    results['grad_details'].append(grad_info)

                results['success'] = results['accessible_grads'] > 0
        except Exception as e:
            results['error'] = str(e)
        return results

    @staticmethod
    def print_results(results: Dict[str, any], verbose: bool = False):
        """Print test results in a formatted way"""
        print(f"\n📊 {results['test_name']}:")
        print(f"  Success: {'✅' if results['success'] else '❌'}")

        if 'accessible_params' in results:
            print(f"  Parameters: {results['accessible_params']}/{results['total_params']} accessible")

        if 'accessible_grads' in results:
            print(f"  Gradients: {results['accessible_grads']}/{results['total_grads']} accessible")
            if results['gradient_norms']:
                avg_norm = sum(results['gradient_norms']) / len(results['gradient_norms'])
                print(f"  Avg gradient norm: {avg_norm:.6f}")

        if 'error' in results:
            print(f"  Error: {results['error']}")

        if verbose and ('param_details' in results or 'grad_details' in results):
            details = results.get('param_details', results.get('grad_details', []))
            for detail in details[:3]:  # Show first 3 for brevity
                print(f"    {detail}")
            if len(details) > 3:
                print(f"    ... and {len(details) - 3} more")


def test_fsdp1_capabilities():
    """Comprehensive FSDP1 testing"""
    print("\n" + "=" * 70)
    print("FSDP1 (FullyShardedDataParallel) Capabilities Test")
    print("=" * 70)

    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        print("✓ FSDP1 imported successfully")
    except ImportError as e:
        print(f"❌ FSDP1 not available: {e}")
        return {}

    # Create and wrap model
    model = SimpleModel(hidden_size=512, num_layers=3).cuda()
    model = FSDP(
        model,
        auto_wrap_policy=lambda module, recurse, nonwrapped_numel: isinstance(module, nn.Linear),
        mixed_precision=None,
        device_id=torch.cuda.current_device(),
    )

    print(f"Model type: {type(model)}")
    print(f"Has no_sync: {hasattr(model, 'no_sync')}")

    # Test setup
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # Generate test data
    batch_size, seq_len, hidden_size = 8, 128, 512
    x = torch.randn(batch_size, seq_len, hidden_size, device="cuda")
    target = torch.randn(batch_size, seq_len, hidden_size, device="cuda")

    results = {}
    tester = ParameterAccessTester()

    # Test 1: Parameter access (no computation)
    results['param_access_initial'] = tester.test_parameter_access(
        model, "FSDP1 Initial Parameter Access"
    )

    # Test 2: Parameter access after forward pass
    optimizer.zero_grad()
    output = model(x)
    results['param_access_after_forward'] = tester.test_parameter_access(
        model, "FSDP1 Parameter Access After Forward"
    )

    # Test 3: Gradient access with normal sync
    loss = loss_fn(output, target)
    loss.backward()
    results['grad_access_with_sync'] = tester.test_gradient_access_fsdp1(
        model, "FSDP1 Gradient Access (Normal Sync)", clear_grads=False
    )

    # Test 4: Gradient access with no_sync
    optimizer.zero_grad()
    if hasattr(model, 'no_sync'):
        with model.no_sync():
            output = model(x)
            loss = loss_fn(output, target)
            loss.backward()
            results['grad_access_no_sync'] = tester.test_gradient_access_fsdp1(
                model, "FSDP1 Gradient Access (No Sync)", clear_grads=False
            )

    # Test 5: Individual microbatch gradients
    print(f"\n🔬 FSDP1 Individual Microbatch Test:")
    microbatch_results = []
    num_microbatches = 3

    for i in range(num_microbatches):
        optimizer.zero_grad()
        mb_x = torch.randn(batch_size//2, seq_len, hidden_size, device="cuda")
        mb_target = torch.randn(batch_size//2, seq_len, hidden_size, device="cuda")

        output = model(mb_x)
        loss = loss_fn(output, mb_target)
        loss.backward()

        mb_result = tester.test_gradient_access_fsdp1(
            model, f"FSDP1 Microbatch {i+1}", clear_grads=False
        )
        microbatch_results.append(mb_result['success'])
        print(f"  Microbatch {i+1}: {'✅' if mb_result['success'] else '❌'}")

    results['microbatch_success_rate'] = sum(microbatch_results) / len(microbatch_results)

    # Print all results
    for result in results.values():
        if isinstance(result, dict):
            tester.print_results(result)

    print(f"\n📈 FSDP1 Microbatch Success Rate: {results['microbatch_success_rate']*100:.1f}%")

    return results


def test_fsdp2_capabilities():
    """Comprehensive FSDP2 testing"""
    print("\n" + "=" * 70)
    print("FSDP2 (fully_shard) Capabilities Test")
    print("=" * 70)

    try:
        from torch.distributed.fsdp import fully_shard
        print("✓ FSDP2 imported successfully")
    except ImportError as e:
        print(f"❌ FSDP2 not available: {e}")
        return {}

    # Device mesh for FSDP2
    device_mesh = init_device_mesh("cuda", (dist.get_world_size(),))

    # Create model on meta device
    with torch.device("meta"):
        model = SimpleModel(hidden_size=512, num_layers=3)

    # Apply FSDP2
    for layer in model.layers:
        fully_shard(layer)
    fully_shard(model)

    # Initialize model
    model.to_empty(device="cuda")
    with torch.no_grad():
        for param in model.parameters():
            if param.requires_grad:
                nn.init.normal_(param, mean=0.0, std=0.02)

    print(f"Model type: {type(model)}")
    print(f"Has set_requires_gradient_sync: {hasattr(model, 'set_requires_gradient_sync')}")

    # Test setup
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # Generate test data
    batch_size, seq_len, hidden_size = 8, 128, 512
    x = torch.randn(batch_size, seq_len, hidden_size, device="cuda")
    target = torch.randn(batch_size, seq_len, hidden_size, device="cuda")

    results = {}
    tester = ParameterAccessTester()

    # Test 1: Parameter access (no computation)
    results['param_access_initial'] = tester.test_parameter_access(
        model, "FSDP2 Initial Parameter Access"
    )

    # Test 2: Parameter access after forward pass
    optimizer.zero_grad()
    output = model(x)
    results['param_access_after_forward'] = tester.test_parameter_access(
        model, "FSDP2 Parameter Access After Forward"
    )

    # # Test 3: Gradient access with normal sync
    loss = loss_fn(output, target)
    loss.backward()
    results['grad_access_with_sync'] = tester.test_gradient_access_fsdp2(
        model, "FSDP2 Gradient Access (Normal Sync)", clear_grads=False
    )

    # Test 4: Gradient access with disabled sync
    optimizer.zero_grad()
    if hasattr(model, 'set_requires_gradient_sync'):
        model.set_requires_gradient_sync(False)
        output = model(x)
        loss = loss_fn(output, target)
        loss.backward()
        results['grad_access_no_sync'] = tester.test_gradient_access_fsdp2(
            model, "FSDP2 Gradient Access (Sync Disabled)", clear_grads=False
        )
        model.set_requires_gradient_sync(True)

    # Test 5: Individual microbatch gradients
    print(f"\n🔬 FSDP2 Individual Microbatch Test:")
    microbatch_results = []
    num_microbatches = 3

    for i in range(num_microbatches):
        optimizer.zero_grad()

        # Disable sync for individual microbatch
        if hasattr(model, 'set_requires_gradient_sync'):
            model.set_requires_gradient_sync(False)

        mb_x = torch.randn(batch_size//2, seq_len, hidden_size, device="cuda")
        mb_target = torch.randn(batch_size//2, seq_len, hidden_size, device="cuda")

        output = model(mb_x)
        loss = loss_fn(output, mb_target)
        loss.backward()

        mb_result = tester.test_gradient_access_fsdp2(
            model, f"FSDP2 Microbatch {i+1}", clear_grads=False
        )
        microbatch_results.append(mb_result['success'])
        print(f"  Microbatch {i+1}: {'✅' if mb_result['success'] else '❌'}")

        # Re-enable sync
        if hasattr(model, 'set_requires_gradient_sync'):
            model.set_requires_gradient_sync(True)

    results['microbatch_success_rate'] = sum(microbatch_results) / len(microbatch_results)

    # Print all results
    for result in results.values():
        if isinstance(result, dict):
            tester.print_results(result)

    print(f"\n📈 FSDP2 Microbatch Success Rate: {results['microbatch_success_rate']*100:.1f}%")

    return results


def compare_results(fsdp1_results: Dict, fsdp2_results: Dict):
    """Compare FSDP1 and FSDP2 results"""
    print("\n" + "=" * 70)
    print("FSDP1 vs FSDP2 DETAILED COMPARISON")
    print("=" * 70)

    # Comparison categories
    categories = [
        ('param_access_initial', 'Initial Parameter Access'),
        ('param_access_after_forward', 'Parameter Access After Forward'),
        ('grad_access_with_sync', 'Gradient Access (With Sync)'),
        ('grad_access_no_sync', 'Gradient Access (No Sync)'),
        ('microbatch_success_rate', 'Microbatch Gradient Collection'),
    ]

    for key, description in categories:
        print(f"\n🔍 {description}:")

        if key == 'microbatch_success_rate':
            fsdp1_rate = fsdp1_results.get(key, 0) * 100
            fsdp2_rate = fsdp2_results.get(key, 0) * 100
            print(f"  FSDP1: {fsdp1_rate:.1f}% success rate")
            print(f"  FSDP2: {fsdp2_rate:.1f}% success rate")

            if fsdp1_rate > fsdp2_rate:
                print(f"  🚨 REGRESSION: FSDP1 outperforms FSDP2 by {fsdp1_rate - fsdp2_rate:.1f}%")
            elif fsdp2_rate > fsdp1_rate:
                print(f"  ✅ IMPROVEMENT: FSDP2 outperforms FSDP1 by {fsdp2_rate - fsdp1_rate:.1f}%")
            else:
                print(f"  ⚖️  EQUAL: Both have same performance")
        else:
            fsdp1_success = fsdp1_results.get(key, {}).get('success', False)
            fsdp2_success = fsdp2_results.get(key, {}).get('success', False)

            fsdp1_status = "✅" if fsdp1_success else "❌"
            fsdp2_status = "✅" if fsdp2_success else "❌"

            print(f"  FSDP1: {fsdp1_status}")
            print(f"  FSDP2: {fsdp2_status}")

            if fsdp1_success and not fsdp2_success:
                print(f"  🚨 REGRESSION: FSDP1 works but FSDP2 fails")
            elif not fsdp1_success and fsdp2_success:
                print(f"  ✅ IMPROVEMENT: FSDP2 works but FSDP1 fails")
            elif fsdp1_success and fsdp2_success:
                print(f"  ✅ BOTH WORK")
            else:
                print(f"  ❌ BOTH FAIL")

    # Overall assessment
    print(f"\n" + "=" * 70)
    print("FINAL ASSESSMENT")
    print("=" * 70)

    # Key regression indicators
    microbatch_regression = (
        fsdp1_results.get('microbatch_success_rate', 0) >
        fsdp2_results.get('microbatch_success_rate', 0)
    )

    grad_no_sync_regression = (
        fsdp1_results.get('grad_access_no_sync', {}).get('success', False) and
        not fsdp2_results.get('grad_access_no_sync', {}).get('success', False)
    )

    if microbatch_regression or grad_no_sync_regression:
        print("🚨 FUNCTIONALITY REGRESSION DETECTED:")
        if microbatch_regression:
            print("  - FSDP2 has reduced microbatch gradient collection capability")
        if grad_no_sync_regression:
            print("  - FSDP2 has issues with gradient access when sync is disabled")
        print("\n  This confirms that FSDP2 has regressions in gradient access")
        print("  capabilities compared to FSDP1.")
    else:
        print("✅ No significant regressions detected between FSDP1 and FSDP2")

    return {
        'microbatch_regression': microbatch_regression,
        'grad_no_sync_regression': grad_no_sync_regression,
        'overall_regression': microbatch_regression or grad_no_sync_regression
    }


def main():
    """Main execution function"""
    setup_distributed()

    print("🚀 FSDP1 vs FSDP2 Parameter/Gradient Access Comparison")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Distributed initialized: {dist.is_initialized()}")
    print(f"World size: {dist.get_world_size()}")
    print(f"Rank: {dist.get_rank()}")

    fsdp1_results = {}
    fsdp2_results = {}

    # Run FSDP1 tests
    try:
        fsdp1_results = test_fsdp1_capabilities()
    except Exception as e:
        print(f"\n❌ FSDP1 test failed: {e}")
        traceback.print_exc()

    # Run FSDP2 tests
    try:
        fsdp2_results = test_fsdp2_capabilities()
    except Exception as e:
        print(f"\n❌ FSDP2 test failed: {e}")
        traceback.print_exc()

    # Compare results
    if fsdp1_results and fsdp2_results:
        comparison = compare_results(fsdp1_results, fsdp2_results)
        return 0 if not comparison['overall_regression'] else 1
    else:
        print("\n❌ Could not complete comparison due to test failures")
        return 1


if __name__ == "__main__":
    sys.exit(main())
