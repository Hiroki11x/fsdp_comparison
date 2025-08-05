# FSDP1 vs FSDP2 Parameter & Gradient Access Comparison Suite

A comprehensive testing framework for comparing parameter and gradient accessibility between PyTorch's FSDP1 (FullyShardedDataParallel) and FSDP2 (fully_shard) implementations, with a focus on identifying functionality regressions in gradient access patterns.

## Overview

This test suite provides detailed comparisons between FSDP1 and FSDP2 implementations, specifically targeting:
- Parameter accessibility patterns
- Gradient collection capabilities with/without synchronization
- Individual microbatch gradient isolation
- Multi-GPU distributed behavior

## Key Features

- **Comprehensive Testing**: Tests both single-GPU and multi-GPU scenarios
- **Gradient Access Analysis**: Detailed comparison of gradient accessibility with different sync settings
- **Regression Detection**: Automatically identifies functionality regressions between FSDP1 and FSDP2
- **Detailed Logging**: Generates separate log files for each test configuration

## Requirements

- Python 3.10+
- PyTorch 2.9.0a0+git55ff4f8
- NVIDIA GPUs (tested on A100)
- NCCL for distributed operations

## Installation

```bash
# Clone the repository
git clone git@github.com:Hiroki11x/fsdp_comparison.git
cd fsdp_comparison
```

## Usage

### Single and Multi-GPU GPU Test
```bash
./run_comparison.sh
```

## Test Components

### 1. Environment Verification
- Checks Python and PyTorch versions
- Verifies CUDA availability and device count
- Confirms FSDP1 and FSDP2 availability

### 2. Parameter Access Tests
- **Initial Parameter Access**: Tests parameter accessibility before any computation
- **Post-Forward Parameter Access**: Tests parameter accessibility after forward pass

### 3. Gradient Access Tests
- **Normal Sync**: Tests gradient accessibility with default synchronization
- **No Sync**: Tests gradient accessibility with synchronization disabled
- **Microbatch Gradients**: Tests ability to collect individual microbatch gradients

### 4. Multi-GPU Tests
- Automatically scales tests to available GPUs
- Tests with 2 and 4 GPU configurations
- Verifies distributed behavior consistency

## Test Results Interpretation

### Success Indicators
- ✅ **Both Work**: Both FSDP1 and FSDP2 pass the test
- ✅ **No Regressions**: No functionality degradation detected

### Regression Indicators
- 🚨 **REGRESSION**: FSDP1 works but FSDP2 fails
- ❌ **Test Failure**: Specific test case failed

### Key Metrics
- **Parameter Accessibility**: Ratio of accessible parameters
- **Gradient Accessibility**: Ratio of accessible gradients
- **Microbatch Success Rate**: Percentage of successful microbatch gradient collections

## Expected Findings

Based on the test results, the suite typically reveals:

1. **Parameter Access**: Both FSDP1 and FSDP2 generally work well
2. **Gradient Access with Sync**: FSDP1 succeeds while FSDP2 may fail
3. **Gradient Access without Sync**: Both implementations typically succeed
4. **Microbatch Collection**: Both achieve 100% success rate in all cases

## Output Files

The test suite generates the following log files:
- `single_gpu_output.log`: Detailed single GPU test results
- `multi_gpu_2_output.log`: 2-GPU distributed test results
- `multi_gpu_4_output.log`: 4-GPU distributed test results

## Architecture

### SimpleModel
A basic neural network with configurable layers used for testing:
```python
- Multiple Linear layers
- ReLU activations
- Configurable hidden size and layer count
```

### ParameterAccessTester
Core testing class that handles:
- Parameter accessibility checks
- Gradient accessibility verification
- DTensor to local tensor conversion
- Comprehensive error handling

## Technical Details

### FSDP1 Implementation
- Uses `torch.distributed.fsdp.FullyShardedDataParallel`
- Supports `no_sync()` context manager
- Direct gradient access through `param.grad`

### FSDP2 Implementation
- Uses `torch.distributed.fsdp.fully_shard`
- Supports `set_requires_gradient_sync()`
- Gradient access through `unsharded_accumulated_grad`

## Acknowledgments

This test suite was developed to help identify and track functionality differences between PyTorch's FSDP implementations, ensuring smooth migration paths for users upgrading from FSDP1 to FSDP2.
Thanks [Wei (Will) Feng](https://github.com/weifengpy) for helping us to implement FSDP2 part.
