#!/bin/bash

# FSDP1 vs FSDP2 Parameter/Gradient Access Comparison Script
# Enhanced version with detailed sync behavior analysis

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                 FSDP1 vs FSDP2 Comparison Suite                     ║"
echo "║            Parameter & Gradient Access Capability Test              ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo
echo "🎯 Purpose: Compare parameter/gradient accessibility between FSDP1 and FSDP2"
echo "🔍 Focus: Sync behavior impact on individual gradient collection"
echo "📋 Expected: FSDP1 allows microbatch gradients, FSDP2 has limitations"
echo

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    local status=$1
    local message=$2
    case $status in
        "SUCCESS") echo -e "${GREEN}✅ $message${NC}" ;;
        "ERROR") echo -e "${RED}❌ $message${NC}" ;;
        "WARNING") echo -e "${YELLOW}⚠️  $message${NC}" ;;
        "INFO") echo -e "${BLUE}ℹ️  $message${NC}" ;;
    esac
}

# Environment verification
echo "🔧 Environment Verification"
echo "─────────────────────────────"

# Check Python and PyTorch
# Stderr is already redirected here, which is correct.
python3 -c "
import sys
import torch
print(f'Python version: {sys.version.split()[0]}')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device count: {torch.cuda.device_count()}')
    print(f'CUDA device name: {torch.cuda.get_device_name(0)}')

# Check FSDP availability
fsdp1_available = False
fsdp2_available = False

try:
    from torch.distributed.fsdp import FullyShardedDataParallel
    fsdp1_available = True
    print('FSDP1 (FullyShardedDataParallel): ✅ Available')
except ImportError as e:
    print(f'FSDP1 (FullyShardedDataParallel): ❌ {e}')

try:
    from torch.distributed.fsdp import fully_shard
    fsdp2_available = True
    print('FSDP2 (fully_shard): ✅ Available')
except ImportError as e:
    print(f'FSDP2 (fully_shard): ❌ {e}')

# Check distributed capabilities
try:
    import torch.distributed as dist
    print('torch.distributed: ✅ Available')
except ImportError:
    print('torch.distributed: ❌ Not available')

if not (fsdp1_available or fsdp2_available):
    print('\\n❌ Neither FSDP1 nor FSDP2 is available!')
    exit(1)
elif not fsdp1_available:
    print('\\n⚠️  FSDP1 not available - comparison will be limited')
elif not fsdp2_available:
    print('\\n⚠️  FSDP2 not available - comparison will be limited')
else:
    print('\\n✅ Both FSDP1 and FSDP2 are available for comparison')
" 2>/dev/null

ENV_CHECK_RESULT=$?

if [ $ENV_CHECK_RESULT -ne 0 ]; then
    print_status "ERROR" "Environment check failed"
    exit 1
fi

print_status "SUCCESS" "Environment check passed"
echo

# GPU detection and configuration
echo "🖥️  GPU Configuration"
echo "────────────────────"

GPU_COUNT=0
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
    if [ $GPU_COUNT -gt 0 ]; then
        print_status "INFO" "Found $GPU_COUNT GPU(s)"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -3
    else
        print_status "WARNING" "nvidia-smi found but no GPUs detected"
    fi
else
    print_status "WARNING" "nvidia-smi not found - GPU detection skipped"
fi

echo

# Single GPU test
echo "🔬 Single GPU Comprehensive Test"
echo "────────────────────────────────"
print_status "INFO" "Running detailed parameter/gradient access comparison..."
echo "Command: python3 fsdp_gradient_comparison.py"
echo

# Redirect stderr (2) to /dev/null to hide all warnings and library logs
python3 fsdp_gradient_comparison.py 2>/dev/null | tee single_gpu_output.log
SINGLE_GPU_RESULT=${PIPESTATUS[0]}

echo
if [ $SINGLE_GPU_RESULT -eq 0 ]; then
    print_status "SUCCESS" "Single GPU test completed successfully"
else
    print_status "ERROR" "Single GPU test failed with exit code $SINGLE_GPU_RESULT"
fi

# Multi-GPU test (if available)
echo
echo "🚀 Multi-GPU Distributed Test"
echo "─────────────────────────────"

if [ "$GPU_COUNT" -gt 1 ]; then
    print_status "INFO" "Running multi-GPU distributed comparison..."

    # Test with 2 GPUs
    if [ "$GPU_COUNT" -ge 2 ]; then
        echo "Command: torchrun --nproc_per_node=2 fsdp_gradient_comparison.py"
        echo

        # Redirect stderr to /dev/null
        torchrun --nproc_per_node=2 fsdp_gradient_comparison.py 2>/dev/null | tee multi_gpu_2_output.log
        MULTI_GPU_2_RESULT=${PIPESTATUS[0]}

        if [ $MULTI_GPU_2_RESULT -eq 0 ]; then
            print_status "SUCCESS" "2-GPU test completed successfully"
        else
            print_status "ERROR" "2-GPU test failed with exit code $MULTI_GPU_2_RESULT"
        fi
    fi

    # Test with 4 GPUs if available
    if [ "$GPU_COUNT" -ge 4 ]; then
        echo
        echo "Command: torchrun --nproc_per_node=4 fsdp_gradient_comparison.py"
        echo

        # Redirect stderr to /dev/null
        torchrun --nproc_per_node=4 fsdp_gradient_comparison.py 2>/dev/null | tee multi_gpu_4_output.log
        MULTI_GPU_4_RESULT=${PIPESTATUS[0]}

        if [ $MULTI_GPU_4_RESULT -eq 0 ]; then
            print_status "SUCCESS" "4-GPU test completed successfully"
        else
            print_status "ERROR" "4-GPU test failed with exit code $MULTI_GPU_4_RESULT"
        fi
    fi
else
    print_status "WARNING" "Only $GPU_COUNT GPU available - skipping multi-GPU tests"
    MULTI_GPU_2_RESULT=0
    MULTI_GPU_4_RESULT=0
fi

# Results analysis
echo
echo "📊 Test Results Summary"
echo "──────────────────────"

# Extract key results from logs
if [ -f "single_gpu_output.log" ]; then
    echo "📋 Single GPU Test Results:"

    # Look for regression indicators
    if grep -q "FUNCTIONALITY REGRESSION DETECTED" single_gpu_output.log; then
        print_status "ERROR" "Functionality regression detected in single GPU test"
        echo "   Regression details:"
        grep -A 5 "FUNCTIONALITY REGRESSION DETECTED" single_gpu_output.log | sed 's/^/     /'
    elif grep -q "No significant regressions detected" single_gpu_output.log; then
        print_status "SUCCESS" "No significant regressions found in single GPU test"
    else
        print_status "WARNING" "Could not determine regression status from single GPU test"
    fi

    # Extract microbatch success rates
    if grep -q "Microbatch Success Rate" single_gpu_output.log; then
        echo "   Microbatch gradient collection rates:"
        grep "Microbatch Success Rate" single_gpu_output.log | sed 's/^/     /'
    fi
fi

# Multi-GPU results
if [ -f "multi_gpu_2_output.log" ] && [ $MULTI_GPU_2_RESULT -eq 0 ]; then
    echo
    echo "📋 2-GPU Test Results:"
    if grep -q "FUNCTIONALITY REGRESSION DETECTED" multi_gpu_2_output.log; then
        print_status "ERROR" "Functionality regression detected in 2-GPU test"
    elif grep -q "No significant regressions detected" multi_gpu_2_output.log; then
        print_status "SUCCESS" "No significant regressions found in 2-GPU test"
    fi
fi

# Final summary
echo
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                           FINAL SUMMARY                             ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"

echo
echo "🎯 Key Findings Expected:"
echo "  • FSDP1: Should allow individual microbatch gradient collection"
echo "  • FSDP2: May have limitations in gradient access with sync disabled"
echo "  • Parameter access: Both should work but may differ in implementation"
echo "  • Sync behavior: Critical difference in gradient collection capabilities"

echo
echo "📁 Log Files Generated:"
if [ -f "single_gpu_output.log" ]; then
    echo "  • single_gpu_output.log - Detailed single GPU test results"
fi
if [ -f "multi_gpu_2_output.log" ]; then
    echo "  • multi_gpu_2_output.log - 2-GPU distributed test results"
fi
if [ -f "multi_gpu_4_output.log" ]; then
    echo "  • multi_gpu_4_output.log - 4-GPU distributed test results"
fi

echo
echo "🔍 Analysis Tips:"
echo "  1. Look for 'FUNCTIONALITY REGRESSION DETECTED' in the output"
echo "  2. Check microbatch success rates between FSDP1 and FSDP2"
echo "  3. Pay attention to gradient access with/without sync"
echo "  4. Compare parameter accessibility patterns"

# Determine overall result
OVERALL_RESULT=0

if [ $SINGLE_GPU_RESULT -ne 0 ]; then
    OVERALL_RESULT=1
fi

if [ -f "single_gpu_output.log" ] && grep -q "FUNCTIONALITY REGRESSION DETECTED" single_gpu_output.log; then
    echo
    print_status "ERROR" "Regression confirmed: FSDP2 has reduced gradient access capabilities"
    OVERALL_RESULT=1
elif [ -f "single_gpu_output.log" ] && grep -q "No significant regressions detected" single_gpu_output.log; then
    echo
    print_status "SUCCESS" "No major regressions found between FSDP1 and FSDP2"
else
    echo
    print_status "WARNING" "Could not definitively determine regression status"
fi

echo
if [ $OVERALL_RESULT -eq 0 ]; then
    print_status "SUCCESS" "Comparison test suite completed successfully"
else
    print_status "ERROR" "Comparison test suite detected issues"
fi

echo
echo "🚀 For detailed analysis, review the generated log files"
echo "📖 This test helps identify FSDP2 regressions in gradient access patterns"

exit $OVERALL_RESULT
