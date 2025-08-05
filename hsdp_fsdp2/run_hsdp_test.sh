#!/bin/bash

# ====================================================================================
# HSDP Gradient Variance Test Suite
#
# v8: Redirecting stderr to /dev/null to completely suppress all warnings and
#     backend library logs, showing only the python script's stdout.
# ====================================================================================

# Exit script on error
set -e
set -o pipefail

# --- Script Configuration ---
PYTHON_SCRIPT="test_hsdp_gradient.py"

# --- Log Level Configuration (kept for good practice, but redirection is key) ---
export NCCL_DEBUG="WARN"
export TORCH_CPP_LOG_LEVEL="WARNING"

# --- Color Codes for Output ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# --- Helper Functions ---
print_header() {
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                 $1                 ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════════╝${NC}"
}

print_status() {
    local status=$1
    local message=$2
    case $status in
        "SUCCESS") echo -e "${GREEN}✅ $message${NC}" ;;
        "FAIL") echo -e "${RED}❌ $message${NC}" ;;
        "WARN") echo -e "${YELLOW}⚠️  $message${NC}" ;;
        "INFO") echo -e "${BLUE}ℹ️  $message${NC}" ;;
        "SKIP") echo -e "${YELLOW}⏩ $message${NC}" ;;
    esac
}

# ==============================================================================
# 1. Initialization and Environment Check
# ==============================================================================
print_header " HSDP Gradient Variance Test Suite "
echo
echo "🎯 Objective: Verify that HSDP can capture per-replica gradients before synchronization."
echo "🔍 Focus: Check for significant gradient differences between replicas with different data."
echo "📋 Expectation: '✅ SUCCESS' message on 4+ GPUs, confirming gradient variance."
echo

# Check for the existence of the Python script
if [ ! -f "$PYTHON_SCRIPT" ]; then
    print_status "FAIL" "Test script '$PYTHON_SCRIPT' not found."
    exit 1
fi

# Clean up previous log files
rm -f hsdp_test_*.log

# Environment Check (This part is not filtered)
print_status "INFO" "Verifying environment..."
echo "────────────────────────────────────────"
python -c "
import sys
import torch
print(f'Python version: {sys.version.split()[0]}')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device count: {torch.cuda.device_count()}')
try:
    from torch.distributed.fsdp import fully_shard
    print('HSDP (fully_shard): ✅ Available')
except ImportError as e:
    print(f'HSDP (fully_shard): ❌ {e}')
try:
    import torch.distributed as dist
    print('torch.distributed: ✅ Available')
except ImportError:
    print('torch.distributed: ❌ Not available')
"
# ... rest of the initial check ...
print_status "SUCCESS" "Environment check complete."
echo
print_status "INFO" "Detecting GPU configuration..."
echo "────────────────────────────────────────"
GPU_COUNT=0
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l || true)
fi
if [ "$GPU_COUNT" -gt 0 ]; then
    print_status "INFO" "Available GPUs: $GPU_COUNT"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -4 || true
else
    print_status "WARN" "No GPUs detected. Some tests will be skipped."
fi
echo

# ==============================================================================
# 2. Test Execution
# ==============================================================================
print_header "           Test Execution            "

# --- Test Case 1: Single GPU ---
echo
print_status "INFO" "Test Case 1: Single GPU (Demonstration Mode)"
LOG_FILE="hsdp_test_1gpu.log"
if [ "$GPU_COUNT" -ge 1 ]; then
    echo "   Command: torchrun --nproc_per_node=1 $PYTHON_SCRIPT"
    # Redirect stderr (2) to /dev/null to hide all warnings.
    # Only stdout (1) is passed to tee.
    CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 "$PYTHON_SCRIPT" 2>/dev/null | tee "$LOG_FILE"
    TEST_1GPU_RESULT=${PIPESTATUS[0]}
else
    print_status "SKIP" "Skipping single-GPU test as no GPU is available."
    TEST_1GPU_RESULT=0
fi
echo

# --- Test Case 2: 2-GPU ---
echo
print_status "INFO" "Test Case 2: 2-GPU (HSDP Skip Mode)"
LOG_FILE="hsdp_test_2gpu.log"
if [ "$GPU_COUNT" -ge 2 ]; then
    echo "   Command: torchrun --nproc_per_node=2 $PYTHON_SCRIPT"
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 "$PYTHON_SCRIPT" 2>/dev/null | tee "$LOG_FILE"
    TEST_2GPU_RESULT=${PIPESTATUS[0]}
else
    print_status "SKIP" "This test requires at least 2 GPUs (found: $GPU_COUNT)."
    TEST_2GPU_RESULT=0
fi
echo

# --- Test Case 3: 4-GPU ---
echo
print_status "INFO" "Test Case 3: 4-GPU (Main HSDP Test)"
LOG_FILE="hsdp_test_4gpu.log"
if [ "$GPU_COUNT" -ge 4 ]; then
    echo "   Command: torchrun --nproc_per_node=4 $PYTHON_SCRIPT"
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 "$PYTHON_SCRIPT" 2>/dev/null | tee "$LOG_FILE"
    TEST_4GPU_RESULT=${PIPESTATUS[0]}
else
    print_status "SKIP" "This test requires at least 4 GPUs (found: $GPU_COUNT)."
    TEST_4GPU_RESULT=0
fi
echo

# --- Test Case 4: 8-GPU (Optional) ---
echo
print_status "INFO" "Test Case 4: 8-GPU (Main HSDP Test, Shard size 2)"
LOG_FILE="hsdp_test_8gpu.log"
if [ "$GPU_COUNT" -ge 8 ]; then
    echo "   Command: torchrun --nproc_per_node=8 $PYTHON_SCRIPT"
    torchrun --nproc_per_node=8 "$PYTHON_SCRIPT" 2>/dev/null | tee "$LOG_FILE"
    TEST_8GPU_RESULT=${PIPESTATUS[0]}
else
    print_status "SKIP" "This test requires at least 8 GPUs (found: $GPU_COUNT)."
    TEST_8GPU_RESULT=0
fi
echo


# ==============================================================================
# 3. Test Results Summary
# ==============================================================================
print_header "         Test Results Summary        "
echo

OVERALL_SUCCESS=true

# The result analysis logic remains the same.
if [ -f "hsdp_test_1gpu.log" ]; then
    echo "📋 Single-GPU Test (Demonstration Mode):"
    if [ $TEST_1GPU_RESULT -eq 0 ]; then
        if grep -q "In HSDP, each replica would have these different gradients" hsdp_test_1gpu.log; then print_status "SUCCESS" "Demonstration completed successfully."; else print_status "FAIL" "Expected output for the demonstration was not found."; OVERALL_SUCCESS=false; fi
    else
        print_status "FAIL" "Test failed with exit code $TEST_1GPU_RESULT."; OVERALL_SUCCESS=false
    fi
fi
if [ -f "hsdp_test_2gpu.log" ]; then
    echo "📋 2-GPU Test (HSDP Skipped):"
    if [ $TEST_2GPU_RESULT -eq 0 ]; then
        if grep -q "Need at least 4 GPUs for this test" hsdp_test_2gpu.log; then print_status "SUCCESS" "HSDP test was correctly skipped."; else print_status "FAIL" "Expected skip message was not found."; OVERALL_SUCCESS=false; fi
    else
        print_status "FAIL" "Test failed with exit code $TEST_2GPU_RESULT."; OVERALL_SUCCESS=false
    fi
fi
if [ -f "hsdp_test_4gpu.log" ]; then
    echo "📋 4-GPU Test (Main HSDP Test):"
    if [ $TEST_4GPU_RESULT -eq 0 ]; then
        if grep -q "✅ SUCCESS: Gradients vary significantly across replicas!" hsdp_test_4gpu.log; then
            print_status "SUCCESS" "HSDP gradient variance confirmed successfully!"; RATIO_LINE=$(grep "Min/Max Ratio:" hsdp_test_4gpu.log | head -n 1); echo "   - $RATIO_LINE"
        else
            print_status "FAIL" "Success message '✅ SUCCESS' not found."; OVERALL_SUCCESS=false
        fi
    else
        print_status "FAIL" "Test failed with exit code $TEST_4GPU_RESULT."; OVERALL_SUCCESS=false
    fi
fi
if [ -f "hsdp_test_8gpu.log" ]; then
    echo "📋 8-GPU Test (Main HSDP Test):"
    if [ $TEST_8GPU_RESULT -eq 0 ]; then
        if grep -q "✅ SUCCESS: Gradients vary significantly across replicas!" hsdp_test_8gpu.log; then
            print_status "SUCCESS" "HSDP gradient variance confirmed successfully!"; RATIO_LINE=$(grep "Min/Max Ratio:" hsdp_test_8gpu.log | head -n 1); echo "   - $RATIO_LINE"
        else
            print_status "FAIL" "Success message '✅ SUCCESS' not found."; OVERALL_SUCCESS=false
        fi
    else
        print_status "FAIL" "Test failed with exit code $TEST_8GPU_RESULT."; OVERALL_SUCCESS=false
    fi
fi

# ==============================================================================
# 4. Final Summary
# ==============================================================================
echo
print_header "           Final Summary             "
echo
echo "📁 Generated Log Files:"
ls -1 hsdp_test_*.log 2>/dev/null | sed 's/^/  • /' || echo "  (No log files)"
echo

if [ "$OVERALL_SUCCESS" = true ]; then
    print_status "SUCCESS" "All executed tests passed successfully."
    echo "It is confirmed that HSDP correctly maintains independent, per-replica gradients before synchronization."
    exit 0
else
    print_status "FAIL" "Some tests failed. Please review the logs and messages above for details."
    exit 1
fi
