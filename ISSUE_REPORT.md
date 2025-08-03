# FSDP2 Regression: Individual Microbatch Gradients Inaccessible

## 🚨 Problem Summary
FSDP2 completely fails to provide access to individual microbatch gradients, while FSDP1 allows full access. This blocks gradient-based analysis workflows.

## 📊 Test Results
**Environment:** PyTorch 2.9.0a0+git55ff4f8
**Test Coverage:** Single GPU + Multi-GPU (2/4 GPUs)

```
FSDP1 Individual Microbatch Success Rate: 100.0% ✅
FSDP2 Individual Microbatch Success Rate: 0.0%   ❌
```

## 🔍 Technical Details

### FSDP1 (Working)
```python
# Individual microbatch gradient collection works
optimizer.zero_grad()
loss.backward()
grad = param.grad  # ✅ Available: shape=torch.Size([1050624]), norm=0.007508

# Also works with no_sync()
with model.no_sync():
    loss.backward()
    grad = param.grad  # ✅ Still available
```

### FSDP2 (Now Working)
```python
# Individual microbatch gradient collection fails
model.set_requires_gradient_sync(False)  # FSDP2 equivalent of no_sync()
optimizer.zero_grad()
loss.backward()
grad = param.grad  # ❌ None - completely inaccessible

# Result: 0/8 parameters have accessible gradients
```

## 🧪 Attempted Solutions (All Failed)

### 1. Direct Parameter Access
```python
# Tried multiple gradient attributes
param.grad          # ❌ None
param.main_grad     # ❌ None
param.cuda_grad     # ❌ None
```

### 2. DTensor Conversion
```python
# Attempted DTensor to local conversion
if isinstance(grad, DTensor):
    local_grad = grad.to_local()  # ❌ DTensor conversion failed
```

### 3. FSDP2 State Access
```python
# Tried accessing internal FSDP2 state
model.set_requires_gradient_sync(False)
# ❌ No gradients exposed at parameter level
```

### 4. Hook-based Collection
```python
# Attempted gradient hooks
def capture_grad(grad):
    captured_grads.append(grad)
param.register_hook(capture_grad)  # ❌ Hooks not triggered
```

## 📈 Impact Assessment

- **Blocks gradient analysis**: Cannot access individual microbatch gradients
- **Migration barrier**: Prevents FSDP1 → FSDP2 migration for gradient-dependent workflows
- **Research impact**: Affects gradient-based optimization techniques

## 🔄 Reproduction Steps

### Quick Test
```bash
python3 fsdp_gradient_comparison.py
```

### Multi-GPU Test
```bash
torchrun --nproc_per_node=2 fsdp_gradient_comparison.py
```

### Expected Output
```
🚨 FUNCTIONALITY REGRESSION DETECTED:
  - FSDP2 has reduced microbatch gradient collection capability
  - FSDP2 has issues with gradient access when sync is disabled
```

## 💡 Expected Behavior
FSDP2 should provide the same gradient access capabilities as FSDP1, allowing individual microbatch gradients to be collected when gradient sync is disabled via `set_requires_gradient_sync(False)`.

## 🎯 Request
1. **Restore gradient access** in FSDP2 equivalent to FSDP1's capabilities
2. **Provide API** for accessing individual microbatch gradients
