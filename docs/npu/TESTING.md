# RKNPU2 Testing Guide

This guide provides a comprehensive test plan for verifying RKNPU2 backend functionality on Rockchip RK3588/RK3576 boards.

## Pre-Flight Checks

### System Readiness

```bash
# Check NPU driver loaded
lsmod | grep npu

# Check driver version (should be 0.9.7+)
cat /sys/module/npu/version

# Verify CMA size (should be 256MB+, 1GB recommended)
cat /proc/meminfo | grep -i cma

# Check RKNN runtime
ldconfig -p | grep rknn
```

### Build Verification

```bash
# Check version shows RKNPU2
./build/bin/llama-cli --version

# Check backend detection (dry run)
./build/bin/llama-cli --dry-run -m models/test.gguf -v 2>&1 | grep -iE "npu|cpu|backend"
```

---

## Functional Tests

### Test 1: Basic NPU Inference

```bash
./build/bin/llama-cli \
    -m models/your-model.gguf \
    -p "Hello, how are you?" \
    --n-gpu-layers 99
```

**Expected:** NPU initialization messages + tokens generated
**Pass criteria:** Completes without errors, NPU mentioned in logs

### Test 2: CPU vs NPU Performance

```bash
# CPU only baseline
time ./build/bin/llama-cli -m model.gguf -p "Hello" -ngl 0 2>&1 | tail -5

# NPU accelerated
time ./build/bin/llama-cli -m model.gguf -p "Hello" --n-gpu-layers 99 2>&1 | tail -5
```

**Expected:** NPU 2-3x faster for FP16/INT8
**Pass criteria:** NPU time < CPU time / 2

---

## Quantization Tests

### Test 3: NPU-Compatible Quantizations

These should run on the NPU:

```bash
# FP16 - fastest on NPU
./build/bin/llama-cli -m model-fp16.gguf -p "Testing FP16" --n-gpu-layers 99

# INT8 - good balance
./build/bin/llama-cli -m model-int8.gguf -p "Testing INT8" --n-gpu-layers 99

# Q8_0 - high quality
./build/bin/llama-cli -m model-q8_0.gguf -p "Testing Q8_0" --n-gpu-layers 99

# Q6_K - good quality/size
./build/bin/llama-cli -m model-q6_k.gguf -p "Testing Q6_K" --n-gpu-layers 99
```

**Pass criteria:** All complete successfully, NPU is used

### Test 4: CPU-Only Quantizations (IQK)

These should fall back to CPU gracefully:

```bash
./build/bin/llama-cli -m model-iq2_ks.gguf -p "Testing IQ2_KS" --n-gpu-layers 99
./build/bin/llama-cli -m model-iq3_ks.gguf -p "Testing IQ3_KS" --n-gpu-layers 99
./build/bin/llama-cli -m model-iq4_nl.gguf -p "Testing IQ4_NL" --n-gpu-layers 99
./build/bin/llama-cli -m model-iq5_ks.gguf -p "Testing IQ5_KS" --n-gpu-layers 99
```

**Expected:** Works but runs on CPU (no NPU messages)
**Pass criteria:** No errors, output is coherent

---

## Multi-Core Scaling Tests

### Test 5: Core Scaling

```bash
# Single core
RKNN_CORE_MASK=0x1 time ./build/bin/llama-cli -m model.gguf -p "Test" --n-gpu-layers 99

# Two cores
RKNN_CORE_MASK=0x3 time ./build/bin/llama-cli -m model.gguf -p "Test" --n-gpu-layers 99

# All cores
RKNN_CORE_MASK=0x7 time ./build/bin/llama-cli -m model.gguf -p "Test" --n-gpu-layers 99
```

**Expected:** ~1.6-1.8x speedup per additional core
**Pass criteria:** More cores = faster tokens/sec

---

## Large Model Tests

### Test 6: IOVA Exhaustion Handling

```bash
# With split factor 2
RKNN_SPLIT_FACTOR=2 ./build/bin/llama-cli -m large-model.gguf --n-gpu-layers 99

# With split factor 4
RKNN_SPLIT_FACTOR=4 ./build/bin/llama-cli -m huge-model.gguf --n-gpu-layers 99
```

**Pass criteria:** Large models load without "IOVA exhaustion" errors

### Test 6b: Bounded RKNN Cache Growth

```bash
# Keep cache sizes conservative during repeated routed runs
RKNPU_B_CACHE_SIZE=32 \
RKNPU_CTX_CACHE_SIZE=32 \
HYBRID_MANIFEST=/path/to/model.gguf.hybrid.json \
HYBRID_PROFILE=balanced \
HYBRID_STRICT=1 \
    ./build/bin/llama-cli -m model.gguf -p "Cache validation" --n-gpu-layers 99 -n 64
```

Repeat the same command several times or run a small loop to confirm cache reuse stays bounded instead of growing without limit.

**Pass criteria:** no crash, no stale-handle symptoms, and logs continue to show successful RKNN init / matmul execution across repeated runs.

---

## Benchmark Test

### Test 7: llama-bench

```bash
./build/bin/llama-bench -m models/your-model.gguf -g 99
```

Compare results against expected performance from [PERFORMANCE.md](./PERFORMANCE.md):

| Model Size | Quantization | Expected Tokens/sec (NPU) |
|------------|--------------|---------------------------|
| 3B | FP16 | 25-35 |
| 3B | INT8 | 30-40 |
| 7B | FP16 | 12-18 |
| 7B | INT8 | 15-22 |

---

## Hybrid Quantization Test

### Test 8: Mixed Precision

```bash
HYBRID_PATTERN=FP16_STANDARD,INT8_STANDARD \
    ./build/bin/llama-cli -m model.gguf -p "Test" --n-gpu-layers 99
```

**Pass criteria:** Completes successfully with mixed precision

### Test 9: Manifest-Driven Route Override

```bash
HYBRID_MANIFEST=/path/to/model.gguf.hybrid.json \
HYBRID_PROFILE=balanced \
HYBRID_DUMP_PLAN=1 \
    ./build/bin/llama-cli -m model.gguf -p "Test" --n-gpu-layers 99
```

**Expected:** manifest routes take precedence over legacy `HYBRID_PATTERN` decisions when a rule matches.
**Pass criteria:** logs show the manifest path/profile and at least one resolved route summary when the manifest matches a tensor.

### Test 10: Strict Manifest Validation

```bash
HYBRID_MANIFEST=/path/to/bad.hybrid.json \
HYBRID_STRICT=1 \
    ./build/bin/llama-cli -m model.gguf -p "Test" --n-gpu-layers 99
```

**Expected:** invalid manifest rules fail fast instead of silently falling back.
**Pass criteria:** process exits non-zero and the error names the bad rule or pipeline.

---

## Stress Tests

### Test 11: Extended Generation

```bash
# Long generation to test stability
./build/bin/llama-cli \
    -m model.gguf \
    -p "Write a detailed explanation of how neural networks work..." \
    -n 500 \
    --n-gpu-layers 99
```

**Pass criteria:** Completes 500 tokens without crash or corruption

### Test 12: Repeated Inference

```bash
for i in {1..10}; do
    echo "Run $i"
    ./build/bin/llama-cli -m model.gguf -p "Hello" -n 50 --n-gpu-layers 99
done
```

**Pass criteria:** All 10 runs complete successfully

For broader hybrid coverage, repeat Test 12 with:
- `HYBRID_MANIFEST`
- `HYBRID_PROFILE`
- `HYBRID_STRICT=1`
- conservative `RKNPU_B_CACHE_SIZE` and `RKNPU_CTX_CACHE_SIZE`

This is the preferred regression check for cache-growth fixes because it exercises many routed tensors without relying on a single one-off run.

---

## RK3588 Sync-Then-SSH Flow

When validating work that was implemented on a different machine, do not copy patches onto an older device checkout.

1. Sync the RK3588 repo to the same starting commit or branch base used locally.
2. Confirm remotes and branch ancestry match before building.
3. SSH into the device and build/test against that synced checkout.
4. If the device repo is significantly behind or has local drift, reconcile it first and only then continue runtime validation.

---

## Troubleshooting Flags

### Verbose Debugging

```bash
# Maximum verbosity
-v --verbosity 3

# Check backend selection
--dry-run -v

# Show manifest-driven plan summary from the backend layer
HYBRID_MANIFEST=/path/to/model.gguf.hybrid.json HYBRID_DUMP_PLAN=1 llama-cli -v

# Capture full log
./build/bin/llama-cli -m model.gguf -p "Test" --n-gpu-layers 99 2>&1 | tee run.log
```

### Common Issues

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| "Failed to find NPU device" | Driver not loaded | `lsmod \| grep npu` |
| "IOVA exhaustion" | CMA too small | Increase `cma=1024M` in cmdline |
| Slow on IQK | Expected - CPU only | Use FP16/INT8/Q8_0 for NPU |
| Garbled output | Vision model issue | Check mrope embedding format |
| Manifest ignored | No matching rule or bad env path | Check `HYBRID_MANIFEST`, `HYBRID_PROFILE`, and `HYBRID_STRICT` |

---

## Test Checklist

- [ ] System readiness (driver, CMA, RKNN runtime)
- [ ] Build verification
- [ ] Basic NPU inference
- [ ] CPU vs NPU performance comparison
- [ ] FP16/INT8/Q8_0/Q6_K quantizations
- [ ] IQK types fall back to CPU
- [ ] Multi-core scaling
- [ ] Large model with split factor
- [ ] llama-bench benchmark
- [ ] Extended generation (500 tokens)
- [ ] Repeated inference (10 runs)

---

## Expected Output Examples

### Successful NPU Init
```
RKNPU2: Initializing RKNN device...
RKNPU2: Device RK3588 selected
RKNPU2: Using 3 NPU cores
RKNPU2: Buffer allocated successfully
llama_model_load: loaded model
```

### Successful Inference
```
llama_model_load: loaded model in X.XX seconds
llama_generate: token generation speed: XX.XX tokens/sec
```
