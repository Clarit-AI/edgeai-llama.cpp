Performance Tuning for RKNPU2
=============================

This guide covers benchmarking and optimization strategies for the RKNPU2 backend.

Benchmarking
------------

### Basic Benchmark Command

```bash
./build/bin/llama-bench -m models/your-model.gguf -g 99
```

### Hybrid Manifest Attribution

For experiment runs that use a manifest-driven route policy, export the policy metadata so it shows up in benchmark output:

```bash
HYBRID_MANIFEST=/path/to/model.gguf.hybrid.json \
HYBRID_PROFILE=balanced \
HYBRID_STRICT=1 \
    ./build/bin/llama-bench -m models/your-model.gguf -g 99
```

The benchmark output now includes `hybrid_manifest`, `hybrid_profile`, and `hybrid_strict` fields so CPU, legacy RKNPU2, and manifest-driven runs can be compared side by side without guessing which policy was active.

### Recommended Comparison Matrix

For each target model, capture these three runs:

```bash
# CPU baseline
./build/bin/llama-bench -m models/your-model.gguf -g 0

# Current RKNPU2 baseline
./build/bin/llama-bench -m models/your-model.gguf -g 99

# Manifest-driven hybrid policy
HYBRID_MANIFEST=/path/to/model.gguf.hybrid.json \
HYBRID_PROFILE=balanced \
    ./build/bin/llama-bench -m models/your-model.gguf -g 99
```

Use the same comparison for `llama-sweep-bench` when you want to see whether the policy helps prompt processing, token generation, or both across different KV sizes.

### Expected Performance (RK3588, 3 NPU cores)

| Model Size | Quantization | Tokens/sec (NPU) | Tokens/sec (CPU) |
|------------|--------------|------------------|------------------|
| 3B | FP16 | 25-35 | 12-15 |
| 3B | INT8 | 30-40 | 15-18 |
| 7B | FP16 | 12-18 | 5-7 |
| 7B | INT8 | 15-22 | 7-10 |
| 13B | FP16 | 6-10 | 2-3 |
| 13B | INT8 | 8-12 | 3-5 |

**Note:** Actual performance varies based on model architecture, sequence length, and system load.

### Measuring NPU vs CPU Speedup

```bash
# CPU only (no GPU offload)
time ./build/bin/llama-cli -m model.gguf -p "Hello" --n-gpu-layers 0

# NPU (full offload)
time ./build/bin/llama-cli -m model.gguf -p "Hello" --n-gpu-layers 99
```

Multi-Core Scaling
------------------

### Core vs Performance

| Cores | Relative Performance | Power Draw |
|-------|---------------------|------------|
| 1 | 1.0x (baseline) | Lowest |
| 2 | 1.6-1.8x | Medium |
| 3 | 2.2-2.6x | Highest |

**Observation:** Scaling is not linear due to:
- Memory bandwidth saturation
- Scheduling overhead
- Model tensor granularity

### Recommended Core Configurations

| Use Case | RKNN_CORE_MASK | Notes |
|----------|---------------|-------|
| Background inference | `0x1` | Single core, minimal power |
| Interactive use | `0x3` | Two cores, balanced |
| Batch processing | `0x7` | All cores, maximum speed |

Quantization Trade-offs
----------------------

### Precision vs Speed

```
FP16 ────────────────> INT8 ────────────────> Q4_0
 fastest               |                     |
 memory usage: high    |                     |
 quality: highest      v                     v
                  Q6_K                   Q8_0
                      |                     |
                      |        quality: acceptable
                      v
                  Q4_0/K
```

### Recommended Quantization by Use Case

| Use Case | Recommended | Reason |
|----------|-------------|--------|
| High quality | FP16, INT8 | Best accuracy |
| Balanced | Q6_K, Q8_0 | Good speed + quality |
| Low memory | Q4_0 | Smallest memory |
| IQK models | IQ2_K+ | ik_llama optimizations (CPU) |

IOVA and Memory Management
--------------------------

### Understanding IOVA Exhaustion

The NPU uses IOVA (I/O Virtual Address) space for DMA transfers. This space is limited and shared with other drivers.

**Symptoms:**
- Model fails to load
- "IOVA allocation failed"
- Partial output before crash

### Mitigation Strategies

1. **Increase CMA** (most effective):
   ```
   cma=1024M
   ```

2. **Use split factor**:
   ```bash
   RKNN_SPLIT_FACTOR=2 ./build/bin/llama-cli -m model.gguf -ngl 99
   ```

3. **Quantize to smaller format**:
   - FP16 → INT8 saves ~50% memory
   - INT8 → Q4_0 saves ~75% memory

4. **Reduce batch size** (in llama.cpp params):
   ```bash
   -c 512  # instead of default 2048
   ```

Performance Profiling
---------------------

### Identifying Bottlenecks

1. **CPU bottleneck**: High CPU usage, low NPU utilization
   - Solution: Ensure `--n-gpu-layers 99` is set

2. **NPU bottleneck**: Low CPU usage, high NPU utilization
   - Solution: Model is compute-bound on NPU, try different quant

3. **Memory bottleneck**: System RAM nearly full
   - Solution: Use smaller quant or enable swap

### NPU Utilization Check

```bash
# Monitor NPU temperature (should stay below 80°C)
cat /sys/class/devfreq/*/temperature

# Check NPU frequency
cat /sys/class/devfreq/*/cur_freq
```

Tips for Maximum Performance
----------------------------

1. **Use appropriate quantization**
   - NPU excels at FP16/INT8
   - Don't use FP32 on NPU

2. **Match context size to workload**
   - Short prompts: smaller context saves memory
   - Long context: increase CMA and use split factor

3. **Disable CPU offloading for NPU layers**
   ```bash
   # Correct: all layers on NPU
   --n-gpu-layers 99

   # Avoid: mixed NPU/CPU causes overhead
   --n-gpu-layers 32
   ```

4. **Prefer longer batches**
   - NPU is optimized for larger matrix operations
   - Single-token generation is CPU-bound

5. **Keep system cool**
   - NPU throttles at high temperatures
   - Ensure adequate heatsink/airflow

Hardware-Specific Notes
----------------------

### RK3588 vs RK3588S

| Aspect | RK3588 | RK3588S |
|--------|--------|---------|
| NPU cores | 3 | 2 |
| Max performance | 100% | ~65% |
| Power consumption | Higher | Lower |
| Cooling requirement | Active recommended | Passive possible |

### RK3576

Currently uses RK3588 configuration as placeholder. Actual performance may vary.

Acknowledgements
---------------

Performance characterization based on testing with:
- Radxa Rock 5C (RK3588S)
- Radxa Rock 5B (RK3588)
- RadxaOS Debian 12
- NPU driver 0.9.8
