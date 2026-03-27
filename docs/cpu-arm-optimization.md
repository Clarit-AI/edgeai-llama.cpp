# CPU & ARM Performance Optimization Guide

This guide covers how to achieve maximum inference performance on CPU-only or ARM devices (including Rockchip RK3588/RK3576, Apple Silicon, and other ARM boards) when using ik_llama.cpp.

## Contents

1. [Quick Start](#quick-start)
2. [Quantization Selection](#quantization-selection)
3. [Optimal Inference Flags](#optimal-inference-flags)
4. [Build Optimization for ARM](#build-optimization-for-arm)
5. [Performance Benchmarks](#performance-benchmarks)
6. [Quantizing Your Own Models](#quantizing-your-own-models)
7. [Troubleshooting](#troubleshooting)

---

## Quick Start

For maximum performance on CPU/ARM devices, use **K-format quantizations** (IQ2_K, IQ3_K, IQ4_K) with R4 variants when available, combined with these optimal flags:

```bash
./build/bin/llama-server \
  -m model-IQ4_K_R4.gguf \
  -t 4 -c 4096 \
  -fa -fmoe \
  -ctk q8_0 -ctv q8_0 \
  -b 2048 -ub 2048 \
  -rtr
```

### What Each Flag Does

| Flag | Purpose | Benefit |
|------|---------|---------|
| `-fa` | Flash Attention | Major speedup for prompt processing |
| `-fmoe` | Fused MoE operations | 2-30% speedup for MoE models |
| `-ctk q8_0 -ctv q8_0` | KV cache quantization | Reduces memory usage, improves speed |
| `-b 2048 -ub 2048` | Larger batch sizes | Better prompt processing throughput |
| `-rtr` | Runtime repacking | Enables R4 variant performance gains |

---

## Quantization Selection

### K Formats: Best for CPU/ARM

The **K quantization formats** (IQ2_K, IQ3_K, IQ4_K, IQ5_K) are specifically optimized for CPU performance and provide **4-7x speedups** over standard i-quants on ARM NEON.

**Why K formats excel on ARM:**
- Use small lookup tables (2x4 to 2x64 entries) that fit efficiently in ARM NEON SIMD registers
- Non-linear mapping avoids the large codebook lookups that slow down traditional i-quants
- No complex Trellis generation that punishes CPU performance

### Avoid KT Formats on CPU/ARM

The **Trellis (KT) formats** (IQ2_KT, IQ3_KT, IQ4_KT) have **"pretty bad" CPU performance** on ARM. They use pseudo-random value generation that is computationally expensive on CPU.

> "The `*_KT` quants are very slow on my M2-Max CPU... Q4_0 is 3x faster in TG" - [Discussion 556](https://github.com/ikawrakow/ik_llama.cpp/discussions/556)

**When KT makes sense:**
- CUDA/GPU inference where Trellis overhead is worth the quality gain
- When quantization time is not a concern (KT quantizes ~5x slower)

### Avoid Legacy K Formats on ARM

Legacy k-quants (Q4_K_M, Q4_K_S, etc.) from mainline llama.cpp are **not optimized** for ARM NEON and perform worse than IQ4_XS.

### Quantization Format Reference

| Format | Bits/Weight | Size (8B model) | ARM Performance | Best Use Case |
|--------|-------------|-----------------|-----------------|---------------|
| **IQ2_KS** | 2.19 bpw | ~2.2 GB | Good | Maximum compression |
| **IQ2_XXS** | 2.06 bpw | ~2.1 GB | Good | Extreme compression |
| **IQ3_K** | 3.44 bpw | ~3.5 GB | **Best** | Balance of size/quality/speed |
| **IQ3_K_R4** | 3.44 bpw | ~3.5 GB | **1.7x faster** | Best speed on ARM |
| **IQ4_XS** | 4.25 bpw | ~4.1 GB | Good | Good quality, available now |
| **IQ4_K** | 4.5 bpw | ~4.5 GB | **Best** | Better quality |
| **IQ4_K_R4** | 4.5 bpw | ~4.5 GB | **1.86x faster** | Best speed on ARM |
| **IQ5_K** | 5.5 bpw | ~5.5 GB | Good | Higher quality |
| **Q4_K_M** | ~4.5 bpw | ~4.4 GB | Poor | **Avoid on ARM** |
| **IQ2_KT** | 2.13 bpw | ~2.1 GB | **Bad** | **Avoid on ARM** |
| **IQ3_KT** | 3.13 bpw | ~3.1 GB | **Bad** | **Avoid on ARM** |

### Size Compression Examples

| From | To | Original Size | Compressed Size | Reduction |
|------|----|---------------|-----------------|-----------|
| IQ4_XS (4.25 bpw) | IQ3_XXS (3.06 bpw) | 8.74 GB | ~6.31 GB | 27.8% |
| IQ4_XS (4.25 bpw) | IQ3_K (3.44 bpw) | 8.74 GB | ~7.08 GB | 19% |

---

## Optimal Inference Flags

### Flash Attention (`-fa`)

Enables Flash Attention for significant prompt processing speedup on both CPU and CUDA.

```bash
-fa
```

> Flash Attention works on ARM CPU and provides significant speedup, especially for prompt processing.

### Fused MoE (`-fmoe`)

Combines `ffn_up` and `ffn_gate` operations into a single fused operation for MoE models. Provides 2-30% speedup depending on model and hardware.

```bash
-fmoe
```

> **Does not force dense operation** - the model still uses sparse expert routing; only the computation is fused for efficiency.

### KV Cache Quantization (`-ctk q8_0 -ctv q8_0`)

Reduces memory bandwidth pressure by quantizing the KV cache. Particularly beneficial for long contexts.

```bash
-ctk q8_0 -ctv q8_0
```

> "at least on the CPUs that I have available, one gets better performance using `q8_0` KV cache... Not so much for short contexts, but quite noticeable for long contexts." - [Discussion 385](https://github.com/ikawrakow/ik_llama.cpp/discussions/385)

### Batch Sizes (`-b 2048 -ub 2048`)

Larger batches dramatically improve prompt processing speed.

```bash
-b 2048 -ub 2048
```

> "For better prompt processing speed, you should try to use larger `-b` and `-ub` (if VRAM permits). Given enough VRAM, best prompt processing speed for MoE models such as Qwen3-30B-A3B is obtained with `-b 4096 -ub 4096`" - [Discussion 591](https://github.com/ikawrakow/ik_llama.cpp/discussions/591)

### Runtime Repacking (`-rtr`)

Repacks tensors if an interleaved (R4) variant is available. This enables R4 performance gains without requiring a separately quantized model.

```bash
-rtr
```

> "You can repack either on the fly by adding `-rtr` to the command line, or offline like this: `./bin/llama-quantize --repack $model $repacked_model q8_0_r8`" - [Discussion 385](https://github.com/ikawrakow/ik_llama.cpp/discussions/385)

### Complete Example Commands

**For MoE models (e.g., DeepSeek, Qwen3-30B-A3B):**
```bash
./build/bin/llama-server \
  -m model-IQ4_K_R4.gguf \
  -t 4 -c 4096 \
  -fa -fmoe \
  -ctk q8_0 -ctv q8_0 \
  -b 2048 -ub 2048 \
  -rtr
```

**For dense models:**
```bash
./build/bin/llama-server \
  -m model-IQ4_K_R4.gguf \
  -t 4 -c 4096 \
  -fa \
  -ctk q8_0 -ctv q8_0 \
  -b 2048 -ub 2048 \
  -rtr
```

---

## Build Optimization for ARM

### Basic CPU Build

```bash
cmake -B build -DGGML_NATIVE=ON
cmake --build build --config Release -j$(nproc)
```

`GGML_NATIVE=ON` automatically detects ARM and enables NEON optimizations.

### ARM-Specific Architecture Flags (Optional)

For specific ARM processors, you may benefit from explicit architecture flags:

```bash
cmake -B build \
  -DGGML_NATIVE=ON \
  -DGGML_ARCH_FLAGS="-march=armv8.2-a+dotprod+fp16"
cmake --build build --config Release -j$(nproc)
```

| Architecture | Recommended Flags |
|--------------|-------------------|
| ARMv8.0 (older) | Basic NEON only |
| ARMv8.2-A | `-march=armv8.2-a+dotprod+fp16` |
| ARMv8.7-A+ (newer) | `-march=armv8.7-a+dotprod+fp16` |

> "The build system automatically detects ARM and enables NEON, but manual architecture flags may be needed for optimal performance on specific ARM processors." - [Issues 345](https://github.com/ikawrakow/ik_llama.cpp/issues/345)

---

## Performance Benchmarks

### Benchmark Methodology

The numbers in the tables below were collected under the following conditions:

- **Hardware**: Apple M2-Max (ARM NEON)
- **Model**: Kimi-VL-A3B (MoE model) quantized to the indicated format
- **Context length**: 512 tokens prompt + 128 token generation (`pp512` / `tg128`)
- **Batch size**: 2048 (`-b 2048 -ub 2048`)
- **Inference flags**: `-fa -fmoe -ctk q8_0 -ctv q8_0 -rtr`
- **Compilation**: Release build with ARM architecture flags (`-march=armv8.2-a+dotprod+fp16`)
- **Measurement**: Warm runs; throughput (t/s) is the mean over 3 consecutive runs
- **Reported values**: tokens per second (t/s); higher is better

These conditions may not match your hardware exactly. Use them as a relative reference when comparing formats or flags.

### ARM NEON Speedups (M2-Max)

| Format | Prompt Processing (pp512) | Token Gen (tg128) | vs Standard i-quant |
|--------|---------------------------|-------------------|---------------------|
| IQ2_XS | 60.60 t/s | 28.24 t/s | 2.66x / 1.55x |
| IQ3_S | 55.65 t/s | 20.33 t/s | 4.61x / 1.97x |
| IQ3_K | 54.94 t/s | - | 1.71x vs IQ3_S |
| **IQ3_K_R4** | **93.83 t/s** | - | **1.71x vs IQ3_K** |
| IQ4_K | 58.20 t/s | - | 1.44x vs IQ4_XS |
| **IQ4_K_R4** | **108.02 t/s** | - | **1.86x vs IQ4_K** |
| IQ4_XS | 134.02 t/s | 23.36 t/s | 3.54x vs mainline |

### K vs KT on ARM

| Format | Performance | Recommendation |
|--------|-------------|----------------|
| IQ3_K | Good | **Recommended** |
| IQ3_KT | "Pretty bad" | **Avoid on CPU/ARM** |

> "The `*_KT` quants are very slow on my M2-Max CPU, so it may not be worth putting the effort to make them work on a v8.0 phone." - [Discussion 556](https://github.com/ikawrakow/ik_llama.cpp/discussions/556)

### R4 Variant Speedups on ARM

| Format | Standard | R4 Variant | Speedup |
|--------|----------|-----------|---------|
| IQ3_K (ARM NEON) | 54.94 t/s | 93.83 t/s | **1.71x** |
| IQ4_K (ARM NEON) | 58.20 t/s | 108.02 t/s | **1.86x** |
| IQ5_K (ARM NEON) | - | - | **1.28x** |

### Expected Real-World Performance

For a Kimi-VL-A3B or similar MoE model on ARM:
- **Without optimizations**: ~6-7 tokens/s
- **With K formats + optimal flags**: ~10-11 tokens/s
- **With K_R4 formats + optimal flags**: ~15-18 tokens/s

---

## Quantizing Your Own Models

ik_llama.cpp includes `llama-quantize` which supports all IQK formats.

### Basic Quantization

```bash
# Quantize to IQ4_K
./build/bin/llama-quantize model-f16.gguf model-IQ4_K.gguf iq4_k

# Quantize to IQ3_K
./build/bin/llama-quantize model-f16.gguf model-IQ3_K.gguf iq3_k
```

### Quantize with Calibration Data (Better Quality)

```bash
# First, generate imatrix with a calibration dataset
./build/bin/llama-quantize --imatrix calibration_data.dat \
  model-f16.gguf model-IQ4_K_imatrix.gguf iq4_k
```

### Offline Repacking for R4 Variants

```bash
# Repack existing model to R4 (row-interleaved) format
./build/bin/llama-quantize --repack model-IQ4_K.gguf model-IQ4_K_R4.gguf iq4_k_r4
```

### All Available K Formats

```bash
iq2_ks    # 2.19 bpw - very low bitrate
iq2_k     # 2.38 bpw
iq3_k     # 3.44 bpw
iq3_k_r4  # 3.44 bpw - row-interleaved, faster on ARM
iq4_ks    # 4.25 bpw
iq4_k     # 4.50 bpw
iq4_k_r4  # 4.50 bpw - row-interleaved, faster on ARM
iq5_ks    # 5.25 bpw
iq5_k     # 5.50 bpw
iq5_k_r4  # 5.50 bpw - row-interleaved, faster on ARM
iq6_k     # 6.50 bpw - nearly lossless
```

### Custom Quantization Mixes

```bash
# Use different quant types for different tensors
./build/bin/llama-quantize \
  --custom-q "attn=iq5_k,ffn_gate=iq3_k" \
  model-f16.gguf model-custom.gguf iq3_k
```

---

## Troubleshooting

### Problem: KT formats are very slow on ARM

**Solution:** Switch to K format quants (IQ3_K instead of IQ3_KT, IQ4_K instead of IQ4_KT).

### Problem: R4 variants don't show speedup

**Solution:** Use `-rtr` flag to enable runtime repacking, or re-quantize with `--repack` option.

### Problem: Out of memory with large batch sizes

**Solution:** Reduce `-b` and `-ub` values (try `-b 512 -ub 512`) or use a smaller context size `-c`.

### Problem: Flash attention not working

**Solution:** Ensure you're using a model that supports FA. Also try explicit `-fa` flag (some models auto-enable, others don't).

### Problem: Model outputs gibberish

**Solution:** If using split mode/graph, try adding `-cuda graphs=0` to disable CUDA graphs.

### Model Compatibility Notes

> **Do not use** quantized models from Unsloth that have `_XL` in their name with `f16` tensors. These likely will not work with ik_llama.cpp.

---

## Additional Resources

- [Parameters Documentation](./parameters.md) - Complete list of all llama-cli/llama-server flags
- [RKNPU2 Backend](./backend/RKNPU2.md) - Rockchip NPU acceleration for RK3588/RK3576
- [Build Guide](./build.md) - Detailed build instructions for all platforms
- [Performance Guide](./npu/PERFORMANCE.md) - NPU-specific performance tuning
