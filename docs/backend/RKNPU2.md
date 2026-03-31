RKNPU2 Backend for Rockchip NPU
===============================

This backend provides acceleration for Rockchip NPUs (Neural Processing Units) found in RK3588, RK3588S, and RK3576 platforms.

Hardware Support
----------------

| SoC | NPU Cores | INT8 Performance | FP16 Performance |
|-----|-----------|------------------|-----------------|
| RK3588 | 3x ARM Mali-G610 | 6 TOPS | 1.5 TOPS |
| RK3588S | 2x ARM Mali-G610 | 4 TOPS | 1 TOPS |
| RK3576 | 3x ARM Mali-G610 | 6 TOPS | 1.5 TOPS |

**Note:** RK3576 support is currently a placeholder - the backend will detect it but use RK3588 configuration.

Prerequisites
-------------

### 1. Install RKNN Runtime

The Rockchip NPU requires the RKNN runtime library (`librknnrt.so`).

On Debian/Rockchip boards:
```bash
sudo apt-get install librknpu2-rk3588  # For RK3588
```

Or install manually from Rockchip's SDK.

### 2. CMA Configuration

The NPU requires contiguous memory allocation (CMA). Add to kernel cmdline:

```
cma=1024M
```

On RadxaOS/Rockbian, edit `/etc/kernel/cmdline` and run `u-boot-update`.

### 3. NPU Driver Version

RKLLM v1.2.3+ requires NPU driver 0.9.7 or later. Check version:

```bash
cat /sys/module/npu/version
```

Build
-----

```bash
mkdir build && cd build
cmake -DGGML_RKNPU2=ON -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

Usage
-----

### Basic Inference

```bash
# Single-core NPU
./build/bin/llama-cli -m model.gguf -p "Hello"

# Multi-core NPU (all available cores)
./build/bin/llama-cli -m model.gguf -p "Hello" --n-gpu-layers 99
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `RKNN_DEVICE` | Select NPU device | "RK3588" |
| `RKNN_CORE_MASK` | Mask for core allocation (hex) | "0xF" (all cores) |
| `RKNN_SPLIT_FACTOR` | Split factor for large models | 1 |
| `RKNPU_B_CACHE_SIZE` | Max cached B-matrix mem handles | 64 |
| `RKNPU_CTX_CACHE_SIZE` | Max cached RKNN matmul contexts | 64 |
| `HYBRID_PATTERN` | Quantization pattern for layers | (auto) |

### Core Selection

```bash
# Use only core 0
RKNN_CORE_MASK=0x1 ./build/bin/llama-cli -m model.gguf --n-gpu-layers 99

# Use cores 0 and 1
RKNN_CORE_MASK=0x3 ./build/bin/llama-cli -m model.gguf --n-gpu-layers 99
```

### Large Model Support (IOVA Exhaustion)

For models that exceed IOVA memory, use split factor:

```bash
# Split across multiple allocations
RKNN_SPLIT_FACTOR=4 ./build/bin/llama-cli -m large-model.gguf --n-gpu-layers 99
```

If repeated or broader hybrid runs still consume too much persistent RKNN/IOMMU space, also bound the runtime caches:

```bash
RKNPU_B_CACHE_SIZE=32 RKNPU_CTX_CACHE_SIZE=32 \
    ./build/bin/llama-cli -m large-model.gguf --n-gpu-layers 99
```

### Hybrid Quantization

Process different layers with different precisions:

```bash
# FP16 on NPU, INT8 on NPU, let CPU handle rest
HYBRID_PATTERN=FP16_STANDARD,INT8_STANDARD ./build/bin/llama-cli -m model.gguf --n-gpu-layers 99
```

Supported Quantization Types
----------------------------

### NPU-Accelerated (on-device)

| Type | Precision | Notes |
|------|----------|-------|
| FP16 | 16-bit float | Fastest on NPU |
| INT8 | 8-bit integer | Good balance |
| Q4_0 | 4-bit quantized | Lower memory |
| Q6_K | 6-bit quantized | Good quality |
| Q8_0 | 8-bit quantized | High quality |

### CPU-Only (not accelerated on NPU)

| Type | Notes |
|------|-------|
| IQ2_K, IQ3_K, IQ4_K, IQ5_K, IQ6_K | ik_llama IQK types |
| All other IQ* types | Handled by CPU backend |

The NPU backend explicitly rejects IQK quantization types because they have different block structures than standard GGML quants. The CPU backend handles these with its optimized kernels.

Architecture
------------

The RKNPU2 backend uses Rockchip's RKNN (Rockchip Neural Network) API for NPU operations:

1. **Quantization**: Weights quantized to NPU-compatible format (FP16/INT8)
2. **Memory Allocation**: DMA-Heap for zero-copy buffer sharing
3. **Matrix Multiplication**: `rknn_matmul_create` / `rknn_matmul_run`
4. **Multi-Core Segmentation**: Large matrices split across NPU cores

Key source files:
- `ggml-rknpu2.cpp` - Main backend implementation
- `rknpu2-configuration.cpp` - Device and pattern configuration
- `rknpu2-quantization.cpp` - Weight quantization
- `rknpu2-calibration.cpp` - Scale factor calibration
- `rknpu2-allocation.cpp` - DMA-Heap buffer management

Troubleshooting
---------------

### "Failed to find NPU device"

1. Check kernel cmdline has `cma=1024M`
2. Verify NPU driver is loaded: `lsmod | grep npu`
3. Check CMA size: `cat /proc/meminfo | grep Cma`

### "IOVA exhaustion" / Model fails to load

1. Increase CMA size (if kernel supports it)
2. Use `RKNN_SPLIT_FACTOR=2` or `4`
3. Lower `RKNPU_B_CACHE_SIZE` and `RKNPU_CTX_CACHE_SIZE` when repeated routed runs are consuming persistent RKNN state
4. Use quantization with smaller memory footprint

`RKNN_SPLIT_FACTOR` reduces allocation granularity, while the two cache limits reduce long-lived RKNN state retained across repeated or broader routed workloads.

### Slow inference on NPU

1. Ensure model uses NPU-compatible quant types
2. Try `RKNN_CORE_MASK=0xF` for maximum parallelism
3. Check CPU is not bottlenecking (disable with `--offload-type cpu`)

References
----------

- Rockchip RKNN SDK Documentation
- [rk-llama.cpp](https://github.com/KHAEntertainment/rk-llama.cpp) - Reference implementation
- [Radxa Rock 5C](https://wiki.radxa.com/Rock5/5c) - Compatible SBC
