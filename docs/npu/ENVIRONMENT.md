Environment Variables for RKNPU2
================================

The RKNPU2 backend can be configured via environment variables to control device selection, core allocation, memory management, and quantization patterns.

RKNN_DEVICE
------------

Selects which NPU device configuration to use.

| Value | Description |
|-------|-------------|
| `"RK3588"` | 3-core RK3588 (default) |
| `"RK3588S"` | 2-core RK3588S |
| `"RK3576"` | 3-core RK3576 (placeholder) |

**Example:**
```bash
RKNN_DEVICE="RK3588S" ./build/bin/llama-cli -m model.gguf --n-gpu-layers 99
```

RKNN_CORE_MASK
--------------

Hexadecimal mask specifying which NPU cores to use. Each bit represents a core.

| Mask | Cores Used |
|------|------------|
| `0x1` | Core 0 only |
| `0x3` | Cores 0 and 1 |
| `0x5` | Cores 0 and 2 |
| `0x7` | Cores 0, 1, and 2 |
| `0xF` | All cores (4-core, future) |

**Example:**
```bash
# Use cores 0 and 1 only
RKNN_CORE_MASK=0x3 ./build/bin/llama-cli -m model.gguf --n-gpu-layers 99

# Use only core 0 (minimum resource usage)
RKNN_CORE_MASK=0x1 ./build/bin/llama-cli -m model.gguf --n-gpu-layers 99
```

RKNN_SPLIT_FACTOR
-----------------

For models that exceed available IOVA (I/O Virtual Address) space, the backend can split tensor operations across multiple allocations.

| Value | Description |
|-------|-------------|
| `1` | No splitting (default) |
| `2` | Split across 2 allocations |
| `4` | Split across 4 allocations |
| Higher | More splits, more memory overhead |

**When to Use:**
- Models fail to load with "IOVA exhaustion" error
- Large models (7B+) with FP16 or INT8 quantization
- When CMA heap is limited

**Example:**
```bash
# Handle large model with 4-way split
RKNN_SPLIT_FACTOR=4 ./build/bin/llama-cli -m large-model.gguf --n-gpu-layers 99
```

**Performance Note:** Higher split factors add overhead from multiple DMA transfers. Use the minimum value that allows your model to load.

RKNPU_B_CACHE_SIZE
------------------

Caps the number of cached B-matrix RKNN memory handles created through `rknn_create_mem_from_fd()`.

| Value | Description |
|-------|-------------|
| `64` | Default, conservative bound |
| Higher | Fewer cache misses, larger persistent RKNN/IOMMU footprint |
| Lower | More aggressive reclamation, more handle churn |

**When to Use:**
- Broader hybrid runs slowly consume IOVA space across many unique routed tensors
- Repeated inference stays stable only when cached weight-handle growth is bounded
- You want to compare throughput vs persistent RKNN memory pressure

**Example:**
```bash
RKNPU_B_CACHE_SIZE=32 ./build/bin/llama-cli -m model.gguf --n-gpu-layers 99
```

RKNPU_CTX_CACHE_SIZE
--------------------

Caps the number of cached RKNN matmul contexts created for unique `(M, K, N segment, core, matmul type)` combinations.

| Value | Description |
|-------|-------------|
| `64` | Default, conservative bound |
| Higher | Fewer matmul context recreations |
| Lower | Smaller persistent RKNN context footprint |

**Example:**
```bash
RKNPU_CTX_CACHE_SIZE=32 ./build/bin/llama-cli -m model.gguf --n-gpu-layers 99
```

**Important:** `RKNN_SPLIT_FACTOR` and the cache limits address different problems.
- `RKNN_SPLIT_FACTOR` reduces per-allocation size for routed tensors.
- `RKNPU_B_CACHE_SIZE` and `RKNPU_CTX_CACHE_SIZE` bound persistent cache growth across repeated or broader routed runs.

HYBRID_PATTERN
--------------

Specifies a quantization pattern for processing different layers on the NPU. Format is comma-separated pipeline names.

| Pattern | Quantization | Hardware |
|---------|--------------|----------|
| `FP16_STANDARD` | FP16 | NPU |
| `INT8_STANDARD` | INT8 with scale factors | NPU |
| `Q4_0_STANDARD` | Q4_0 | NPU |
| `Q6_K_STANDARD` | Q6_K | NPU |
| `Q8_0_STANDARD` | Q8_0 | NPU |

**Example:**
```bash
# All layers FP16 on NPU
HYBRID_PATTERN=FP16_STANDARD ./build/bin/llama-cli -m model.gguf --n-gpu-layers 99

# Alternating pattern (if model has multiple weight tensors)
HYBRID_PATTERN=FP16_STANDARD,INT8_STANDARD,FP16_STANDARD,INT8_STANDARD \
    ./build/bin/llama-cli -m model.gguf --n-gpu-layers 99

# Three-way alternating
HYBRID_PATTERN=FP16_STANDARD,INT8_STANDARD,Q6_K_STANDARD \
    ./build/bin/llama-cli -m model.gguf --n-gpu-layers 99
```

**How It Works:**

The pattern cycles through weight tensors in sequence. For a model with 24 transformer layers:
- With `FP16_STANDARD,INT8_STANDARD`: layers 0,2,4,... use FP16; layers 1,3,5,... use INT8
- Pattern repeats if model has more tensors than pattern entries

**IQK Types Note:**

IQK quantization types (IQ2_K, IQ3_K, IQ4_K, IQ5_K, IQ6_K, etc.) are **always processed on CPU** regardless of HYBRID_PATTERN. The NPU backend explicitly rejects these types because their block structures differ from standard GGML quants.

Complete Example
----------------

```bash
# Production configuration for RK3588 with 3 cores
export RKNN_DEVICE="RK3588"
export RKNN_CORE_MASK="0x7"
export RKNN_SPLIT_FACTOR="1"
export RKNPU_B_CACHE_SIZE="64"
export RKNPU_CTX_CACHE_SIZE="64"
export HYBRID_PATTERN="FP16_STANDARD,INT8_STANDARD"

./build/bin/llama-cli \
    -m models/qwen3-8b-fp16.gguf \
    -p "Explain quantum computing in simple terms" \
    --n-gpu-layers 99 \
    -c 2048
```

Debugging
---------

To see what configuration is being used, run with verbose output or check initialization messages:

```bash
# NPU initialization messages appear at startup
./build/bin/llama-cli -m model.gguf --n-gpu-layers 99 2>&1 | grep -i rknpu
```

Common Issues
-------------

| Issue | Likely Cause | Solution |
|-------|--------------|----------|
| "Device not found" | RKNN runtime not installed | Install librknnrt.so |
| Only 1 core used | RKNN_CORE_MASK=0x1 | Set to 0x3 or 0x7 |
| OOM errors | Split factor too low | Increase to 2 or 4 |
| Repeated hybrid runs degrade or fail | RKNN caches keep growing | Lower `RKNPU_B_CACHE_SIZE` / `RKNPU_CTX_CACHE_SIZE` |
| Slow inference | Model has IQK types | CPU handles these, use different quant |
