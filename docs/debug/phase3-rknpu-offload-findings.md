# Phase 3 RKNPU Offload Findings

## Scope

This pass patched only [`src/llama.cpp`](/home/radxa/edgeai-llama.cpp-phase3/src/llama.cpp) and preserved existing local edits in:

- [`ggml/src/CMakeLists.txt`](/home/radxa/edgeai-llama.cpp-phase3/ggml/src/CMakeLists.txt)
- [`ggml/src/ggml-backend.cpp`](/home/radxa/edgeai-llama.cpp-phase3/ggml/src/ggml-backend.cpp)
- [`ggml/src/ggml-rknpu2/ggml-rknpu2.cpp`](/home/radxa/edgeai-llama.cpp-phase3/ggml/src/ggml-rknpu2/ggml-rknpu2.cpp)

## What Changed

Patched [`src/llama.cpp`](/home/radxa/edgeai-llama.cpp-phase3/src/llama.cpp) to wire `GGML_USE_RKNPU2` into the existing offload path:

- `llama_supports_gpu_offload()`
- `llama_get_device_count()`
- `llama_default_buffer_type_offload()`
- `llama_get_device_memory()`
- backend initialization in `llama_init_from_model()`

Then added a minimal hybrid model-tensor placement path:

- resolved hybrid routes are threaded through model params
- load-time model tensor defaults stay on CPU for hybrid+RKNPU runs
- only manifest-matched model tensors are routed to the RKNPU context
- a pre-allocation byte summary is printed:
  - total CPU bytes
  - total RKNPU bytes
  - count of tensors routed each way

## Commands Used

```bash
cmake --build build -j4 --target llama-cli

env HYBRID_MANIFEST=./examples/hybrid-manifests/dense-balanced.json \
    HYBRID_PROFILE=dense-balanced \
    HYBRID_STRICT=1 \
    ./build/bin/llama-cli \
    -m /home/radxa/hf_models/Qwen3-4B-Thinking-2507-Q8_0.gguf \
    --n-gpu-layers 99 -c 512 -b 32 -ub 32 --no-warmup -t 1 -tb 1 \
    --hybrid-dry-run \
    --hybrid-dump-plan /tmp/qwen-q8-hybrid-plan.json \
    -p 'phase 3 q8 hybrid smoke.' -n 1 \
    > /tmp/qwen-q8-hybrid-smoke.log 2>&1

env HYBRID_MANIFEST=./examples/hybrid-manifests/dense-balanced.json \
    HYBRID_PROFILE=dense-balanced \
    HYBRID_STRICT=1 \
    ./build/bin/llama-cli \
    -m /home/radxa/hf_models/qwen3-4b-q4_0.gguf \
    --n-gpu-layers 99 -c 512 -b 32 -ub 32 --no-warmup -t 1 -tb 1 \
    --hybrid-dry-run \
    --hybrid-dump-plan /tmp/qwen-q4-hybrid-plan.json \
    -p 'phase 3 q4 hybrid smoke.' -n 1 \
    > /tmp/qwen-q4-hybrid-smoke.log 2>&1
```

## Result

The original CPU-only gating problem is fixed.

Evidence from the new logs:

- The old warning about `not compiled with GPU offload support` is gone.
- RKNPU backend selection now happens:
  - `/tmp/qwen-q8-hybrid-smoke.log`: `RKNPU2: Using device 'RK3588' with core_mask=0, split_factor=1`
  - `/tmp/qwen-q8-hybrid-smoke.log`: `RKNPU: using device RKNPU - 0 MiB free`
- The same RKNPU activation appears in `/tmp/qwen-q4-hybrid-smoke.log`.

## New Blocker

Both repros now fail later during model buffer allocation:

- `/tmp/qwen-q8-hybrid-smoke.log`: `RKNPU_DMA_ALLOC: ioctl DMA_HEAP_IOCTL_ALLOC failed: Cannot allocate memory`
- `/tmp/qwen-q8-hybrid-smoke.log`: `Failed to allocate buffer type RKNPU`
- `/tmp/qwen-q8-hybrid-smoke.log`: `llama_model_load: error loading model: unable to allocate backend buffer`

The Q4_0 repro fails the same way.

This means the work has progressed from:

- "RKNPU is never selected"

to:

- "RKNPU is selected, but backend buffer allocation fails"

## Updated Selective Placement Results

After the minimal hybrid model-tensor placement implementation:

### Q8_0

`/tmp/qwen-q8-hybrid-smoke.log` now shows:

- `llm_load_tensors: hybrid model tensor placement summary before allocation:`
- `CPU   tensors = 291, bytes = 1745.24 MiB`
- `RKNPU tensors = 108, bytes = 2725.31 MiB`

The run still fails with:

- `RKNPU_DMA_ALLOC: ioctl DMA_HEAP_IOCTL_ALLOC failed: Cannot allocate memory`

Conclusion:

- the fix stopped blanket “everything on RKNPU” placement
- but the current manifest still routes enough Q8_0 tensors to RKNPU to require about 2.7 GiB of DMA-backed allocation
- that is still too large for the current device/runtime setup

### Q4_0

`/tmp/qwen-q4-hybrid-smoke.log` shows:

- `CPU   tensors = 399, bytes = 2558.38 MiB`
- `RKNPU tensors = 0, bytes = 0.00 MiB`

This matches the manifest, because `q4_0` is not in the route `quant_allow` list.

Conclusion:

- selective placement is working
- quant/shape gating is active
- the remaining problem is specifically the size of the Q8_0 tensor subset being routed to RKNPU, not broad unconditional load-time placement

## Next Debug Target

If continuing past the `src/llama.cpp`-only patch, inspect the RKNPU allocation and buffer-sizing path next, starting with:

- [`ggml/src/ggml-rknpu2/rknpu2-allocation.cpp`](/home/radxa/edgeai-llama.cpp-phase3/ggml/src/ggml-rknpu2/rknpu2-allocation.cpp)
- [`ggml/src/ggml-rknpu2/ggml-rknpu2.cpp`](/home/radxa/edgeai-llama.cpp-phase3/ggml/src/ggml-rknpu2/ggml-rknpu2.cpp)
- model buffer assignment/allocation in [`src/llama.cpp`](/home/radxa/edgeai-llama.cpp-phase3/src/llama.cpp)

Likely question:

- whether the current offload selection is trying to allocate a single oversized RKNPU DMA buffer for tensors that should remain on CPU under the hybrid plan

## Reduced Q8_0 Manifest Tuning

This pass kept the current selective-placement code unchanged and tuned only manifests.

One parser constraint surfaced immediately:

- hybrid manifests currently only accept `dense-balanced`, `dense-cpu-only`, and `moe-balanced`
- that means reduced test manifests must keep `profile: "dense-balanced"` even when the file name is new

### Experiment 1: Late FFN Only

Created [`examples/hybrid-manifests/dense-q8-late-ffn.json`](/home/radxa/edgeai-llama.cpp-phase3/examples/hybrid-manifests/dense-q8-late-ffn.json).

Diff versus the broad balanced route:

```diff
-      "name": "dense-backbone-balanced",
-      "match": "^blk\\.[0-9]+\\.(attn|ffn_(up|down|gate))\\.",
+      "name": "dense-late-ffn-q8",
+      "match": "^blk\\.(20|21|22|23|24|25|26|27)\\.ffn_(up|down|gate)\\.",
```

Command used:

```bash
env HYBRID_MANIFEST=./examples/hybrid-manifests/dense-q8-late-ffn.json \
    HYBRID_PROFILE=dense-balanced \
    HYBRID_STRICT=1 \
    ./build/bin/llama-cli \
    -m /home/radxa/hf_models/Qwen3-4B-Thinking-2507-Q8_0.gguf \
    --n-gpu-layers 99 -c 512 -b 32 -ub 32 --no-warmup -t 1 -tb 1 \
    --hybrid-dry-run \
    --hybrid-dump-plan /tmp/qwen-q8-dense-q8-late-ffn.json \
    -p 'phase 3 q8 reduced hybrid smoke.' -n 1 \
    > /tmp/qwen-q8-dense-q8-late-ffn.log 2>&1
```

Placement summary:

- CPU tensors = 375, bytes = 3864.93 MiB
- RKNPU tensors = 24, bytes = 605.62 MiB

Result:

- model load did not complete
- 1-token inference did not run
- failure remained `RKNPU_DMA_ALLOC: ioctl DMA_HEAP_IOCTL_ALLOC failed: Cannot allocate memory`

### Experiment 2: Tail `ffn_down` Only

Created [`examples/hybrid-manifests/dense-q8-tail-down.json`](/home/radxa/edgeai-llama.cpp-phase3/examples/hybrid-manifests/dense-q8-tail-down.json).

Diff versus the broad balanced route:

```diff
-      "name": "dense-backbone-balanced",
-      "match": "^blk\\.[0-9]+\\.(attn|ffn_(up|down|gate))\\.",
+      "name": "dense-tail-down-q8",
+      "match": "^blk\\.(24|25|26|27)\\.ffn_down\\.",
```

Command used:

```bash
env HYBRID_MANIFEST=./examples/hybrid-manifests/dense-q8-tail-down.json \
    HYBRID_PROFILE=dense-balanced \
    HYBRID_STRICT=1 \
    ./build/bin/llama-cli \
    -m /home/radxa/hf_models/Qwen3-4B-Thinking-2507-Q8_0.gguf \
    --n-gpu-layers 99 -c 512 -b 32 -ub 32 --no-warmup -t 1 -tb 1 \
    --hybrid-dry-run \
    --hybrid-dump-plan /tmp/qwen-q8-dense-q8-tail-down.json \
    -p 'phase 3 q8 reduced hybrid smoke.' -n 1 \
    > /tmp/qwen-q8-dense-q8-tail-down.log 2>&1
```

Placement summary:

- CPU tensors = 395, bytes = 4369.62 MiB
- RKNPU tensors = 4, bytes = 100.94 MiB

Result:

- model load did not complete
- 1-token inference did not run
- failure remained `RKNPU_DMA_ALLOC: ioctl DMA_HEAP_IOCTL_ALLOC failed: Cannot allocate memory`

### Recommendation

Manifest tuning is no longer the main limiter.

Evidence:

- broad balanced profile failed at 108 tensors / 2725.31 MiB on RKNPU
- late-FFN-only profile failed at 24 tensors / 605.62 MiB on RKNPU
- tail-`ffn_down` profile still failed at only 4 tensors / 100.94 MiB on RKNPU

Recommendation:

- stop manifest tuning for this branch and move to allocator investigation next
- specifically inspect why even a ~101 MiB RKNPU tensor subset cannot allocate on this ROCK 5 path

## Allocator Investigation

This pass moved from manifest tuning to the RKNPU allocator/runtime path using the smallest failing repro:

```bash
env HYBRID_MANIFEST=./examples/hybrid-manifests/dense-q8-tail-down.json \
    HYBRID_PROFILE=dense-balanced \
    HYBRID_STRICT=1 \
    ./build/bin/llama-cli \
    -m /home/radxa/hf_models/Qwen3-4B-Thinking-2507-Q8_0.gguf \
    --n-gpu-layers 99 -c 512 -b 32 -ub 32 --no-warmup -t 1 -tb 1 \
    --hybrid-dry-run \
    --hybrid-dump-plan /tmp/qwen-q8-dense-q8-tail-down.json \
    -p 'phase 3 q8 reduced hybrid smoke.' -n 1 \
    > /tmp/qwen-q8-dense-q8-tail-down.log 2>&1
```

### Exact Failing Allocation

Added narrow logging in:

- [`src/llama.cpp`](/home/radxa/edgeai-llama.cpp-phase3/src/llama.cpp)
- [`ggml/src/ggml-rknpu2/ggml-rknpu2.cpp`](/home/radxa/edgeai-llama.cpp-phase3/ggml/src/ggml-rknpu2/ggml-rknpu2.cpp)
- [`ggml/src/ggml-rknpu2/rknpu2-allocation.cpp`](/home/radxa/edgeai-llama.cpp-phase3/ggml/src/ggml-rknpu2/rknpu2-allocation.cpp)

The failing log now shows:

- `RKNPU buffer group -> tensors=4, alloc=149422080 bytes (142.50 MiB), largest=blk.24.ffn_down.weight (49807360 bytes / 47.50 MiB)`
- `RKNPU_BUFFER_ALLOC: one-shot backend buffer request size=149422080 bytes (142.50 MiB), alignment=64`
- `RKNPU_DMA_ALLOC: request size=149422080 bytes (142.50 MiB), page_size=4096, heap=/dev/dma_heap/cma`
- `RKNPU_DMA_ALLOC: ioctl DMA_HEAP_IOCTL_ALLOC failed for size=149422080 bytes on /dev/dma_heap/cma: Cannot allocate memory`

Conclusion:

- the 100.94 MiB routed tensor payload becomes a 142.50 MiB backend-buffer allocation after RKNPU packed-size calculation plus GGML padding
- this is one single contiguous DMA allocation, not four separate per-tensor allocations

### Why It Is One-Shot

The top-level model buffer is not segmented by tensor.

Evidence:

- [`ggml_backend_alloc_ctx_tensors_from_buft()`](/home/radxa/edgeai-llama.cpp-phase3/ggml/src/ggml-alloc.c) aggregates all tensors in the RKNPU context into one buffer unless the backend advertises a `max_size`
- the RKNPU backend does not provide `get_max_size`
- segmentation in [`ggml/src/ggml-rknpu2/ggml-rknpu2.cpp`](/home/radxa/edgeai-llama.cpp-phase3/ggml/src/ggml-rknpu2/ggml-rknpu2.cpp) only affects per-tensor packing and per-op `rknn_create_mem_from_fd()` views, not the top-level DMA heap allocation

### Standalone DMA Heap Probe

Built and ran a direct DMA-heap ioctl probe outside the repo to test realistic allocation sizes on this board.

Probe result:

```text
FAIL heap=/dev/dma_heap/cma size=105840640 bytes (100.94 MiB) errno=12 Cannot allocate memory
FAIL heap=/dev/dma_heap/cma size=104857600 bytes (100.00 MiB) errno=12 Cannot allocate memory
FAIL heap=/dev/dma_heap/cma size=100663296 bytes (96.00 MiB) errno=12 Cannot allocate memory
FAIL heap=/dev/dma_heap/cma size=67108864 bytes (64.00 MiB) errno=12 Cannot allocate memory
FAIL heap=/dev/dma_heap/cma size=33554432 bytes (32.00 MiB) errno=12 Cannot allocate memory
FAIL heap=/dev/dma_heap/cma size=16777216 bytes (16.00 MiB) errno=12 Cannot allocate memory
FAIL heap=/dev/dma_heap/cma size=8388608 bytes (8.00 MiB) errno=12 Cannot allocate memory
FAIL heap=/dev/dma_heap/cma size=4194304 bytes (4.00 MiB) errno=12 Cannot allocate memory
OK   heap=/dev/dma_heap/cma size=1048576 bytes (1.00 MiB)
OK   heap=/dev/dma_heap/system size=105840640 bytes (100.94 MiB)
```

Conclusion:

- CMA on this board is effectively unusable above ~1 MiB for this path
- the issue is not just “one request too big”; the CMA heap itself is far too small / constrained for the current runtime expectations
- `/dev/dma_heap/system` can satisfy the same allocation sizes

### Board Runtime Evidence

Live system state:

```bash
cat /proc/cmdline
grep -E 'MemTotal|MemFree|CmaTotal|CmaFree' /proc/meminfo
```

Observed values:

- no `cma=...` override is present in `/proc/cmdline`
- `CmaTotal: 8192 kB`
- `CmaFree: 4008 kB`

This matches the direct probe and explains the CMA allocation failures.

### Why `RKNPU: using device RKNPU - 0 MiB free`

That output is not real memory telemetry.

[`ggml_backend_rknpu_device_get_memory()`](/home/radxa/edgeai-llama.cpp-phase3/ggml/src/ggml-rknpu2/ggml-rknpu2.cpp) currently hardcodes:

- `*free = 0`
- `*total = 0`

So `0 MiB free` is just an unimplemented reporting stub.

### Minimal Patch Tried

Patched [`ggml/src/ggml-rknpu2/rknpu2-allocation.cpp`](/home/radxa/edgeai-llama.cpp-phase3/ggml/src/ggml-rknpu2/rknpu2-allocation.cpp) so allocator fallback is attempted not only when `/dev/dma_heap/cma` cannot be opened, but also when CMA `DMA_HEAP_IOCTL_ALLOC` fails.

Result with the same `dense-q8-tail-down` repro:

- CMA allocation still fails first
- allocator retries `/dev/dma_heap/system`
- model load gets past the old allocation failure
- process later segfaults during startup, after:
  - `llm_load_tensors:      RKNPU buffer size =   142.50 MiB`
  - `llama_init_from_model: KV self size  =   72.00 MiB`

Conclusion:

- the original load-time blocker was real and is bypassed by system-heap fallback
- but system-heap-backed buffers are not yet proven safe for the full RKNN import/runtime path on this setup

### Root-Cause Hypothesis

Primary cause:

- environment/runtime mismatch: the board CMA pool is only 8 MiB, while the RKNPU backend currently assumes large DMA-heap allocations can come from CMA

Secondary code issue:

- allocator fallback was incomplete; it did not retry `system` after CMA allocation failure

Current best interpretation:

- manifest breadth is not the blocker anymore
- segmentation of the top-level model buffer would not solve this board as configured, because CMA fails even at 4 MiB and only succeeds at 1 MiB in the direct probe
- the likely next fix direction is either:
  - runtime/environment: boot with a much larger CMA reservation, matching repo docs
  - backend/runtime compatibility: prove whether `system` heap is supported end-to-end by RKNN `create_mem_from_fd` on RK3588

### Recommendation

Do not spend more time tuning manifests.

Next step should be one of:

- increase CMA on the ROCK 5 and rerun the smallest repro first
- or isolate the post-load segfault with a tiny RKNN `create_mem_from_fd` import probe using a `system`-heap fd

At this point the strongest evidence says the dominant blocker is runtime/environment configuration, not tensor routing policy.

## Post-Allocation Crash Isolation

After enabling system-heap fallback, the old DMA allocation failure disappeared, but the process segfaulted later during startup.

### Exact Repro Command

```bash
env HYBRID_MANIFEST=./examples/hybrid-manifests/dense-q8-tail-down.json \
    HYBRID_PROFILE=dense-balanced \
    HYBRID_STRICT=1 \
    ./build/bin/llama-cli \
    -m /home/radxa/hf_models/Qwen3-4B-Thinking-2507-Q8_0.gguf \
    --n-gpu-layers 99 -c 512 -b 32 -ub 32 --no-warmup -t 1 -tb 1 \
    --hybrid-dry-run \
    --hybrid-dump-plan /tmp/qwen-q8-dense-q8-tail-down.json \
    -p 'phase 3 q8 reduced hybrid smoke.' -n 1 \
    > /tmp/qwen-q8-dense-q8-tail-down.log 2>&1
```

### Backtrace

Ran the same repro under `gdb`:

```bash
gdb -q --batch \
  -ex 'set pagination off' \
  -ex 'set env HYBRID_MANIFEST=./examples/hybrid-manifests/dense-q8-tail-down.json' \
  -ex 'set env HYBRID_PROFILE=dense-balanced' \
  -ex 'set env HYBRID_STRICT=1' \
  -ex 'run -m /home/radxa/hf_models/Qwen3-4B-Thinking-2507-Q8_0.gguf --n-gpu-layers 99 -c 512 -b 32 -ub 32 --no-warmup -t 1 -tb 1 --hybrid-dry-run --hybrid-dump-plan /tmp/qwen-q8-dense-q8-tail-down-gdb.json -p \"phase 3 q8 reduced hybrid smoke.\" -n 1' \
  -ex 'bt' \
  -ex 'info registers' \
  --args ./build/bin/llama-cli
```

Backtrace:

```text
Thread 1 "llama-cli" received signal SIGSEGV, Segmentation fault.
0x0000fffff7583eb4 in ggml_guid_matches () from build/ggml/src/libggml.so
#0  ggml_guid_matches
#1  llama_init_from_model
#2  llama_init_from_gpt_params
#3  main
```

### Root Cause

The segfault was not in RKNN import, graph build, first op preparation, or first execution.

It happened earlier during backend classification in `llama_init_from_model()` while iterating `ctx->backends`.

Root cause:

- the RKNPU backend object in [`ggml/src/ggml-rknpu2/ggml-rknpu2.cpp`](/home/radxa/edgeai-llama.cpp-phase3/ggml/src/ggml-rknpu2/ggml-rknpu2.cpp) was still being created with:
  - `.guid = {0}`
- later, scheduler setup calls `ggml_backend_is_cpu(backend)`
- `ggml_backend_is_cpu()` calls `ggml_guid_matches(backend->guid, ggml_backend_cpu_guid())`
- the zero / invalid RKNPU backend GUID caused the crash

This means:

- the crash was unrelated to heap choice
- the system-heap fallback did not introduce the segfault
- it only exposed an existing backend-identity bug that had been masked by the earlier allocation failure

### Minimal Fix

Patched [`ggml/src/ggml-rknpu2/ggml-rknpu2.cpp`](/home/radxa/edgeai-llama.cpp-phase3/ggml/src/ggml-rknpu2/ggml-rknpu2.cpp) to add a real static backend GUID and use it when constructing the RKNPU backend.

### Result After Fix

The same `dense-q8-tail-down` repro now succeeds end-to-end.

Evidence from `/tmp/qwen-q8-dense-q8-tail-down.log`:

- model buffer allocation:
  - `RKNPU_DMA_ALLOC ... heap=/dev/dma_heap/cma`
  - `ioctl ... failed`
  - retry on `heap=/dev/dma_heap/system`
- scheduler / compute setup succeeds:
  - `RKNPU compute buffer size = 1.50 MiB`
  - `graph nodes = 905`
  - `graph splits = 9`
- run completes successfully:
  - `prompt eval time = 2732.51 ms / 9 tokens`
  - `sample time = 0.16 ms / 1 runs`
  - `total time = 3769.50 ms / 10 tokens`

This proves:

- the system-heap fallback is valid for the downstream RKNN path in this minimal repro
- the crash happened before RKNN op execution and was caused by backend GUID bookkeeping

### Updated Recommendation

Keep both fixes:

- CMA allocation retry to system heap
- valid backend GUID for RKNPU

The correct next-fix category was:

- later runtime bug unrelated to heap choice

More specifically:

- backend metadata / identity bug in the RKNPU backend object

At this point, for the small `dense-q8-tail-down` profile:

- load succeeds
- prompt evaluation succeeds
- 1-token inference succeeds

## Phase 3 Ladder After CMA Fallback + Backend GUID Fix

Ran a three-profile ladder with the two fixes kept together:

- [`ggml/src/ggml-rknpu2/rknpu2-allocation.cpp`](/home/radxa/edgeai-llama.cpp-phase3/ggml/src/ggml-rknpu2/rknpu2-allocation.cpp)
- [`ggml/src/ggml-rknpu2/ggml-rknpu2.cpp`](/home/radxa/edgeai-llama.cpp-phase3/ggml/src/ggml-rknpu2/ggml-rknpu2.cpp)

Command pattern:

```bash
env HYBRID_MANIFEST=./examples/hybrid-manifests/<manifest>.json \
    HYBRID_PROFILE=dense-balanced \
    HYBRID_STRICT=1 \
    ./build/bin/llama-cli \
    -m /home/radxa/hf_models/Qwen3-4B-Thinking-2507-Q8_0.gguf \
    --n-gpu-layers 99 -c 512 -b 32 -ub 32 --no-warmup -t 1 -tb 1 \
    --hybrid-dry-run \
    --hybrid-dump-plan /tmp/<plan>.json \
    -p '<ladder prompt>' -n 1 \
    > /tmp/<log>.log 2>&1
```

### 1. Minimal Known-Good

Manifest:

- [`examples/hybrid-manifests/dense-q8-tail-down.json`](/home/radxa/edgeai-llama.cpp-phase3/examples/hybrid-manifests/dense-q8-tail-down.json)

Placement summary:

- CPU tensors = 395, bytes = 4369.62 MiB
- RKNPU tensors = 4, bytes = 100.94 MiB
- RKNPU buffer group = 142.50 MiB

Fallback diagnostics:

- CMA model-buffer allocation failed
- allocator retried `/dev/dma_heap/system`
- RKNPU compute buffer stayed on CMA at 1.50 MiB

Result:

- load success: yes
- 1-token success: yes

Timings:

- sample time = 0.16 ms / 1 run
- prompt eval time = 2296.36 ms / 8 tokens
- total time = 3253.15 ms / 9 tokens

### 2. Moderate

Manifest:

- [`examples/hybrid-manifests/dense-q8-late-ffn.json`](/home/radxa/edgeai-llama.cpp-phase3/examples/hybrid-manifests/dense-q8-late-ffn.json)

Placement summary:

- CPU tensors = 375, bytes = 3864.93 MiB
- RKNPU tensors = 24, bytes = 605.62 MiB
- RKNPU buffer group = 855.00 MiB

Fallback diagnostics:

- CMA model-buffer allocation failed
- allocator retried `/dev/dma_heap/system`
- RKNPU compute buffer still allocated from CMA at 1.50 MiB

Result:

- load success: yes
- 1-token success: no

Observed failure:

- abort in [`ggml_backend_rknpu_graph_compute()`](/home/radxa/edgeai-llama.cpp-phase3/ggml/src/ggml-rknpu2/ggml-rknpu2.cpp)
- assertion:
  - `GGML_ASSERT(it != src0_buf_ctx->quantized_tensor_scales.end() && "Quantized scale not found") failed`

Interpretation:

- this is no longer a heap-allocation problem
- it is a broader runtime / backend state-management issue triggered when more than the minimal tensor subset is exercised

### 3. Original Balanced

Manifest:

- [`examples/hybrid-manifests/dense-balanced.json`](/home/radxa/edgeai-llama.cpp-phase3/examples/hybrid-manifests/dense-balanced.json)

Placement summary:

- CPU tensors = 291, bytes = 1745.24 MiB
- RKNPU tensors = 108, bytes = 2725.31 MiB
- RKNPU buffer group = 3847.50 MiB

Fallback diagnostics:

- CMA model-buffer allocation failed
- allocator retried `/dev/dma_heap/system`
- RKNPU compute buffer allocated from CMA at 1.50 MiB
- RKNN runtime emitted:
  - `failed to convert fd(10) to handle`
  - `failed to submit!, op id: 0, op name: MatMul`
  - `RKNPU2: rknn_matmul_run failed for segment ...`

Result:

- load success: yes
- 1-token success: yes

Timings:

- sample time = 0.36 ms / 1 run
- prompt eval time = 6798.38 ms / 8 tokens
- total time = 19450.81 ms / 9 tokens

Interpretation:

- the profile can now run end-to-end
- but broader runtime issues remain under heavier RKNPU use
- completion alone is not enough to call the balanced profile clean

### Decision

The current balanced profile is no longer blocked by manifest breadth alone.

Evidence:

- minimal profile succeeds
- moderate profile fails with backend state / scale bookkeeping, not allocation
- balanced profile completes, but with RKNN import / submit failures in the log

Conclusion:

- broader runtime work is still required
- manifest tuning may still help performance or reduce stress, but it is not the primary blocker anymore
- the next work item should focus on:
  - quantized-scale lifecycle / bookkeeping for larger routed subsets
  - RKNN fd-to-handle import reliability for large system-heap-backed model buffers
  - whether failed RKNPU segments are silently falling back or partially degrading correctness / performance
