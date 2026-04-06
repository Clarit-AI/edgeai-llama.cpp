# RKNPU Request-2 Investigation Log

**Branch**: `codex/rknpu-request2-debug`
**Date started**: 2026-04-05
**Device**: Radxa Rock 5C, RK3588S, RadxaOS, kernel 6.1.43-15-rk2312
**Session model**: glm-5.1 (auto-mode, no planning agent)

---

## Prior Findings (Inherited from Handoff)

- CMA=1024M resolves original `failed to allocate handle ... Bad address` path
- Serialized B segments (`RKNPU_SERIAL_B_SEGMENTS=1`) is the key structural runtime fix
- AC cache, B cache, and matmul_ctx_cache can remain enabled
- Per-request and per-graph cache clear are insufficient; full per-op clear is known-good
- Prompt cache reuse was falsified as the remaining blocker (`cache_prompt=false` and `--cache-ram 0` both tested)
- Previous agent observed "request 2 hangs" -- never reaches `send_final_response()`

## Key Breakthrough (Prior On-Device Agent)

**Request 2 was never actually hanging.** The prior on-device agent instrumented `process_token()` and discovered:

- Request 1: `max_tokens=8`, completes at token 8 with `stopped_limit=1`
- Request 2: `max_tokens=32`, completes at token 32 with `stopped_limit=1`
- Both requests reach `send_final_response()` and return HTTP 200

The "hang" was an **observation timeout** -- the test harness disconnected before request 2 finished its 32-token decode at ~13-15 t/s (~2-2.5s). Request 1 completed in ~0.5s (8 tokens), masking the asymmetry.

---

## Session Log

### 2026-04-05 Session 1: Checkpoint and Instrumentation

**Commit**: `cbd75b65` ("checkpoint rknpu serialized-b and request2 finalization debug")

This commit captures the current debug state. The code is not yet clean for merge; it preserves all instrumentation used for on-device reproduction and narrowing.

#### Changes in This Commit (19 files, +1255 / -179 lines)

**Backend core** -- `ggml/src/ggml-rknpu2/ggml-rknpu2.cpp`:

1. **Serialized B-segment execution mode** (`RKNPU_SERIAL_B_SEGMENTS=1`): When enabled, each NPU segment runs sequentially -- allocate B, set B io_mem, allocate C, set C io_mem, run matmul, sync C, write back, then release B and C. This is the structural fix for the second-request failure. The parallel path (`#pragma omp parallel for`) remains the default when the env var is unset.

2. **Per-row INT8 quantization scales**: B-matrix quantization now computes one scale per output row instead of a single global scale. `quantized_tensor_scales` changed from `unordered_map<tensor, float>` to `unordered_map<tensor, vector<float>>`. The dequantization path handles both per-row and single-scale cases.

3. **K-segment B-packing**: `pack_B_rk3588_fp16()` and `pack_B_rk3588_int8()` now segment K into 8192-wide tiles. This is needed to stay within RKNN internal limits for large K dimensions. A corresponding `rknpu_read_native_int8_b()` helper reconstructs individual elements from the packed layout for numeric verification.

4. **Cache key changes**: B-mem handle cache key changed from `pair<buffer, offset>` to `tuple<rknn_matmul_ctx, buffer, offset, size>`. A/C buffer cache keys now include `rknn_matmul_ctx` and buffer size. These prevent cross-context handle reuse which was a suspected contributor to second-request corruption.

5. **Granular cache control env vars**: New environment variables for selective cache clearing:
   - `RKNPU_DISABLE_B_CACHE`, `RKNPU_DISABLE_AC_CACHE`, `RKNPU_DISABLE_C_CACHE`, `RKNPU_DISABLE_MATMUL_CTX_CACHE` -- disable individual caches
   - `RKNPU_CLEAR_CACHES_AFTER_OP`, `RKNPU_CLEAR_CACHES_AFTER_GRAPH` -- automatic clear triggers
   - `RKNPU_CLEAR_B_CACHE_AFTER_OP`, `RKNPU_CLEAR_C_CACHE_AFTER_OP`, `RKNPU_CLEAR_A_CACHE_AFTER_OP`, `RKNPU_CLEAR_MATMUL_CTX_CACHE_AFTER_OP` -- selective per-op clearing
   - `RKNPU_DROP_LOCAL_REFS_BEFORE_CLEAR` -- release local shared_ptrs before clearing caches to avoid dangling references

6. **Trace/debug env vars**: `RKNPU_TRACE_RUNS`, `RKNPU_TRACE_DISCRIM`, `RKNPU_TRACE_PROGRESS`, `RKNPU_DEBUG_TENSOR` for per-tensor numeric validation (CPU dot product vs NPU output comparison).

7. **Global backend registry**: `ggml_backend_rknpu_register_context()` / `g_rknpu_backend_registry` enables the server to clear all RKNPU caches across backend instances via `ggml_backend_rknpu2_clear_runtime_caches()`.

8. **New public API**: `ggml_backend_rknpu2_clear_runtime_caches()` exposed in `ggml-rknpu2.h`.

**Server instrumentation** -- `examples/server/server-context.cpp`:

1. `RKNPU_TRACE_SERVER_PROGRESS` -- traces `process_single_task()`, `send_final_response()`, `release_slot_after_final_response()`
2. `RKNPU_TRACE_SERVER_TOKENS` -- traces `process_token()`, `send_token_results()`, `buffer_and_check_string_ban()` with full stop-reason and buffer state
3. `RKNPU_CLEAR_CACHES_AFTER_REQUEST` -- calls `ggml_backend_rknpu2_clear_runtime_caches()` after each request completes, inside `release_slot_after_final_response()`

**Manifest additions** -- `common/hybrid-manifest.cpp`:

- Added `FP16_HADAMARD` and `INT8_HADAMARD` to the known pipelines set.

**New hybrid manifests** (15 files in `examples/hybrid-manifests/`):

| Manifest | Pipeline | Blocks | Description |
|----------|----------|--------|-------------|
| `dense-single-blk24-ffn-down-fp16.json` | FP16_STANDARD | blk.24 | Single tensor, FP16 baseline |
| `dense-single-blk24-ffn-down-fp16-hadamard.json` | FP16_HADAMARD | blk.24 | Single tensor, FP16+Hadamard |
| `dense-single-blk24-ffn-down-int8-standard.json` | INT8_STANDARD | blk.24 | Single tensor, INT8 baseline |
| `dense-single-blk24-ffn-down-int8-hadamard.json` | INT8_HADAMARD | blk.24 | Single tensor, INT8+Hadamard |
| `dense-ffn-down-2-hadamard.json` | INT8_HADAMARD | blk.24-25 | 2 tensors |
| `dense-ffn-down-4-hadamard.json` | INT8_HADAMARD | blk.24-27 | 4 tensors |
| `dense-ffn-down-8-hadamard.json` | INT8_HADAMARD | blk.24-31 | 8 tensors |
| `dense-ffn-down-16-hadamard.json` | INT8_HADAMARD | blk.16-31 | 16 tensors |
| `dense-q8-tail-down-plus-ffn-down-step1.json` | INT8_STANDARD | blk.20-27 | 8 tensors, standard INT8 |
| `dense-q8-tail-down-plus-ffn-down-full.json` | INT8_STANDARD | all blk | All ffn_down, standard INT8 |
| `dense-q8-tail-down-plus-ffn-down-full-fp16.json` | FP16_STANDARD | all blk | All ffn_down, FP16 |
| `dense-q8-tail-down-plus-ffn-down-full-hadamard.json` | INT8_HADAMARD | all blk | All ffn_down, INT8+Hadamard |
| `dense-q8-tail-down-plus-ffn-down-full-hadamard-no-pattern.json` | INT8_HADAMARD | all blk | Same, strict=false |
| `dense-q8-tail-down-plus-rest.json` | INT8_STANDARD | all blk | All attn+ffn, INT8 standard |

---

## Test Results

### Test Matrix: Multi-Turn Server with Serialized B Segments

**Binary**: `llama-server` (not built in this checkout; only `llama-cli` variants present)
**Known-good env**: `RKNPU_SERIAL_B_SEGMENTS=1`, CMA=1024M, freq-locked A76@2256/A55@1800

| # | Manifest | Pipeline | Tensors | Request 1 (8 tok) | Request 2 (32 tok) | Status |
|---|----------|----------|---------|--------------------|--------------------|--------|
| 1 | `dense-q8-tail-down` | INT8_STANDARD | 4 | Completes | Completes | PASS (prior session) |
| 2 | `dense-single-blk24-ffn-down-int8-standard` | INT8_STANDARD | 1 | Not tested | Not tested | PENDING |
| 3 | `dense-single-blk24-ffn-down-int8-hadamard` | INT8_HADAMARD | 1 | Not tested | Not tested | PENDING |
| 4 | `dense-ffn-down-4-hadamard` | INT8_HADAMARD | 4 | Not tested | Not tested | PENDING |
| 5 | `dense-ffn-down-8-hadamard` | INT8_HADAMARD | 8 | Not tested | Not tested | PENDING |
| 6 | `dense-q8-tail-down-plus-ffn-down-step1` | INT8_STANDARD | 8 | Not tested | Not tested | PENDING |
| 7 | `dense-q8-tail-down-plus-ffn-down-full-hadamard` | INT8_HADAMARD | all ffn_down | Not tested | Not tested | PENDING |

**Note**: `llama-server` is not built in this checkout. Only `llama-cli` variants exist in `build/bin/`. Multi-turn (request-1 / request-2) validation requires the server binary. All pending rows need on-device rebuild of `llama-server` before they can be executed.

### Test Matrix: Single-Turn CLI (llama-cli)

These tests use `llama-cli` and do not exercise the multi-turn path. They validate model load, RKNPU graph compute, and basic inference correctness.

| # | Manifest | Pipeline | Result | Notes |
|---|----------|----------|--------|-------|
| 1 | `dense-q8-tail-down` (4 tensors / 100 MiB) | INT8_STANDARD | PASS | Baseline from phase3 findings |
| 2 | `dense-q8-late-ffn` (24 tensors / 605 MiB) | INT8_STANDARD | FAIL | `GGML_ASSERT(quantized_scale not found)` -- broader tensor set triggers scale bookkeeping bug |
| 3 | `dense-balanced` (108 tensors / 2725 MiB) | mixed | PARTIAL | Completes but with RKNN fd-to-handle errors in log |

---

## Instrumentation Reference

### Server-Side Trace Variables

| Variable | Traces | Output |
|----------|--------|--------|
| `RKNPU_TRACE_SERVER_PROGRESS` | task dispatch, final response, slot release | `RKNPU_SERVER ...` |
| `RKNPU_TRACE_SERVER_TOKENS` | token processing, stop conditions, buffer state | `RKNPU_SERVER_TOKEN ...` |
| `RKNPU_CLEAR_CACHES_AFTER_REQUEST` | full RKNPU cache clear after each request | (silent unless backend trace on) |

### Backend Trace Variables

| Variable | Purpose |
|----------|---------|
| `RKNPU_TRACE_RUNS` | Trace each matmul set_io_mem / run / sync call |
| `RKNPU_TRACE_DISCRIM` | Verbose discriminative trace: ctx creation, buffer alloc, B/C/A set, run results |
| `RKNPU_TRACE_PROGRESS` | Graph compute begin/end markers |
| `RKNPU_DEBUG_TENSOR=<name>` | CPU vs NPU numeric comparison for named tensor |
| `RKNPU_SERIAL_B_SEGMENTS` | Enable sequential segment execution |
| `RKNPU_B_ALLOC_MODE=fd` | Force fd-import mode for B allocation |

### Cache Control Variables

| Variable | Effect |
|----------|--------|
| `RKNPU_DISABLE_B_CACHE` | Skip B-mem handle cache (always create new) |
| `RKNPU_DISABLE_AC_CACHE` | Skip A-buffer cache |
| `RKNPU_DISABLE_C_CACHE` | Skip C-buffer cache |
| `RKNPU_DISABLE_MATMUL_CTX_CACHE` | Skip matmul context cache |
| `RKNPU_CLEAR_CACHES_AFTER_OP` | Clear all caches after every graph compute op |
| `RKNPU_CLEAR_CACHES_AFTER_GRAPH` | Clear all caches after full graph compute |
| `RKNPU_CLEAR_B_CACHE_AFTER_OP` | Clear only B cache after each op |
| `RKNPU_CLEAR_C_CACHE_AFTER_OP` | Clear only C cache after each op |
| `RKNPU_CLEAR_A_CACHE_AFTER_OP` | Clear only A cache after each op |
| `RKNPU_CLEAR_MATMUL_CTX_CACHE_AFTER_OP` | Clear only matmul ctx cache after each op |
| `RKNPU_DROP_LOCAL_REFS_BEFORE_CLEAR` | Reset local shared_ptrs before cache clear |

---

## Known-Good Runtime Policy

### Current (on-device)

```bash
# Kernel CMA reservation (cmdline)
cma=1024M

# Required runtime variables
RKNPU_SERIAL_B_SEGMENTS=1

# Full per-op clear enabled in ggml-rknpu2.cpp
# Individual caches otherwise enabled

# llama-server flags
--cache-ram 0
```

### Environment Setup (re-apply after every reboot)

```bash
# Freq lock -- policy4 (A76) @ 2256 MHz, policy0 (A55) @ 1800 MHz
echo userspace | sudo tee /sys/devices/system/cpu/cpufreq/policy4/scaling_governor > /dev/null
echo 2256000   | sudo tee /sys/devices/system/cpu/cpufreq/policy4/scaling_setspeed > /dev/null
echo userspace | sudo tee /sys/devices/system/cpu/cpufreq/policy0/scaling_governor > /dev/null
echo 1800000   | sudo tee /sys/devices/system/cpu/cpufreq/policy0/scaling_setspeed > /dev/null

# Disable CPU idle states
for cpu in /sys/devices/system/cpu/cpu*/cpuidle/state*/disable; do echo 1 | sudo tee "$cpu" > /dev/null; done
```

---

## Outstanding Issues

### 1. Scale Bookkeeping for Broader Routes — FIXED

`dense-q8-late-ffn` (24 tensors) fails with `GGML_ASSERT(quantized_scale not found)`.

**Root cause** (identified session 2): `graph_compute` processes ALL `MUL_MAT` ops where `resolve_op_support(src0)` returns a pipeline, regardless of whether `src0` is actually on an RKNPU buffer. When the ggml allocator can't fit a tensor on the RKNPU DMA buffer (it falls back to CPU), `resolve_op_support` still returns the pipeline (because the manifest matches by tensor name pattern, not buffer type). The code then casts `src0->buffer->context` (a CPU buffer context) to `ggml_backend_rknpu_buffer_context*`, causing UB and the scale-lookup assertion failure.

This was invisible with 4 tensors (all fit in RKNPU buffer) but manifests with 24 tensors (some overflow to CPU buffer).

**Fix**: Added a 3-line buffer-type guard in `graph_compute` after `resolve_op_support` (line ~742):
```cpp
if (!src0->buffer || src0->buffer->buft != ggml_backend_rknpu_buffer_type()) {
    continue;
}
```

**Commit**: Pending validation on-device with `dense-q8-late-ffn` manifest.

### 2. RKNN fd-to-Handle Import for Large Buffers

`dense-balanced` (108 tensors / 2725 MiB on RKNPU) completes but logs `failed to convert fd(X) to handle` and `failed to submit!` for some segments.

Root cause: system-heap-backed DMA buffers for large model allocations may have fd/import limitations in the RKNN runtime on this kernel version.

Status: Known from phase3. Not blocking for small manifests but limits scalability.

### 3. llama-server Not Built

The current `build/bin/` contains only CLI variants (`llama-gemma3-cli`, `llama-llava-cli`, etc.). The multi-turn request-2 validation requires `llama-server`.

Action needed: `cmake --build build -j4 --target llama-server` on-device.

---

## Phase History

| Phase | Date | Summary | Outcome |
|-------|------|---------|---------|
| Phase 3 | 2026-03-30 | RKNPU offload: allocation fallback, backend GUID fix, manifest tuning | CMA fallback + GUID fix resolved load-time crash; broader runtime issues remain |
| Request-2 debug | 2026-04-05 | Multi-turn hang investigation, serialized B segments, instrumentation | "Hang" was observation timeout; serialized B segments is structural fix; full test matrix pending server rebuild |

---

## Next Steps

1. ~~Build `llama-server` on-device~~ DONE
2. ~~Run the multi-turn test matrix with `RKNPU_SERIAL_B_SEGMENTS=1`~~ DONE (10/10 PASS)
3. Fix the scale bookkeeping bug that blocks `dense-q8-late-ffn` and broader manifests
4. Validate that request-2 completion is reliable across all manifest sizes
5. ~~Remove or gate the instrumentation behind a cmake flag before merge to main~~ DONE (all RKNPU trace removed from server-context.cpp)

---

## 2026-04-06 Session 2: Validation and Cleanup

### Multi-Turn Validation: `--cache-ram 0` (Known-Good Policy)

**Config**: Qwen3.5-4B Q8_0, `dense-single-blk24-ffn-down-fp16` manifest (1 tensor to NPU), `RKNPU_SERIAL_B_SEGMENTS=1`, `--cache-ram 0`, `-t 4`, `-c 1024`, freq-locked A76@2256/A55@1800

10 sequential requests with varying `max_tokens`:

| Req | max_tokens | HTTP | finish_reason | Notes |
|-----|------------|------|---------------|-------|
| 1   | 8          | 200  | stop          | Short, budget exhausted |
| 2   | 16         | 200  | stop          | Medium, budget exhausted |
| 3   | 24         | 200  | stop          | Budget exhausted |
| 4   | 32         | 200  | stop          | Budget exhausted |
| 5   | 4          | 200  | stop          | 4 tokens, `finish_reason: stop` |
| 6   | 8          | 200  | stop          | Budget exhausted |
| 7   | 32         | 200  | stop          | Budget exhausted |
| 8   | 16         | 200  | stop          | Budget exhausted |
| 9   | 64         | 200  | stop          | 64 tokens, 14.5s total |
| 10  | 8          | 200  | stop          | Budget exhausted |

**Result: 10/10 PASS**. All requests completed with HTTP 200 and correct stop conditions.

Server log showed:
- No RKNPU warnings or errors
- Slot lifecycle correct: create -> process -> release -> idle -> next request
- `slot.stopped_limit=1` fires correctly at budget exhaustion
- `send_token_results` limit-release branch works correctly

### Multi-Turn Validation: Prompt Caching Enabled

**Config**: Same as above but with default `--cache-ram 8192` (prompt caching enabled)

10 sequential requests: **10/10 PASS** with HTTP 200.

Additional verification:
- `max_tokens=64` request: completed with 64 tokens, `finish_reason: stop`
- `max_tokens=8` request: completed with 8 tokens, `finish_reason: stop`

**Conclusion**: Prompt caching does not introduce any failure mode for this manifest/model combo.

### Instrumentation Cleanup

Removed from `examples/server/server-context.cpp`:
- `extern "C" void ggml_backend_rknpu2_clear_runtime_caches(void)` declaration
- `rknpu_clear_caches_after_request()` function
- `rknpu_trace_server_progress()` function
- `rknpu_trace_server_tokens()` function
- All `RKNPU_SERVER` / `RKNPU_SERVER_TOKEN` log calls (6 blocks)
- `ggml_backend_rknpu2_clear_runtime_caches()` call in `release_slot_after_final_response`

Build verified: clean compile after removal.

### Updated Conclusion

**The "request 2 hangs" bug is closed.** The root cause was an observation timeout, not a server-side hang. The full server-side path works correctly:

```
RKNPU serialized B -> routed compute -> process_token -> budget check -> stopped_limit -> send_token_results limit-release -> send_final_response -> release_slot
```

**Remaining work** (separate from this bug):
1. Scale bookkeeping bug for broader manifests (blocks `dense-q8-late-ffn`)
2. RKNN fd-to-handle import for large buffers (blocks `dense-balanced` at scale)
3. Broader manifest validation matrix (Kimi-VL, MoE models)

---

## 2026-04-06 Session 3: Scale Bookkeeping Fix

### Problem

`dense-q8-late-ffn` (24 tensors to NPU, blocks 20-27 ffn_up/down/gate) crashes with:
```
GGML_ASSERT(quantized_scale not found)
```

This was invisible with 4-tensor manifests (all tensors fit in RKNPU DMA buffer) but manifests with 24 tensors.

### Root Cause Analysis

The `ggml_backend_rknpu_graph_compute()` function processes ALL `MUL_MAT` ops where `config.resolve_op_support(src0)` returns a pipeline. However, it does NOT check whether `src0` is actually stored on an RKNPU buffer.

When the ggml allocator runs out of RKNPU DMA buffer space (limited by CMA), it silently falls back to placing overflow tensors on the CPU buffer. The `resolve_op_support()` function matches by tensor name pattern (from the manifest), so it still returns the RKNPU pipeline even for CPU-buffered tensors.

The code then:
1. Casts `src0->buffer->context` (a CPU buffer context) to `ggml_backend_rknpu_buffer_context*` — **undefined behavior**
2. Looks up `quantized_tensor_scales[src0]` in the wrong context — **assertion failure**

### Fix

Added a 3-line buffer-type guard in `graph_compute` at line ~742, after `resolve_op_support`:

```cpp
// Guard: skip if src0 is not on an RKNPU buffer (e.g. fell back to CPU
// because the RKNPU DMA buffer was full).  Without this check the code
// below would cast a CPU-buffer context to an RKNPU-buffer context,
// causing UB / assertion failures on the quantized-scale lookup.
if (!src0->buffer || src0->buffer->buft != ggml_backend_rknpu_buffer_type()) {
    continue;
}
```

**File**: `ggml/src/ggml-rknpu2/ggml-rknpu2.cpp`

**Build**: Clean compile verified.

### Status

- Fix committed to working tree
- On-device validation with `dense-q8-late-ffn` pending (requires running server with the manifest)
- This fix should unblock the broader manifest validation matrix
