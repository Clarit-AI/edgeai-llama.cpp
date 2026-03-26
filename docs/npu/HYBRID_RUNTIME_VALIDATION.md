# Hybrid Runtime Validation on ROCK 5

This guide is the executable Phase 3 path for RK3588 / ROCK 5 validation. It is designed to collect:

- CPU-only baseline results
- legacy RKNPU2 baseline results
- manifest-driven hybrid results
- strict-mode and dry-run artifacts
- fallback notes from real device logs

## Assets

- Runtime manifests: `examples/hybrid-manifests/`
- Benchmark harness: `scripts/rock5/run-hybrid-phase3.sh`
- Generated output: `reports/rock5/<timestamp>/`

## Prerequisites

1. Build on the target device with RKNPU2 enabled.
2. Confirm `llama-cli` exists in `build/bin/`.
3. Confirm the target board has the RKNN runtime available.
4. Provide at least one dense model. Provide a MoE model if available.

## Recommended Invocation

```bash
./scripts/rock5/run-hybrid-phase3.sh \
  --build-dir ./build \
  --dense-model /path/to/dense-model.gguf \
  --moe-model /path/to/moe-model.gguf
```

## What The Harness Produces

For each run profile it writes:

- raw `llama-cli` log
- manifest dry-run output when applicable
- resolved plan JSON when applicable
- a generated markdown report with prompt and generation throughput parsed from the logs

## Included Runtime Manifests

- `dense-balanced.json`
- `dense-npu-heavy.json`
- `moe-balanced.json`

These are starting points for Phase 3 tuning, not final tuned results. They should be revised only after repeated on-device measurements show a clear improvement.
