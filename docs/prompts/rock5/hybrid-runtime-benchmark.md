# ROCK 5 Hybrid Runtime Benchmark Session

You are running on or against a ROCK 5 / RK3588 environment for the repository:

`<repository-root>/edgeai-llama.cpp`

## Goal
Benchmark and compare CPU-only, legacy NPU, and manifest-driven hybrid execution on real hardware.

## Required Work
- Identify the available benchmark binaries and model files.
- Choose at least one dense model and, if available, one MoE model.
- Run the comparison matrix for each target model:
  1. CPU-only baseline
  2. legacy RKNPU2 baseline
  3. manifest-driven hybrid profile
- Collect:
  - prompt processing throughput
  - token generation throughput
  - manifest/profile attribution
  - fallback observations
  - repeated-run consistency
- Summarize where hybrid mode helps, hurts, or falls back.

## Deliverables
- compact benchmark table
- fallback summary
- recommendation for manifest tuning or code changes

## Constraints
- Keep runs attributable with `HYBRID_MANIFEST`, `HYBRID_PROFILE`, and `HYBRID_STRICT` where relevant.
- Do not claim conclusions from a single noisy run if repeated runs disagree.
