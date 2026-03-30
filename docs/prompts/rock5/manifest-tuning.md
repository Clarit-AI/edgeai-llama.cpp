# ROCK 5 Manifest Tuning Session

You are running on or against a ROCK 5 / RK3588 environment for the repository:

`<repository-root>/edgeai-llama.cpp`

## Goal
Tune manifest rules based on measured RK3588 behavior.

## Required Work
- Inspect the current example manifests.
- Identify which layers/tensors are actually succeeding or falling back.
- Adjust manifest rules to improve either:
  - throughput
  - determinism
  - fallback clarity
- Re-run targeted benchmarks after each meaningful manifest change.
- Prefer the smallest manifest edits that produce measurable gains.

## Deliverables
- revised manifest files
- before/after benchmark comparison
- explanation of which rules improved results
- list of unresolved bottlenecks

## Constraints
- Do not broaden runtime support beyond current matmul scope.
- Do not change multiple unrelated variables at once.
