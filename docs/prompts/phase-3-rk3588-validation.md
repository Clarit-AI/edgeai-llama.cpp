# Phase 3: RK3588 Runtime Validation And Manifest Tuning

You are working in the repository at:

`<repository-root>/edgeai-llama.cpp`

## Context
- PR #6 (phase-1 hybrid manifest routing) is merged.
- Phase 2 parser/policy unification is assumed complete or should be verified before proceeding.
- This phase is primarily about real RK3588 / ROCK 5 validation, manifest tuning, and benchmark-backed policy decisions.
- The target hardware matters here; do not treat desktop/macOS results as authoritative for NPU behavior.

## Goal
Execute Phase 3: RK3588 runtime validation and manifest tuning for dense and MoE models.

## Required Outcomes
1. Validate that manifest-driven routing behaves correctly on actual ROCK 5 / RK3588 hardware.
2. Produce benchmark-backed comparison data for:
   - CPU-only baseline
   - legacy RKNPU2 baseline
   - manifest-driven hybrid mode
3. Tune manifests based on observed fallback behavior and throughput.
4. Confirm CPU fast paths still activate where expected.
5. Produce a go/no-go recommendation for deeper hybrid execution investment.

## Scope
- Dense MLA models first.
- Then MoE models.
- Focus on real token throughput, fallback reasons, deterministic routing, and correctness/stability.

## Required Work
- Verify current build/run instructions for ROCK 5.
- Run manifest-driven validation with:
  - `--hybrid-dry-run`
  - `--hybrid-dump-plan`
  - strict mode
- Collect benchmark results for:
  - prompt processing
  - token generation
  - repeated-run stability
  - per-manifest behavior
- Capture fallback behavior:
  - unsupported source types
  - alignment mismatches
  - CPU-owned paths
- Check that FlashMLA / Flash Attention / KV quantization / fused MoE CPU paths are not accidentally disabled by hybrid routing choices.
- Tune at least:
  - one dense balanced profile
  - one dense NPU-heavy profile
  - one MoE balanced profile

## Expected Outputs
- benchmark table for dense
- benchmark table for MoE
- notes on fallback hotspots
- notes on where hybrid routing helps and where it hurts
- revised example manifests if tuning improves outcomes
- clear recommendation:
  - continue to phase 4
  - or narrow scope first

## Constraints
- Do not add a new quant format in this phase.
- Do not rewrite scheduling.
- Do not attempt partial split of one matmul across CPU and NPU.
- Keep experiments attributable through manifest/profile metadata.

## Deliverables
- code or manifest updates only where justified by measured results
- benchmark summary suitable for a PR or design note → `benchmarks/rk3588-results.json`
- explicit recommendation for the next phase

## Execution Rule
Do not just discuss the benchmarking plan. Perform the runtime validation and tune manifests.