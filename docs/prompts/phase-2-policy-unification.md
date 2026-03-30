# Phase 2: Policy Unification And Deterministic Validation

You are working in the repository at:

`<repository-root>/edgeai-llama.cpp`

## Context
- This project is a fork combining ik-llama.cpp and rk-llama.cpp work.
- PR #6, which added phase-1 hybrid manifest routing, has already been merged into `main`.
- Phase 1 established:
  - hybrid manifest CLI/API surface
  - loader-owned deterministic per-tensor routing plan
  - tensor placement using loader routes
  - RKNPU2 explicit pipeline route consumption
  - docs/benchmark attribution fields
- Current known technical debt:
  - there are effectively two manifest-policy implementations in-tree:
    - common parser/types in `common/hybrid-manifest.*`
    - active loader-local manifest parsing/resolution in `src/llama-model-loader.cpp`
- This phase should NOT add a new `ggml_type`, NOT redesign the scheduler, and NOT attempt op-level CPU/NPU split execution inside one matmul.

## Goal
Implement Phase 2: unify the hybrid manifest policy engine and add deterministic validation infrastructure.

## Required Outcomes
1. Remove manifest parser/resolver duplication.
2. Make `src/llama-model-loader.cpp` consume `common/hybrid-manifest.*` directly.
3. Keep the loader as the control point for final tensor routing.
4. Preserve current phase-1 behavior and backward compatibility where no manifest is provided.
5. Add deterministic validation for:
   - `--hybrid-dry-run`
   - `--hybrid-dump-plan`
   - strict manifest failures
6. Add checked-in example manifests and invalid-manifest fixtures.

## Constraints
- Do not break legacy `-ot`, `--cpu-moe`, `--n-cpu-moe`, or `HYBRID_PATTERN` fallback behavior.
- Do not broaden NPU execution beyond current `MUL_MAT` scope.
- Keep routing decisions deterministic and reproducible.
- Prefer small, reviewable commits.

## Implementation Tasks
- Audit `common/hybrid-manifest.h` and `common/hybrid-manifest.cpp` against the loader-local implementation.
- Refactor loader route planning to use the common manifest parser and resolved-plan types.
- Remove redundant local schema/parser code from `src/llama-model-loader.cpp`.
- Ensure route precedence remains:
  1. manifest route
  2. tensor override regex
  3. legacy placement
- Ensure default sidecar discovery (`model.gguf.hybrid.json`) still works.
- Keep explicit route registration into RKNPU2 config manager intact.
- Add deterministic tests or validation harness coverage for:
  - stable plan output across repeated runs
  - strict rejection of unknown pipelines
  - strict rejection of incompatible quant allowlists
  - strict rejection of impossible alignment/shape requirements
- Add example manifests:
  - dense-balanced
  - dense-cpu-only
  - moe-balanced
- Add bad fixtures:
  - bad-pipeline
  - bad-quant-allow
  - bad-shape
  - bad-profile

## Validation Requirements
- Confirm same GGUF + same manifest yields the same plan output across runs.
- Confirm no-manifest mode preserves legacy behavior.
- Confirm strict mode fails loudly and specifically.
- If full local build is blocked by unrelated platform issues, still run the narrowest possible syntax/build validation for touched targets and state limits clearly.

## Deliverables
- code changes
- test/fixture changes:
  - deterministic tests → `tests/test-hybrid-*.cpp`
  - manifest fixtures → `fixtures/manifest-*.json` (examples: dense-balanced, dense-cpu-only, moe-balanced, bad-pipeline, bad-quant-allow, bad-shape, bad-profile)
- concise summary of architecture after unification
- residual risks
- suggested next benchmark steps on RK3588

## Execution Rule
Do not stop at planning. Implement.