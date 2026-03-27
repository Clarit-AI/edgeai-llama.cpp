# Phase 4: Hybrid Artifact And Cache Strategy

You are working in the repository at:

`<repository-root>`

## Context
- Phase 1 manifest routing is merged.
- Phase 2 parser/policy unification is assumed complete.
- Phase 3 produced real RK3588 benchmark data and manifest tuning results.
- This phase is about reducing startup cost and making hybrid routing artifacts more operationally useful.
- This is still NOT the phase for inventing a new GGUF tensor encoding or new `ggml_type`.

## Goal
Implement Phase 4: hybrid artifact and prepack/cache strategy.

## Primary Objective
Turn the manifest-driven runtime planner into something operationally efficient by caching and reusing resolved hybrid/NPU preparation work where appropriate.

## Required Outcomes
1. Design and implement a practical cache/artifact strategy for hybrid routing.
2. Avoid recomputing expensive NPU preparation when model + manifest + target config are unchanged.
3. Keep the design phase-compatible with a future true hybrid quant format, but do not implement that format yet.

## Required Work
- Identify what preparation work is worth caching:
  - resolved tensor plan
  - manifest fingerprint
  - per-tensor pipeline decisions
  - NPU packing/prepack metadata
- Define a cache key that includes at minimum:
  - model identity
  - manifest identity
  - profile
  - target device / pipeline assumptions
- Implement cache invalidation rules conservatively.
- Add observability:
  - cache hit/miss
  - cache reuse summary
  - reasons for invalidation
- Ensure cache behavior does not change correctness.
- Ensure strict mode still behaves predictably when cache entries are stale or incompatible.
- Add tests for:
  - cache hit on unchanged config
  - cache miss when manifest changes
  - cache miss when model changes
  - cache miss when pipeline assumptions change

## Non-goals
- no new GGUF extension
- no new `ggml_type`
- no quantizer rewrite
- no speculative artifact format that cannot be validated

## Deliverables
- implementation → `cache/artifacts/`
- test coverage
- concise design note explaining cache/artifact boundaries
- assessment of whether this materially reduces startup or repeated-run cost on RK3588

## Success Criteria
- hybrid routing remains deterministic
- cached reuse is explainable and safe
- benchmark evidence shows whether cache/prepack work is worth carrying forward

## Execution Rule
Implement, validate, and summarize the resulting artifact strategy.