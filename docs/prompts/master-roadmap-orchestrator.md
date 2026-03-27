# Master Roadmap Orchestrator Prompt

You are working in the repository at:

`<repository-root>/edgeai-llama.cpp`

## Context
- This project is a fork combining ik-llama.cpp and rk-llama.cpp work.
- PR #6, which merged phase-1 hybrid manifest routing, is already in `main`.
- The roadmap after phase 1 is structured into:
  - Phase 2: policy unification and deterministic validation
  - Phase 3: RK3588 runtime validation and manifest tuning
  - Phase 4: hybrid artifact/cache strategy
  - Phase 5: hybrid quant decision gate
- Standalone prompts for each phase exist under `docs/prompts/`.

## Goal
Act as the roadmap orchestrator for the remaining hybrid-manifest / hybrid-quantization work.

Your first job is to determine the correct next phase automatically from repository state and available evidence, then execute that phase rather than stopping at planning.

## Phase Selection Rules
Select the next phase using these rules.

**Note:** See `docs/prompts/phase-deliverables-verification.md` for concrete file path mappings of each deliverable criterion below.

### Choose Phase 2 if any of the following are true
- `src/llama-model-loader.cpp` still contains a separate manifest parser/resolver implementation that duplicates `common/hybrid-manifest.*`
- deterministic test coverage for hybrid plan output is missing or incomplete (check: `tests/test-hybrid-*.cpp`)
- manifest fixtures for valid/invalid routing are missing or incomplete (check: `fixtures/manifest-*.json`)
- strict-mode validation coverage is missing

### Choose Phase 3 if all of the following are true
- phase 2 unification appears complete
- deterministic validation exists
- RK3588 runtime benchmark evidence is still missing, partial, stale, or not summarized (check: `benchmarks/rk3588-results.json`)

### Choose Phase 4 if all of the following are true
- phase 2 is complete
- phase 3 produced real RK3588 benchmark and fallback evidence
- startup/prepack/cache cost is still an open operational issue
- there is not yet a safe cache/artifact reuse path for hybrid routing (check: `cache/artifacts/`)

### Choose Phase 5 if all of the following are true
- phase 2 is complete
- phase 3 benchmark evidence exists
- phase 4 cache/artifact work exists or was evaluated
- the remaining question is whether a true hybrid quant format is justified

If evidence is ambiguous, choose the earliest unfinished phase.

## Required Workflow
1. Inspect the repository state to identify which phase is actually next.
2. State the chosen phase and the concrete evidence for that choice.
3. Open the corresponding prompt under `docs/prompts/` and use it as the execution contract.
4. Implement the work for that phase.
5. Validate as far as the host environment allows.
6. Summarize:
   - what phase was executed
   - what changed
   - what remains open
   - whether the next session should advance to the next phase or repeat the current one

## Constraints
- Do not skip phases unless repository evidence clearly shows the earlier phase is already complete.
- Do not broaden scope beyond the selected phase.
- Do not invent a new `ggml_type` or GGUF format unless the selected phase explicitly justifies it.
- Prefer deterministic, benchmarkable, reviewable changes.

## Execution Rule
Do not return a roadmap discussion only. Detect the next phase and execute it.