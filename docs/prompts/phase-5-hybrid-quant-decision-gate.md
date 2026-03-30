# Phase 5: Hybrid Quant Decision Gate

You are working in the repository at:

`<repository-root>/edgeai-llama.cpp`

## Context
- Phase 1 manifest routing is merged.
- Phase 2 unified the policy engine.
- Phase 3 produced RK3588 runtime validation data.
- Phase 4 evaluated caching/prepack/artifact strategies.
- We are now at the decision gate for whether a true hybrid quant format is justified.

## Goal
Execute Phase 5: decide and, only if justified by evidence, prototype the real hybrid quant format direction.

## Two Possible Outcomes
1. No-go: conclude that manifest + policy + cache strategy is sufficient for now.
2. Go: define and prototype the minimum viable true hybrid quant format path.

## Required First Step
Review the benchmark and operational evidence from prior phases and answer:
- What measurable problem remains unsolved by manifest-driven routing plus caching?
- Does a real hybrid quant format solve that problem materially?
- Is the added complexity justified?

## If The Answer Is No
- write the no-go conclusion clearly
- document the reasons
- stop without adding speculative format machinery

## If The Answer Is Yes
Implement only a narrow prototype with strict scope.

## Prototype Scope
- define the minimal artifact or encoding boundary
- keep compatibility strategy explicit
- avoid broad invasive changes until the prototype proves value
- make the prototype measurable on RK3588

## Prototype Must Answer
- what exactly is stored additionally?
- how does it reduce runtime cost or improve execution quality?
- how is backward compatibility handled?
- how is correctness validated?
- how is fallback handled if the artifact is absent or stale?

## Constraints
- no vague architecture astronautics
- no sweeping scheduler rewrite
- no broad refactor without benchmark justification
- every added mechanism must tie to a measured benefit

## Deliverables
- go/no-go decision with rationale
- if go: minimal prototype implementation
- benchmark comparison against manifest-only baseline
- explicit risks and migration implications

## Success Criteria
- the decision is evidence-based
- if a prototype is built, it demonstrates clear measurable value
- if value is not demonstrated, the repo remains simpler rather than more complex

## Execution Rule
Do the decision work first, then implement only what the evidence supports.