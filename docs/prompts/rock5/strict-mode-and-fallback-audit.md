# ROCK 5 Strict-Mode And Fallback Audit

You are running on or against a ROCK 5 / RK3588 environment for the repository:

`<repository-root>/edgeai-llama.cpp`

## Goal
Audit strict-mode behavior and fallback correctness on real hardware.

## Required Work
- Exercise invalid manifest cases on-device.
- Verify strict mode fails loudly and specifically.
- Verify non-strict mode falls back to CPU in an attributable way.
- Confirm unsupported source types and impossible alignments do not silently route to NPU.
- Confirm no-manifest mode still behaves like legacy mode.

## Deliverables
- pass/fail list for strict-mode cases
- pass/fail list for fallback cases
- notes on any ambiguous or misleading logs
- recommended logging or validation improvements

## Constraints
- This is a correctness session first, not a tuning session.
