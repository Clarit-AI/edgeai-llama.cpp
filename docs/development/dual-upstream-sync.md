# Dual-Upstream Sync Workflow

This fork keeps real ancestry to two upstream sources:

- `ik/main` from [`ikawrakow/ik_llama.cpp`](https://github.com/ikawrakow/ik_llama.cpp)
- `rk/rknpu2` from [`KHAEntertainment/rk-llama.cpp`](https://github.com/KHAEntertainment/rk-llama.cpp)

`origin/main` remains the published integration branch for this repository.

## Remote policy

The repository should use these remote names consistently:

```bash
origin  -> https://github.com/KHAEntertainment/edgeai-llama.cpp
ik      -> https://github.com/ikawrakow/ik_llama.cpp.git
rk      -> https://github.com/KHAEntertainment/rk-llama.cpp.git
ggml    -> https://github.com/ggml-org/ggml
```

`ggml` is retained as a reference remote only. It is not one of the official sync parents for `main`.

Run the setup helper once per clone:

```bash
./scripts/setup-dual-upstream.sh
```

That script also enables Git `rerere` so repeated conflict resolutions are reused across sync cycles.

## Official sync cycle

The default sync order is:

1. `git fetch --all --prune`
2. create a short-lived branch from `main`
3. merge `ik/main`
4. run backend checks
5. merge `rk/rknpu2`
6. run backend checks again
7. review the merge branch and then merge it back into `main`

Use the helper script to rehearse or perform that cycle:

```bash
./scripts/sync-dual-upstream.sh --dry-run
```

That command:

- fast-forwards local `main` from `origin/main`
- creates a `sync/dual-upstream-*` branch
- makes real merge commits for `ik/main` and `rk/rknpu2`
- runs `build/bin/test-backend-ops` if available, otherwise `ctest -R test-backend-ops`
- prints a short graph so the two-parent history is visible

When the rehearsal branch looks correct, run the same script without `--dry-run` and merge the resulting branch into `main`, or pass `--merge-back` to let the script complete the final merge.

## Notes

- The current `rk/rknpu2` branch has no shared ancestry with this repository, so the first bridge merge is a one-time special case.
- The helper script refuses that bridge merge unless you pass `--allow-unrelated-rk-history`.
- Expect the first bridge to produce many conflicts if the RK repository was initialized independently and later drifted across the full tree. Resolve that one-time bridge on a dedicated branch, then future syncs can use normal merge ancestry.
- Do not use selective checkout as the standard update mechanism. It loses ancestry and makes future syncs more manual.
- Do not treat RK support as a patch queue unless you intentionally decide to demote `rk` from official upstream status.
- If your current worktree is dirty, use a fresh worktree or clean up before running the sync helper. The script refuses to run with local changes to avoid accidental branch churn.

## Review fixes on sync PRs

When a sync PR receives review comments after the upstream commits have already been imported:

1. fetch the sync PR branch locally and create a separate fix branch on top of it
2. verify each finding against the current code instead of applying comments blindly
3. add minimal local follow-up commits on top of the imported upstream commits
4. re-run the relevant checks for the touched files
5. update the PR with those follow-up commits
6. merge the PR with a standard merge commit only

Do not squash or rebase sync PRs. The imported upstream commits and their merge ancestry must remain intact so future upstream syncs stay low-friction and RK-specific work remains trackable.
