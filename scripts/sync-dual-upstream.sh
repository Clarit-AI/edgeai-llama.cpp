#!/bin/bash
#
# Rehearse or perform the official dual-upstream sync workflow:
#   1. fetch all remotes
#   2. create a short-lived branch from main
#   3. merge ik/main
#   4. run backend checks
#   5. merge rk/rknpu2
#   6. run backend checks again
#   7. optionally merge the sync branch back into main
#

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)
cd "$ROOT_DIR"

MAIN_BRANCH="${MAIN_BRANCH:-main}"
IK_REMOTE="${IK_REMOTE:-ik}"
IK_BRANCH="${IK_BRANCH:-main}"
RK_REMOTE="${RK_REMOTE:-rk}"
RK_BRANCH="${RK_BRANCH:-rknpu2}"
SYNC_BRANCH="${SYNC_BRANCH:-sync/dual-upstream-$(date +%Y%m%d-%H%M%S)}"

DRY_RUN=0
RUN_TESTS=1
MERGE_BACK=0
ALLOW_UNRELATED_RK=0

usage() {
    cat <<EOF
Usage: $0 [options]

Options:
  --dry-run            Merge on a short-lived sync branch only
  --merge-back         Merge the completed sync branch back into ${MAIN_BRANCH}
  --no-test            Skip backend checks after each merge
  --allow-unrelated-rk-history
                       Allow the one-time unrelated-history bridge merge for ${RK_REMOTE}/${RK_BRANCH}
  --sync-branch NAME   Override the generated sync branch name
  --main-branch NAME   Override the integration branch (default: ${MAIN_BRANCH})
  --ik-remote NAME     Override the primary upstream remote (default: ${IK_REMOTE})
  --ik-branch NAME     Override the primary upstream branch (default: ${IK_BRANCH})
  --rk-remote NAME     Override the secondary upstream remote (default: ${RK_REMOTE})
  --rk-branch NAME     Override the secondary upstream branch (default: ${RK_BRANCH})
EOF
}

require_option_value() {
    local option=$1
    if [[ $# -lt 2 || -z "${2:-}" ]]; then
        echo "Error: ${option} requires a value" >&2
        exit 1
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --merge-back)
            MERGE_BACK=1
            shift
            ;;
        --no-test)
            RUN_TESTS=0
            shift
            ;;
        --allow-unrelated-rk-history)
            ALLOW_UNRELATED_RK=1
            shift
            ;;
        --sync-branch)
            require_option_value "$1" "$2"
            SYNC_BRANCH="$2"
            shift 2
            ;;
        --main-branch)
            require_option_value "$1" "$2"
            MAIN_BRANCH="$2"
            shift 2
            ;;
        --ik-remote)
            require_option_value "$1" "$2"
            IK_REMOTE="$2"
            shift 2
            ;;
        --ik-branch)
            require_option_value "$1" "$2"
            IK_BRANCH="$2"
            shift 2
            ;;
        --rk-remote)
            require_option_value "$1" "$2"
            RK_REMOTE="$2"
            shift 2
            ;;
        --rk-branch)
            require_option_value "$1" "$2"
            RK_BRANCH="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ "$DRY_RUN" -eq 1 && "$MERGE_BACK" -eq 1 ]]; then
    echo "--dry-run and --merge-back are mutually exclusive" >&2
    exit 1
fi

if [[ -n "$(git status --porcelain)" ]]; then
    echo "Working tree is not clean. Commit, stash, or use a separate worktree before syncing." >&2
    exit 1
fi

require_remote_branch() {
    local remote=$1
    local branch=$2

    if ! git show-ref --verify --quiet "refs/remotes/${remote}/${branch}"; then
        echo "Missing remote branch ${remote}/${branch}. Run setup and fetch first." >&2
        exit 1
    fi
}

run_backend_checks() {
    if [[ "$RUN_TESTS" -eq 0 ]]; then
        return 0
    fi

    if [[ -x ./build/bin/test-backend-ops ]]; then
        ./build/bin/test-backend-ops
        return 0
    fi

    if command -v ctest >/dev/null 2>&1 && [[ -d ./build ]]; then
        ctest --test-dir ./build --output-on-failure -R test-backend-ops
        return 0
    fi

    echo "No backend test binary found; skipping tests." >&2
}

merge_remote_branch() {
    local remote=$1
    local branch=$2
    local message=$3
    local merge_args=(--no-ff -m "$message")

    if ! git merge-base HEAD "${remote}/${branch}" >/dev/null 2>&1; then
        if [[ "$remote" == "$RK_REMOTE" && "$branch" == "$RK_BRANCH" && "$ALLOW_UNRELATED_RK" -eq 1 ]]; then
            echo "No shared ancestry with ${remote}/${branch}; allowing the one-time unrelated-history bridge merge." >&2
        else
            echo "No shared ancestry with ${remote}/${branch}." >&2
            echo "If this is the intentional first bridge merge for ${RK_REMOTE}/${RK_BRANCH}, rerun with --allow-unrelated-rk-history." >&2
            exit 1
        fi
        merge_args+=(--allow-unrelated-histories)
    fi

    git merge "${merge_args[@]}" "${remote}/${branch}"
}

git fetch --all --prune

require_remote_branch "$IK_REMOTE" "$IK_BRANCH"
require_remote_branch "$RK_REMOTE" "$RK_BRANCH"
require_remote_branch origin "$MAIN_BRANCH"

git switch "$MAIN_BRANCH"
git pull --ff-only origin "$MAIN_BRANCH"

if git show-ref --verify --quiet "refs/heads/${SYNC_BRANCH}"; then
    echo "Branch ${SYNC_BRANCH} already exists." >&2
    exit 1
fi

git switch -c "$SYNC_BRANCH"

merge_remote_branch "$IK_REMOTE" "$IK_BRANCH" "merge: sync ${IK_REMOTE}/${IK_BRANCH}"
run_backend_checks

merge_remote_branch "$RK_REMOTE" "$RK_BRANCH" "merge: sync ${RK_REMOTE}/${RK_BRANCH}"
run_backend_checks

echo
git log --graph --oneline --decorate -n 8
echo

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "Dry run complete on ${SYNC_BRANCH}."
    echo "Review the branch, then delete it or merge it manually."
    exit 0
fi

if [[ "$MERGE_BACK" -eq 1 ]]; then
    git switch "$MAIN_BRANCH"
    git merge --no-ff -m "merge: integrate ${SYNC_BRANCH}" "$SYNC_BRANCH"
    echo "Merged ${SYNC_BRANCH} back into ${MAIN_BRANCH}."
    exit 0
fi

echo "Sync branch ${SYNC_BRANCH} is ready for review."
echo "Merge it into ${MAIN_BRANCH} after validation."