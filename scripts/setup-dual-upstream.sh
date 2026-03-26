#!/bin/bash
#
# Configure the repository for the official dual-upstream workflow.
#
# - origin: publishing remote for this fork
# - ik:     primary upstream (ikawrakow/ik_llama.cpp)
# - rk:     secondary upstream layer (KHAEntertainment/rk-llama.cpp)
# - ggml:   optional reference remote for ggml-only work
#

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)
cd "$ROOT_DIR"

IK_URL="${IK_URL:-https://github.com/ikawrakow/ik_llama.cpp.git}"
RK_URL="${RK_URL:-https://github.com/KHAEntertainment/rk-llama.cpp.git}"
GGML_URL="${GGML_URL:-https://github.com/ggml-org/ggml}"

ensure_remote() {
    local name=$1
    local url=$2

    if git remote get-url "$name" >/dev/null 2>&1; then
        git remote set-url "$name" "$url"
    else
        git remote add "$name" "$url"
    fi
}

if git remote get-url upstream >/dev/null 2>&1 && ! git remote get-url ggml >/dev/null 2>&1; then
    git remote rename upstream ggml
fi

ensure_remote ik "$IK_URL"
ensure_remote rk "$RK_URL"
ensure_remote ggml "$GGML_URL"

git config rerere.enabled true

echo "Configured remotes:"
git remote -v
echo
echo "rerere.enabled=$(git config --get rerere.enabled)"
