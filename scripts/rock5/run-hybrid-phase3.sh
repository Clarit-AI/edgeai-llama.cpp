#!/usr/bin/env bash

set -euo pipefail

BUILD_DIR="./build"
MANIFEST_DIR="./examples/hybrid-manifests"
OUT_DIR="./reports/rock5/$(date +%Y%m%d-%H%M%S)"
DENSE_MODEL=""
MOE_MODEL=""
PROMPT_TOKENS=256
GEN_TOKENS=64
PROMPT="Explain in one paragraph why deterministic routing validation matters for Rockchip NPU benchmarking."

usage() {
    cat <<'EOF'
usage: scripts/rock5/run-hybrid-phase3.sh [options]

options:
  --build-dir DIR        build directory containing bin/llama-cli
  --manifest-dir DIR     directory containing runtime manifests
  --out-dir DIR          output directory for logs, plans, and report
  --dense-model FILE     dense model GGUF path
  --moe-model FILE       MoE model GGUF path
  --prompt-tokens N      prompt tokens to request in report metadata
  --gen-tokens N         generation tokens per run
  -h, --help             show this message
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        --manifest-dir)
            MANIFEST_DIR="$2"
            shift 2
            ;;
        --out-dir)
            OUT_DIR="$2"
            shift 2
            ;;
        --dense-model)
            DENSE_MODEL="$2"
            shift 2
            ;;
        --moe-model)
            MOE_MODEL="$2"
            shift 2
            ;;
        --prompt-tokens)
            PROMPT_TOKENS="$2"
            shift 2
            ;;
        --gen-tokens)
            GEN_TOKENS="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ -z "${DENSE_MODEL}" ]]; then
    echo "error: --dense-model is required" >&2
    exit 1
fi

LLAMA_CLI="${BUILD_DIR}/bin/llama-cli"
if [[ ! -x "${LLAMA_CLI}" ]]; then
    echo "error: expected executable ${LLAMA_CLI}" >&2
    exit 1
fi

mkdir -p "${OUT_DIR}"

run_case() {
    local label="$1"
    local model_path="$2"
    local ngl="$3"
    local manifest_path="$4"
    local profile_name="$5"

    local log_path="${OUT_DIR}/${label}.log"
    local dry_run_path="${OUT_DIR}/${label}.dry-run.log"
    local plan_path="${OUT_DIR}/${label}.plan.json"

    echo "==> ${label}"

    local -a env_cmd=()
    if [[ -n "${manifest_path}" ]]; then
        env_cmd+=("HYBRID_MANIFEST=${manifest_path}")
        env_cmd+=("HYBRID_PROFILE=${profile_name}")
        env_cmd+=("HYBRID_STRICT=1")
    fi

    if [[ -n "${manifest_path}" ]]; then
        env "${env_cmd[@]}" \
            "${LLAMA_CLI}" \
            -m "${model_path}" \
            --n-gpu-layers "${ngl}" \
            --hybrid-dry-run \
            --hybrid-dump-plan "${plan_path}" \
            -p "${PROMPT}" \
            -n 1 \
            >"${dry_run_path}" 2>&1
    fi

    env "${env_cmd[@]}" \
        "${LLAMA_CLI}" \
        -m "${model_path}" \
        --n-gpu-layers "${ngl}" \
        -p "${PROMPT}" \
        -n "${GEN_TOKENS}" \
        >"${log_path}" 2>&1
}

extract_metric() {
    local log_path="$1"
    local prefix="$2"

    python3 - "$log_path" "$prefix" <<'PY'
import re
import sys

log_path = sys.argv[1]
prefix = sys.argv[2]
text = open(log_path, "r", encoding="utf-8", errors="replace").read()
pattern = re.compile(rf"{re.escape(prefix)}.*?,\s*([0-9]+(?:\.[0-9]+)?)\s+tokens per second", re.IGNORECASE)
matches = pattern.findall(text)
print(matches[-1] if matches else "")
PY
}

extract_fallbacks() {
    local log_path="$1"

    python3 - "$log_path" <<'PY'
import re
import sys

text = open(sys.argv[1], "r", encoding="utf-8", errors="replace").read().splitlines()
needles = re.compile(r"fallback|unsupported|align|cpu", re.IGNORECASE)
hits = []
for line in text:
    if needles.search(line):
        hits.append(line.strip())
seen = []
for line in hits:
    if line not in seen:
        seen.append(line)
print(" | ".join(seen[:3]))
PY
}

render_report() {
    local report_path="${OUT_DIR}/REPORT.md"

    {
        echo "# RK3588 Hybrid Runtime Validation"
        echo
        echo "- Date: $(date -Iseconds)"
        echo "- Host: $(hostname)"
        echo "- Git branch: $(git rev-parse --abbrev-ref HEAD)"
        echo "- Git commit: $(git rev-parse HEAD)"
        echo "- Build dir: ${BUILD_DIR}"
        echo "- Manifest dir: ${MANIFEST_DIR}"
        echo "- Prompt tokens target: ${PROMPT_TOKENS}"
        echo "- Generation tokens target: ${GEN_TOKENS}"
        echo
        echo "| Case | Manifest | Prompt tok/s | Gen tok/s | Fallback notes |"
        echo "|---|---|---:|---:|---|"

        while IFS='|' read -r label manifest_name log_path; do
            local pp
            local tg
            local notes
            pp="$(extract_metric "${log_path}" "prompt eval time =")"
            tg="$(extract_metric "${log_path}" "eval time =")"
            notes="$(extract_fallbacks "${log_path}")"
            echo "| ${label} | ${manifest_name} | ${pp:-n/a} | ${tg:-n/a} | ${notes:-} |"
        done < "${OUT_DIR}/cases.txt"

        echo
        echo "## Artifacts"
        echo
        echo "- Logs: \`${OUT_DIR}/*.log\`"
        echo "- Plans: \`${OUT_DIR}/*.plan.json\`"
        echo "- Dry runs: \`${OUT_DIR}/*.dry-run.log\`"
        echo
        echo "## Manual Follow-up"
        echo
        echo "- Compare repeated runs for noise before claiming wins."
        echo "- Inspect any fallback lines tied to unsupported source types, alignment mismatches, or CPU-owned tensors."
        echo "- Revise only one manifest dimension at a time."
    } > "${report_path}"
}

: > "${OUT_DIR}/cases.txt"

run_case "dense-cpu-only" "${DENSE_MODEL}" 0 "" "cpu-only"
echo "dense-cpu-only|none|${OUT_DIR}/dense-cpu-only.log" >> "${OUT_DIR}/cases.txt"

run_case "dense-legacy-npu" "${DENSE_MODEL}" 99 "" "legacy-npu"
echo "dense-legacy-npu|none|${OUT_DIR}/dense-legacy-npu.log" >> "${OUT_DIR}/cases.txt"

run_case "dense-manifest-balanced" "${DENSE_MODEL}" 99 "${MANIFEST_DIR}/dense-balanced.json" "dense-balanced"
echo "dense-manifest-balanced|dense-balanced|${OUT_DIR}/dense-manifest-balanced.log" >> "${OUT_DIR}/cases.txt"

run_case "dense-manifest-npu-heavy" "${DENSE_MODEL}" 99 "${MANIFEST_DIR}/dense-npu-heavy.json" "dense-npu-heavy"
echo "dense-manifest-npu-heavy|dense-npu-heavy|${OUT_DIR}/dense-manifest-npu-heavy.log" >> "${OUT_DIR}/cases.txt"

if [[ -n "${MOE_MODEL}" ]]; then
    run_case "moe-cpu-only" "${MOE_MODEL}" 0 "" "cpu-only"
    echo "moe-cpu-only|none|${OUT_DIR}/moe-cpu-only.log" >> "${OUT_DIR}/cases.txt"

    run_case "moe-legacy-npu" "${MOE_MODEL}" 99 "" "legacy-npu"
    echo "moe-legacy-npu|none|${OUT_DIR}/moe-legacy-npu.log" >> "${OUT_DIR}/cases.txt"

    run_case "moe-manifest-balanced" "${MOE_MODEL}" 99 "${MANIFEST_DIR}/moe-balanced.json" "moe-balanced"
    echo "moe-manifest-balanced|moe-balanced|${OUT_DIR}/moe-manifest-balanced.log" >> "${OUT_DIR}/cases.txt"
fi

render_report

echo "report written to ${OUT_DIR}/REPORT.md"
