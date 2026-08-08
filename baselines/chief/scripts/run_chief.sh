#!/usr/bin/env bash
# Run the CHIEF baseline across models × datasets, sequentially, on one GPU.
#
# Usage:
#   bash baselines/chief/scripts/run_chief.sh [GPU] [extra --set overrides...]
#
# Env knobs:
#   MODELS   space-separated shorthands (default: every model in the configs)
#   DATASETS space-separated datasets  (default: ww correct-error traceelephant)
#
# Examples:
#   bash baselines/chief/scripts/run_chief.sh 4
#   MODELS="qwen3.5-9b" DATASETS="ww" bash baselines/chief/scripts/run_chief.sh 0
#   bash baselines/chief/scripts/run_chief.sh 4 --set overwrite=true
#
# Idempotent: already-written predictions are skipped.
set -euo pipefail

GPU="${1:-0}"; shift || true          # first arg = GPU index (default 0)
EXTRA=("$@")                          # any further args passed straight to the sweep

MODELS="${MODELS:-qwen3.5-9b deepseek-8b llama-3.1-8b qwen3-8b qwen3-14b qwen3-27b}"
DATASETS="${DATASETS:-ww correct-error traceelephant}"

# DeepSeek-R1-Distill always emits a <think> block and gen_max_tokens caps
# thinking + answer combined; CHIEF's long structured stage outputs would be
# eaten by reasoning at the config default. Same fix as prompting's
# run_deepseek.sh. Override via DEEPSEEK_GEN_MAX_TOKENS=...
DEEPSEEK_GEN_MAX_TOKENS="${DEEPSEEK_GEN_MAX_TOKENS:-8192}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs

echo "=== CHIEF on GPU ${GPU} | models: ${MODELS} | datasets: ${DATASETS} ==="
for model in ${MODELS}; do
  for ds in ${DATASETS}; do
    log="logs/chief_${ds}_${model}.log"
    echo ">>> [$(date '+%F %T')] ${model} / ${ds}  (log: ${log})"
    TOKENS=()
    [[ "${model}" == deepseek-8b ]] && TOKENS=(--set "gen_max_tokens=${DEEPSEEK_GEN_MAX_TOKENS}")
    CUDA_VISIBLE_DEVICES="${GPU}" VLLM_LOGGING_LEVEL=WARNING \
      uv run python -m baselines.chief.sweep \
        --config "baselines/chief/configs/${ds}.yaml" \
        --set "models=[${model}]" \
        "${TOKENS[@]}" \
        "${EXTRA[@]}" 2>&1 | tee "${log}"
    echo "<<< [$(date '+%F %T')] done ${model} / ${ds}"
  done
done

echo "=== CHIEF: all runs complete ==="
