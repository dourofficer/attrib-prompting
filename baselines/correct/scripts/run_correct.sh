#!/usr/bin/env bash
# Run the CORRECT baseline across models × datasets, sequentially, on one GPU.
# Per (model, dataset) the sweep runs three idempotent stages: schema generation
# (vLLM), trajectory similarities (BGE-M3), and schema-guided detection (vLLM).
#
# Usage (override knobs via env — matches scripts/_common.sh conventions):
#   GPU=5 bash baselines/correct/scripts/run_correct.sh
#   GPU=5 MODELS="qwen3.5-9b" DATASETS="ww" bash baselines/correct/scripts/run_correct.sh
#   GPU=5 DRY_RUN=1 bash baselines/correct/scripts/run_correct.sh          # preview commands
#   GPU=5 EXTRA_SET="--set overwrite=true" bash baselines/correct/scripts/run_correct.sh
#
# Backward-compatible positional form still works: `... run_correct.sh 5`.
# Env knobs:
#   GPU      CUDA device index (default 0)
#   MODELS   space-separated shorthands (default: qwen3.5-9b deepseek-8b)
#   DATASETS space-separated datasets  (default: ww traceelephant correct-error)
#   DRY_RUN  1 → forward --dry-run to the sweep (prints commands without running)
#   EXTRA_SET  extra `--set k=v` overrides forwarded to every sweep invocation
#
# Idempotent: already-written schemata / similarities / predictions are skipped.
set -euo pipefail

# GPU: env wins (GPU=5 bash ...); else a bare numeric first arg (old form); else 0.
if [[ -z "${GPU:-}" ]]; then
  if [[ "${1:-}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then GPU="$1"; shift; else GPU=0; fi
fi
EXTRA=("$@")                          # any further args passed straight to the sweep

MODELS="${MODELS:-qwen3.5-9b deepseek-8b}"
DATASETS="${DATASETS:-ww traceelephant correct-error}"
DRY=(); [[ "${DRY_RUN:-0}" == 1 ]] && DRY=(--dry-run)
read -r -a EXTRA_SET_ARR <<< "${EXTRA_SET:-}"

# DeepSeek-R1-Distill always emits a <think> block and gen_max_tokens caps
# thinking + answer combined; the vendored 1024 would be eaten by reasoning
# before the answer (and mid-think truncated schemata would be garbage). Same
# fix as prompting's run_deepseek.sh / chief's run_chief.sh — applied to BOTH
# the detection and schema-generation stages. Override via DEEPSEEK_GEN_MAX_TOKENS=...
DEEPSEEK_GEN_MAX_TOKENS="${DEEPSEEK_GEN_MAX_TOKENS:-8192}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs

echo "=== CORRECT on GPU ${GPU} | models: ${MODELS} | datasets: ${DATASETS} ==="
for model in ${MODELS}; do
  for ds in ${DATASETS}; do
    log="logs/correct_${ds}_${model}.log"
    echo ">>> [$(date '+%F %T')] ${model} / ${ds}  (log: ${log})"
    TOKENS=()
    [[ "${model}" == deepseek-8b ]] && TOKENS=(
      --set "gen_max_tokens=${DEEPSEEK_GEN_MAX_TOKENS}"
      --set "schema_gen.max_tokens=${DEEPSEEK_GEN_MAX_TOKENS}"
    )
    CUDA_VISIBLE_DEVICES="${GPU}" VLLM_LOGGING_LEVEL=WARNING \
      uv run python -m baselines.correct.sweep \
        --config "baselines/correct/configs/${ds}.yaml" \
        --set "models=[${model}]" \
        "${TOKENS[@]}" \
        "${DRY[@]}" \
        "${EXTRA_SET_ARR[@]}" \
        "${EXTRA[@]}" 2>&1 | tee "${log}"
    echo "<<< [$(date '+%F %T')] done ${model} / ${ds}"
  done
done

echo "=== CORRECT: all runs complete ==="
