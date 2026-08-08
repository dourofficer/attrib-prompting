#!/usr/bin/env bash
# Run the prompting baselines for deepseek-8b across ALL datasets, sequentially.
#
# Usage (override knobs via env — matches scripts/_common.sh conventions):
#   GPU=5 bash baselines/prompting/scripts/run_deepseek.sh
#   GPU=5 DATASETS="traceelephant correct-error" bash baselines/prompting/scripts/run_deepseek.sh
#   GPU=5 DRY_RUN=1 bash baselines/prompting/scripts/run_deepseek.sh          # preview commands
#   GPU=5 EXTRA_SET="--set overwrite=true" bash baselines/prompting/scripts/run_deepseek.sh
#
# Backward-compatible positional form still works: `... run_deepseek.sh 5`.
# Runs on one GPU; pair with run_qwen.sh on a different GPU to parallelise the
# two models. Idempotent: already-written predictions are skipped.
set -euo pipefail

MODEL="deepseek-8b"

# GPU: env wins (GPU=5 bash ...); else a bare numeric first arg (old form); else 0.
if [[ -z "${GPU:-}" ]]; then
  if [[ "${1:-}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then GPU="$1"; shift; else GPU=0; fi
fi
EXTRA=("$@")                          # any further args passed straight to the sweep

# DeepSeek-R1-Distill always emits a <think> block, and gen_max_tokens caps
# thinking + answer combined. 1024 (fine for non-thinking qwen) would be eaten by
# reasoning before the answer is produced, yielding empty/unparsable output. Give
# it enough headroom to finish reasoning AND emit the answer. Override via env:
# GEN_MAX_TOKENS=... or append your own --set gen_max_tokens=... in EXTRA_SET.
GEN_MAX_TOKENS="${GEN_MAX_TOKENS:-8192}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs

# DATASETS: space-separated env override; default = all three.
read -r -a DATASETS <<< "${DATASETS:-ww traceelephant correct-error}"
# DRY_RUN=1 → forward --dry-run to the sweep (prints commands without running).
DRY=(); [[ "${DRY_RUN:-0}" == 1 ]] && DRY=(--dry-run)
# EXTRA_SET: extra `--set k=v` overrides forwarded to every sweep invocation.
read -r -a EXTRA_SET_ARR <<< "${EXTRA_SET:-}"

echo "=== ${MODEL} on GPU ${GPU} | datasets: ${DATASETS[*]} ==="
for ds in "${DATASETS[@]}"; do
  log="logs/${ds}_deepseek.log"
  echo ">>> [$(date '+%F %T')] ${MODEL} / ${ds}  (log: ${log})"
  CUDA_VISIBLE_DEVICES="${GPU}" VLLM_LOGGING_LEVEL=WARNING \
    uv run python -m baselines.prompting.sweep \
      --config "baselines/prompting/configs/${ds}.yaml" \
      --set "models=[${MODEL}]" \
      --set "gen_max_tokens=${GEN_MAX_TOKENS}" \
      "${DRY[@]}" "${EXTRA_SET_ARR[@]}" "${EXTRA[@]}" 2>&1 | tee "${log}"
  echo "<<< [$(date '+%F %T')] done ${MODEL} / ${ds}"
done

echo "=== ${MODEL}: all datasets complete ==="
