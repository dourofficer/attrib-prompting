#!/usr/bin/env bash
# Front door for the RAFFLES baseline — runs the Judge-Evaluator loop via
# baselines.raffles.sweep.
#
# Examples:
#   DATASET=ww MODEL=gpt-4o bash scripts/raffles/run.sh                  # everything
#   DATASET=ww SUBSET=hand-crafted MODEL=gpt-4o bash scripts/raffles/run.sh
#   DATASET=ww MODEL=gpt-4o MAX_ITERS=5 bash scripts/raffles/run.sh     # paper's K=5
#   DATASET=ww MODEL=gpt-4o END_IDX=10 DRY_RUN=1 bash scripts/raffles/run.sh  # preview
set -euo pipefail

usage() {
  echo "Usage: DATASET=<ww|correct-error|traceelephant> [MODEL=<name>] [SUBSET=<subset>] bash scripts/raffles/run.sh" >&2
  echo "Optional env: GT=with|without (default: config — without, the paper setting)," >&2
  echo "  MAX_ITERS=<K>, THRESHOLD=<C>, GPU, START_IDX, END_IDX, DRY_RUN=1, OVERWRITE=1," >&2
  echo "  CONFIG=<path>, EXTRA_SET=\"--set k=v ...\"" >&2
  exit 1
}
[[ -n "${DATASET:-}" ]] || usage

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Config resolution: whichever of <ds>.yaml (local vLLM) / <ds>-api.yaml
# (closed-source) declares MODEL in model_specs; otherwise the local config if
# it exists, else the API one. Only the API configs ship — see
# baselines/raffles/configs/README.md to add local models.
CFG_DIR="baselines/raffles/configs"
LOCAL_CFG="${CFG_DIR}/${DATASET}.yaml"
API_CFG="${CFG_DIR}/${DATASET}-api.yaml"
CONFIG="${CONFIG:-}"
if [[ -z "$CONFIG" && -n "${MODEL:-}" ]]; then
  for c in "$LOCAL_CFG" "$API_CFG"; do
    [[ -f "$c" ]] && grep -qE "^  ${MODEL}:" "$c" && { CONFIG="$c"; break; }
  done
fi
if [[ -z "$CONFIG" ]]; then
  if   [[ -f "$LOCAL_CFG" ]]; then CONFIG="$LOCAL_CFG"
  elif [[ -f "$API_CFG"   ]]; then CONFIG="$API_CFG"
  else
    echo "error: no config for DATASET='${DATASET}' (${CFG_DIR}/${DATASET}[-api].yaml)" >&2
    exit 1
  fi
fi

ARGS=(--config "$CONFIG")
[[ -n "${MODEL:-}" ]]     && ARGS+=(--set "models=[${MODEL}]")
[[ -n "${SUBSET:-}" ]]    && ARGS+=(--set "subsets=[${SUBSET}]")
# GT=with adds the answer line (the paper setting is without).
[[ -n "${GT:-}" ]]        && ARGS+=(--gt "${GT}")
[[ -n "${MAX_ITERS:-}" ]] && ARGS+=(--set "max_iters=${MAX_ITERS}")
[[ -n "${THRESHOLD:-}" ]] && ARGS+=(--set "threshold=${THRESHOLD}")
[[ -n "${START_IDX:-}" ]] && ARGS+=(--set "start_idx=${START_IDX}")
[[ -n "${END_IDX:-}" ]]   && ARGS+=(--set "end_idx=${END_IDX}")
[[ "${OVERWRITE:-0}" == 1 ]] && ARGS+=(--set "overwrite=true")
[[ "${DRY_RUN:-0}" == 1 ]]   && ARGS+=(--dry-run)
# EXTRA_SET: extra sweep args, word-split (e.g. "--set method_dir=raffles.k5").
read -r -a EXTRA_SET_ARR <<< "${EXTRA_SET:-}"

[[ -n "${GPU:-}" ]] && export CUDA_VISIBLE_DEVICES="$GPU"

echo "=== raffles | model=${MODEL:-<config>} dataset=${DATASET} subset=${SUBSET:-<all>} config=${CONFIG} ==="
exec python -m baselines.raffles.sweep "${ARGS[@]}" ${EXTRA_SET_ARR[@]+"${EXTRA_SET_ARR[@]}"}
