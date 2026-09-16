#!/usr/bin/env bash
# Front door for the CORRECT baseline — runs the 3-stage pipeline
# (schemagen → similarity → detection) via baselines.correct.sweep.
#
# Examples:
#   DATASET=ww GPU=0 bash scripts/correct/run.sh                         # everything, local models
#   DATASET=ww SUBSET=hand-crafted MODEL=gpt-4o bash scripts/correct/run.sh
#   DATASET=correct-error STAGES=schemagen MODEL=gpt-5 bash scripts/correct/run.sh   # schemata only
#   DATASET=ww MODEL=qwen3.5-9b END_IDX=10 DRY_RUN=1 bash scripts/correct/run.sh    # preview
set -euo pipefail

usage() {
  echo "Usage: DATASET=<ww|correct-error|traceelephant|tracertraj> [MODEL=<name>] [SUBSET=<subset>] bash scripts/correct/run.sh" >&2
  echo "Optional env: METHOD=correct|correct_baseline, STAGES=schemagen,similarity,predict," >&2
  echo "  GT=with|without (default: config — without, the paper setting), GPU, START_IDX," >&2
  echo "  END_IDX, DRY_RUN=1, OVERWRITE=1, CONFIG=<path>, EXTRA_SET=\"--set k=v ...\"" >&2
  exit 1
}
[[ -n "${DATASET:-}" ]] || usage

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Config resolution: whichever of <ds>.yaml (local vLLM) / <ds>-api.yaml
# (closed-source) declares MODEL in model_specs; otherwise the local config if
# it exists, else the API one. Only the API configs ship — see
# baselines/correct/configs/README.md to add local models.
CFG_DIR="baselines/correct/configs"
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
[[ -n "${METHOD:-}" ]]    && ARGS+=(--set "methods=[${METHOD}]")
[[ -n "${SUBSET:-}" ]]    && ARGS+=(--set "subsets=[${SUBSET}]")
# GT=with inserts the answer line (the vendored/paper setting is without).
[[ -n "${GT:-}" ]]        && ARGS+=(--gt "${GT}")
[[ -n "${STAGES:-}" ]]    && ARGS+=(--stages "${STAGES}")
[[ -n "${START_IDX:-}" ]] && ARGS+=(--set "start_idx=${START_IDX}")
[[ -n "${END_IDX:-}" ]]   && ARGS+=(--set "end_idx=${END_IDX}")
[[ "${OVERWRITE:-0}" == 1 ]] && ARGS+=(--set "overwrite=true")
[[ "${DRY_RUN:-0}" == 1 ]]   && ARGS+=(--dry-run)
# EXTRA_SET: extra sweep args, word-split (e.g. "--set num_schemata=5").
read -r -a EXTRA_SET_ARR <<< "${EXTRA_SET:-}"

[[ -n "${GPU:-}" ]] && export CUDA_VISIBLE_DEVICES="$GPU"

echo "=== correct | model=${MODEL:-<config>} dataset=${DATASET} subset=${SUBSET:-<all>} stages=${STAGES:-all} config=${CONFIG} ==="
exec python -m baselines.correct.sweep "${ARGS[@]}" ${EXTRA_SET_ARR[@]+"${EXTRA_SET_ARR[@]}"}
