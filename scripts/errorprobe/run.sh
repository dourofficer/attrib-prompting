#!/usr/bin/env bash
# Front door for the ErrorProbe baseline — runs the Analyzer→Verifier
# diagnosis via baselines.errorprobe.sweep.
#
# Examples:
#   DATASET=ww MODEL=gpt-4o bash scripts/errorprobe/run.sh                 # both modes
#   DATASET=ww SUBSET=hand-crafted MODEL=gpt-4o bash scripts/errorprobe/run.sh
#   DATASET=ww MODEL=gpt-4o MODE=truncated bash scripts/errorprobe/run.sh  # cheap mode only
#   DATASET=ww MODEL=gpt-4o MODE=paper bash scripts/errorprobe/run.sh      # the paper pipeline (opt-in)
#   DATASET=ww MODEL=gpt-4o END_IDX=10 DRY_RUN=1 bash scripts/errorprobe/run.sh  # preview
set -euo pipefail

usage() {
  echo "Usage: DATASET=<ww|correct-error|traceelephant> [MODEL=<name>] [SUBSET=<subset>] bash scripts/errorprobe/run.sh" >&2
  echo "Optional env: MODE=truncated|backward|paper (default: config — the two vendored modes)," >&2
  echo "  GT=with|without (default: config — without, the vendored setting)," >&2
  echo "  GPU, START_IDX, END_IDX, DRY_RUN=1, OVERWRITE=1," >&2
  echo "  CONFIG=<path>, EXTRA_SET=\"--set k=v ...\"" >&2
  exit 1
}
[[ -n "${DATASET:-}" ]] || usage

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Config resolution: whichever of <ds>.yaml (local vLLM) / <ds>-api.yaml
# (closed-source) declares MODEL in model_specs; otherwise the local config if
# it exists, else the API one. Only the API configs ship — see
# baselines/errorprobe/configs/README.md to add local models.
CFG_DIR="baselines/errorprobe/configs"
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
[[ -n "${MODE:-}" ]]      && ARGS+=(--set "modes=[${MODE}]")
# GT=with adds the answer line (the vendored setting is without).
[[ -n "${GT:-}" ]]        && ARGS+=(--gt "${GT}")
[[ -n "${START_IDX:-}" ]] && ARGS+=(--set "start_idx=${START_IDX}")
[[ -n "${END_IDX:-}" ]]   && ARGS+=(--set "end_idx=${END_IDX}")
[[ "${OVERWRITE:-0}" == 1 ]] && ARGS+=(--set "overwrite=true")
[[ "${DRY_RUN:-0}" == 1 ]]   && ARGS+=(--dry-run)
# EXTRA_SET: extra sweep args, word-split (e.g. "--set method_dir=errorprobe.x").
read -r -a EXTRA_SET_ARR <<< "${EXTRA_SET:-}"

[[ -n "${GPU:-}" ]] && export CUDA_VISIBLE_DEVICES="$GPU"

echo "=== errorprobe | model=${MODEL:-<config>} dataset=${DATASET} subset=${SUBSET:-<all>} mode=${MODE:-<config>} config=${CONFIG} ==="
exec python -m baselines.errorprobe.sweep "${ARGS[@]}" ${EXTRA_SET_ARR[@]+"${EXTRA_SET_ARR[@]}"}
