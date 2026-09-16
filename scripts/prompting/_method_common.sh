#!/usr/bin/env bash
# Shared driver for the per-method scripts (all_at_once.sh, step_by_step.sh,
# binary_search.sh). Not run directly — each method script sources this with
# METHOD already set. See scripts/README.md for usage.
set -euo pipefail

usage() {
  echo "Usage: MODEL=<name> DATASET=<ww|correct-error|correct-error-nogt|traceelephant|tracertraj> [SUBSET=<subset>] scripts/prompting/${METHOD}.sh" >&2
  echo "Optional env: GT=with|without, GPU, START_IDX, END_IDX, DRY_RUN=1, OVERWRITE=1, EXTRA_SET=\"--set k=v ...\"" >&2
  exit 1
}
[[ -n "${MODEL:-}" && -n "${DATASET:-}" ]] || usage

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Config resolution: the API config if it declares MODEL in model_specs,
# otherwise the local-vLLM config for the dataset.
CFG_DIR="baselines/prompting/configs"
CONFIG="${CONFIG:-}"
if [[ -z "$CONFIG" ]]; then
  if [[ -f "${CFG_DIR}/${DATASET}-api.yaml" ]] \
     && grep -qE "^  ${MODEL}:" "${CFG_DIR}/${DATASET}-api.yaml"; then
    CONFIG="${CFG_DIR}/${DATASET}-api.yaml"
  elif [[ -f "${CFG_DIR}/${DATASET}.yaml" ]]; then
    CONFIG="${CFG_DIR}/${DATASET}.yaml"
  else
    echo "error: no config for DATASET='${DATASET}' (${CFG_DIR}/${DATASET}[-api].yaml)" >&2
    exit 1
  fi
fi

ARGS=(--config "$CONFIG"
      --set "models=[${MODEL}]"
      --set "methods=[${METHOD}]")
[[ -n "${SUBSET:-}" ]]    && ARGS+=(--set "subsets=[${SUBSET}]")
# GT=without drops the answer line from prompts and mirrors into outputs-nogt/.
[[ -n "${GT:-}" ]]        && ARGS+=(--gt "${GT}")
[[ -n "${START_IDX:-}" ]] && ARGS+=(--set "start_idx=${START_IDX}")
[[ -n "${END_IDX:-}" ]]   && ARGS+=(--set "end_idx=${END_IDX}")
[[ "${OVERWRITE:-0}" == 1 ]] && ARGS+=(--set "overwrite=true")
[[ "${DRY_RUN:-0}" == 1 ]]   && ARGS+=(--dry-run)
# EXTRA_SET: extra sweep args, word-split (e.g. "--set step_mode=batch").
read -r -a EXTRA_SET_ARR <<< "${EXTRA_SET:-}"

[[ -n "${GPU:-}" ]] && export CUDA_VISIBLE_DEVICES="$GPU"

echo "=== ${METHOD} | model=${MODEL} dataset=${DATASET} subset=${SUBSET:-<all>} config=${CONFIG} ==="
exec python -m baselines.prompting.sweep "${ARGS[@]}" ${EXTRA_SET_ARR[@]+"${EXTRA_SET_ARR[@]}"}
