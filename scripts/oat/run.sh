#!/usr/bin/env bash
# Front door for the OAT baseline — extracts hidden states, trains the neural
# CDE on successful trajectories, and scores the failures, via oat.sweep.
#
# Examples:
#   DATASET=ww MODEL=qwen3.5-9b GPU=0 bash scripts/oat/run.sh       # everything
#   DATASET=ww SUBSET=hand-crafted MODEL=qwen3.5-9b GPU=0 bash scripts/oat/run.sh
#   DATASET=ww MODEL=qwen3.5-9b STAGES=states-train,train bash scripts/oat/run.sh  # train only
#   DATASET=ww MODEL=qwen3.5-9b GT=with GPU=1 bash scripts/oat/run.sh
#   DATASET=ww DRY_RUN=1 bash scripts/oat/run.sh                    # preview
set -euo pipefail

usage() {
  echo "Usage: DATASET=<ww|correct-error|traceelephant|tracertraj> [MODEL=<extractor>] [SUBSET=<subset>] bash scripts/oat/run.sh" >&2
  echo "Optional env: GT=with|without (default: config — without, the vendored setting)," >&2
  echo "  SEEDS=42,43, LAYER=-1, AGGREGATION=mean, TOP_K=3, ALPHA=0.2, EPOCHS, PATIENCE," >&2
  echo "  STAGES=states-train,train,states-test,score, GPU, START_IDX, END_IDX," >&2
  echo "  DRY_RUN=1, OVERWRITE=1, CONFIG=<path>, EXTRA_SET=\"--set k=v ...\"" >&2
  exit 1
}
[[ -n "${DATASET:-}" ]] || usage

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# The representation-based baselines live in their own source root, whose
# hyphen keeps it from colliding with baselines/. `pip install -e .` puts both
# on the path; exporting PYTHONPATH makes the script work without installing.
export PYTHONPATH="baselines-rp${PYTHONPATH:+:$PYTHONPATH}"

# One config per dataset — every extractor is local, so there is no API variant.
CFG_DIR="baselines-rp/oat/configs"
CONFIG="${CONFIG:-${CFG_DIR}/${DATASET}.yaml}"
if [[ ! -f "$CONFIG" ]]; then
  echo "error: no config for DATASET='${DATASET}' (${CFG_DIR}/${DATASET}.yaml)" >&2
  exit 1
fi

ARGS=(--config "$CONFIG")
[[ -n "${MODEL:-}" ]]       && ARGS+=(--set "models=[${MODEL}]")
[[ -n "${SUBSET:-}" ]]      && ARGS+=(--set "subsets=[${SUBSET}]")
# GT=with adds the answer line to the question (the vendored setting is without).
[[ -n "${GT:-}" ]]          && ARGS+=(--gt "${GT}")
[[ -n "${SEEDS:-}" ]]       && ARGS+=(--set "seeds=[${SEEDS}]")
[[ -n "${LAYER:-}" ]]       && ARGS+=(--set "layer=${LAYER}")
[[ -n "${AGGREGATION:-}" ]] && ARGS+=(--set "aggregation=${AGGREGATION}")
[[ -n "${TOP_K:-}" ]]       && ARGS+=(--set "top_k=${TOP_K}")
[[ -n "${ALPHA:-}" ]]       && ARGS+=(--set "alpha=${ALPHA}")
[[ -n "${EPOCHS:-}" ]]      && ARGS+=(--set "epochs=${EPOCHS}")
[[ -n "${PATIENCE:-}" ]]    && ARGS+=(--set "patience=${PATIENCE}")
[[ -n "${STAGES:-}" ]]      && ARGS+=(--stages "${STAGES}")
[[ -n "${START_IDX:-}" ]]   && ARGS+=(--set "start_idx=${START_IDX}")
[[ -n "${END_IDX:-}" ]]     && ARGS+=(--set "end_idx=${END_IDX}")
[[ "${OVERWRITE:-0}" == 1 ]] && ARGS+=(--set "overwrite=true")
[[ "${DRY_RUN:-0}" == 1 ]]   && ARGS+=(--dry-run)
# EXTRA_SET: extra sweep args, word-split (e.g. "--set method_dir_prefix=oat.layer24").
read -r -a EXTRA_SET_ARR <<< "${EXTRA_SET:-}"

[[ -n "${GPU:-}" ]] && export CUDA_VISIBLE_DEVICES="$GPU"
# Trajectories run to ~80k tokens; a segmented allocator keeps one long
# forward pass from fragmenting the arena that the next one needs.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "=== oat | extractor=${MODEL:-<config>} dataset=${DATASET} subset=${SUBSET:-<all>} config=${CONFIG} ==="
exec python -m oat.sweep "${ARGS[@]}" ${EXTRA_SET_ARR[@]+"${EXTRA_SET_ARR[@]}"}
