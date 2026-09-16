#!/usr/bin/env bash
# Front door for the StepFinder baseline — embeds every step, trains the
# temporal scorer on labelled failures, and scores the corpus, via
# stepfinder.sweep.
#
# Examples:
#   DATASET=ww MODEL=qwen3-embedding-0.6b GPU=0 bash scripts/stepfinder/run.sh
#   DATASET=ww SUBSET=hand-crafted MODEL=qwen3-embedding-0.6b GPU=0 bash scripts/stepfinder/run.sh
#   DATASET=ww MODEL=qwen3-embedding-0.6b STAGES=feats-train,train bash scripts/stepfinder/run.sh
#   DATASET=ww MODEL=qwen3-embedding-0.6b PROTOCOL=in-corpus GPU=1 bash scripts/stepfinder/run.sh
#   DATASET=ww MODEL=qwen3-embedding-0.6b GT=with GPU=1 bash scripts/stepfinder/run.sh
#   DATASET=ww DRY_RUN=1 bash scripts/stepfinder/run.sh
set -euo pipefail

usage() {
  echo "Usage: DATASET=<ww|correct-error|traceelephant|tracertraj> [MODEL=<encoder>] [SUBSET=<subset>] bash scripts/stepfinder/run.sh" >&2
  echo "Optional env: GT=with|without (default: config — without, the vendored setting)," >&2
  echo "  PROTOCOL=regen|in-corpus (comma-separated), MODEL_SELECTION=val|test|vendored," >&2
  echo "  REDUCE=slice|pca (how the wide embedding becomes the 128/32 input)," >&2
  echo "  SEEDS=42,43, EVAL_SEEDS=1,2,3, AGENT_NORMALIZE=standardize|raw," >&2
  echo "  EPOCHS, BATCH_SIZE, PATIENCE, TOP_K, STAGES=feats-train,train,feats-test,score," >&2
  echo "  GPU, START_IDX, END_IDX, DRY_RUN=1, OVERWRITE=1, CONFIG=<path>, EXTRA_SET=\"--set k=v ...\"" >&2
  exit 1
}
[[ -n "${DATASET:-}" ]] || usage

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# The representation-based baselines live in their own source root, whose
# hyphen keeps it from colliding with baselines/. `pip install -e .` puts both
# on the path; exporting PYTHONPATH makes the script work without installing.
export PYTHONPATH="baselines-rp${PYTHONPATH:+:$PYTHONPATH}"

# One config per dataset — every encoder is local, so there is no API variant.
CFG_DIR="baselines-rp/stepfinder/configs"
CONFIG="${CONFIG:-${CFG_DIR}/${DATASET}.yaml}"
if [[ ! -f "$CONFIG" ]]; then
  echo "error: no config for DATASET='${DATASET}' (${CFG_DIR}/${DATASET}.yaml)" >&2
  exit 1
fi

ARGS=(--config "$CONFIG")
[[ -n "${MODEL:-}" ]]            && ARGS+=(--set "models=[${MODEL}]")
[[ -n "${SUBSET:-}" ]]           && ARGS+=(--set "subsets=[${SUBSET}]")
# GT=with appends the answer to the first step's content (the vendored setting
# is without, which is what the training corpus was built under).
[[ -n "${GT:-}" ]]               && ARGS+=(--gt "${GT}")
[[ -n "${PROTOCOL:-}" ]]         && ARGS+=(--protocol "${PROTOCOL}")
[[ -n "${SEEDS:-}" ]]            && ARGS+=(--set "seeds=[${SEEDS}]")
[[ -n "${EVAL_SEEDS:-}" ]]       && ARGS+=(--set "eval_seeds=[${EVAL_SEEDS}]")
[[ -n "${MODEL_SELECTION:-}" ]]  && ARGS+=(--set "model_selection=${MODEL_SELECTION}")
[[ -n "${REDUCE:-}" ]]           && ARGS+=(--set "reduce=${REDUCE}")
[[ -n "${AGENT_NORMALIZE:-}" ]]  && ARGS+=(--set "agent_normalize=${AGENT_NORMALIZE}")
[[ -n "${EPOCHS:-}" ]]           && ARGS+=(--set "epochs=${EPOCHS}")
[[ -n "${BATCH_SIZE:-}" ]]       && ARGS+=(--set "batch_size=${BATCH_SIZE}")
[[ -n "${PATIENCE:-}" ]]         && ARGS+=(--set "patience=${PATIENCE}")
[[ -n "${TOP_K:-}" ]]            && ARGS+=(--set "top_k=${TOP_K}")
[[ -n "${STAGES:-}" ]]           && ARGS+=(--stages "${STAGES}")
[[ -n "${START_IDX:-}" ]]        && ARGS+=(--set "start_idx=${START_IDX}")
[[ -n "${END_IDX:-}" ]]          && ARGS+=(--set "end_idx=${END_IDX}")
[[ "${OVERWRITE:-0}" == 1 ]]     && ARGS+=(--set "overwrite=true")
[[ "${DRY_RUN:-0}" == 1 ]]       && ARGS+=(--dry-run)
# EXTRA_SET: extra sweep args, word-split (e.g. "--set method_dir_prefix=stepfinder.raw").
read -r -a EXTRA_SET_ARR <<< "${EXTRA_SET:-}"

[[ -n "${GPU:-}" ]] && export CUDA_VISIBLE_DEVICES="$GPU"

echo "=== stepfinder | encoder=${MODEL:-<config>} dataset=${DATASET} subset=${SUBSET:-<all>} config=${CONFIG} ==="
exec python -m stepfinder.sweep "${ARGS[@]}" ${EXTRA_SET_ARR[@]+"${EXTRA_SET_ARR[@]}"}
