#!/usr/bin/env bash
# Run every prompt-based baseline on the open backbones, without ground truth.
#
# This is the grid the SOAP manuscript's main table needs for its Qwen3.5-9B and
# DeepSeek-R1-Distill-Llama-8B blocks: six methods x five reported subsets, with
# the task answer removed from every prompt, so results land in outputs-nogt/.
#
#   bash scripts/run_open_backbones.sh <model> <dataset> <gpu> [method ...]
#
#   bash scripts/run_open_backbones.sh qwen3.5-9b ww 0
#   bash scripts/run_open_backbones.sh deepseek-8b correct-error 1 correct chief
#   GT=with bash scripts/run_open_backbones.sh qwen3.5-9b ww 0 correct chief raffles
#   MODE=truncated bash scripts/run_open_backbones.sh qwen3.5-9b ww 0 errorprobe
#
# GT defaults to `without` — the manuscript's main table. `GT=with` restores the
# answer line and writes to outputs/ instead, for the with-ground-truth table.
#
# One method runs at a time inside a strand, each in its own vLLM process, so a
# strand needs one GPU. File existence is the resume ledger: rerun the same
# command after a crash and only the missing trajectories execute.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export PATH="$REPO/.venv/bin:$PATH"

MODEL="${1:?usage: run_open_backbones.sh <model> <dataset> <gpu> [method ...]}"
DATASET="${2:?missing dataset}"
GPU="${3:?missing gpu}"
shift 3
METHODS=("$@")
GT="${GT:-without}"
if [ ${#METHODS[@]} -eq 0 ]; then
  METHODS=(all_at_once binary_search correct chief raffles step_by_step)
fi

LOGDIR="logs/open"
mkdir -p "$LOGDIR"

for method in "${METHODS[@]}"; do
  suffix=""
  [ -n "${MODE:-}" ] && suffix="-${MODE}"
  [ "$GT" = "with" ] && suffix="${suffix}-gt"
  log="$LOGDIR/${MODEL}-${DATASET}-${method}${suffix}.log"
  echo "=== $(date +%H:%M:%S) $MODEL $DATASET $method gt=$GT (gpu $GPU) -> $log"
  case "$method" in
    # The three Who&When strategies share one front door per method.
    all_at_once|step_by_step|binary_search)
      MODEL="$MODEL" DATASET="$DATASET" GT="$GT" GPU="$GPU" \
        bash "scripts/prompting/${method}.sh" > "$log" 2>&1 ;;
    # CORRECT and CHIEF read their offline artifacts from artifacts/, which are
    # committed and keyed by the generator/encoder — never by this detector.
    correct|chief)
      MODEL="$MODEL" DATASET="$DATASET" GT="$GT" GPU="$GPU" STAGES=predict \
        bash "scripts/${method}/run.sh" > "$log" 2>&1 ;;
    raffles)
      MODEL="$MODEL" DATASET="$DATASET" GT="$GT" GPU="$GPU" \
        bash scripts/raffles/run.sh > "$log" 2>&1 ;;
    # ErrorProbe runs every mode its config lists; MODE narrows it to one.
    errorprobe)
      MODEL="$MODEL" DATASET="$DATASET" GT="$GT" GPU="$GPU" ${MODE:+MODE="$MODE"} \
        bash scripts/errorprobe/run.sh > "$log" 2>&1 ;;
    *) echo "unknown method: $method" >&2; exit 2 ;;
  esac
  grep -E '^  wrote' "$log" || { echo "!! no output written; see $log" >&2; exit 1; }
done
echo "=== $(date +%H:%M:%S) done: $MODEL $DATASET gt=$GT"
