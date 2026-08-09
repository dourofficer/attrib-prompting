#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/tmp/attrib-prompting-run}"
ENV_FILE="${ENV_FILE:-/mnt/DeepResearch/junlinfang/.env}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-/mnt/DeepResearch/junlinfang/attrib-prompting-results/full}"
CONCURRENCY="${CONCURRENCY:-64}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-10}"

mkdir -p "$ARCHIVE_ROOT"
cd "$REPO_ROOT"
set -a
. "$ENV_FILE"
set +a
export OPENAI_API_KEY="$LLMHUB_API_KEY"

LEDGER="$ARCHIVE_ROOT/run_ledger.tsv"
if [[ ! -f "$LEDGER" ]]; then
  printf 'model\tdataset\tsubset\tmethod\tstarted_utc\tfinished_utc\tstatus\trows\texpected\n' > "$LEDGER"
fi

expected_rows() {
  python3 - "$1" <<'PY'
import sys
from baselines.prompting.predict import load_records
print(len(load_records(sys.argv[1])))
PY
}

completed_rows() {
  local dir="$1"
  if [[ ! -d "$dir" ]]; then
    echo 0
    return
  fi
  find "$dir" -maxdepth 1 -type f -name '*.json' ! -name '_run.json' 2>/dev/null | wc -l
}

run_one() {
  local model="$1" channel="$2" dataset="$3" subset="$4" method="$5"
  local input="data/$dataset/$subset"
  local output="outputs/$dataset/$subset/$model"
  local method_dir="$output/$method"
  local expected rows attempt started finished status
  expected="$(expected_rows "$input")"
  started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "=== START model=$model dataset=$dataset subset=$subset method=$method expected=$expected concurrency=$CONCURRENCY at=$started ==="

  for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    rows="$(completed_rows "$method_dir")"
    if [[ "$rows" -eq "$expected" ]]; then
      break
    fi
    echo "--- attempt=$attempt complete=$rows/$expected ---"
    args=(
      --backend openai
      --model "$model"
      --model-name "$model"
      --api-base-url "$LLMHUB_BASE_URL"
      --api-key-env OPENAI_API_KEY
      --api-header "X-Llmhub-Channel=$channel"
      --api-concurrency "$CONCURRENCY"
      --api-max-retries 6
      --input "$input"
      --output "$output"
      --method "$method"
    )
    if [[ "$model" == "gpt-4o" ]]; then
      args+=(--api-param max_tokens=1024 --api-param temperature=0.6 --api-param top_p=0.95 --api-param seed=0)
    else
      args+=(--api-param max_completion_tokens=16384)
    fi
    PYTHONUNBUFFERED=1 python3 -m baselines.prompting.predict "${args[@]}"
  done

  rows="$(completed_rows "$method_dir")"
  finished="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  status=DONE
  [[ "$rows" -eq "$expected" ]] || status="INCOMPLETE"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$model" "$dataset" "$subset" "$method" "$started" "$finished" "$status" "$rows" "$expected" >> "$LEDGER"

  if [[ -d "$method_dir" ]]; then
    tar -C "$REPO_ROOT" -czf \
      "$ARCHIVE_ROOT/${model}_${dataset}_${subset}_${method}.tar.gz" \
      "outputs/$dataset/$subset/$model/$method"
  fi
  echo "=== END model=$model dataset=$dataset subset=$subset method=$method status=$status rows=$rows/$expected at=$finished ==="
  [[ "$status" == DONE ]]
}

run_model() {
  local model="$1" channel="$2"
  local methods=(all_at_once binary_search step_by_step)
  local specs=(
    'ww algorithm-generated'
    'ww hand-crafted'
    'traceelephant magentic'
    'traceelephant captain'
    'correct-error arc'
    'correct-error gaia'
    'correct-error hotpot'
    'correct-error math500'
    'correct-error mmlu_pro'
    'correct-error musique'
    'correct-error wikimqa'
  )
  local method spec dataset subset
  for method in "${methods[@]}"; do
    for spec in "${specs[@]}"; do
      read -r dataset subset <<< "$spec"
      run_one "$model" "$channel" "$dataset" "$subset" "$method"
    done
  done
}

run_model gpt-4o 4
run_model gpt-5 1

tar -C "$REPO_ROOT" -czf "$ARCHIVE_ROOT/all_outputs_final.tar.gz" outputs
date -u +%Y-%m-%dT%H:%M:%SZ > "$ARCHIVE_ROOT/COMPLETE"
echo "ALL EXPERIMENTS COMPLETE"
