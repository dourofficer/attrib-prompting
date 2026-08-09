#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/tmp/attrib-prompting-run}"
RESULTS="${RESULTS:-/mnt/DeepResearch/junlinfang/attrib-prompting-results/full}"
CONCURRENCY="${CONCURRENCY:-25}"

cd "$ROOT"
set -a
source /mnt/DeepResearch/junlinfang/.env
set +a
export OPENAI_API_KEY="${LLMHUB_API_KEY:?LLMHUB_API_KEY is required}"

empty_files() {
  python3 - <<'PY'
from pathlib import Path
import json

for path in Path("outputs/ww").rglob("gpt-5/all_at_once/*.json"):
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("predicted_step") is None and doc.get("raw") == "":
        print(path)
PY
}

mapfile -t initial < <(empty_files)
if [[ ${#initial[@]} -ne 25 ]]; then
  echo "Expected 25 empty GPT-5 all_at_once outputs, found ${#initial[@]}" >&2
  exit 1
fi

tar -czf "$RESULTS/gpt5_all_at_once_empty_before_rerun.tar.gz" "${initial[@]}"

for attempt in 1 2 3; do
  mapfile -t bad < <(empty_files)
  if [[ ${#bad[@]} -eq 0 ]]; then
    break
  fi
  echo "repair attempt=$attempt empty=${#bad[@]}"
  for path in "${bad[@]}"; do
    case "$path" in
      outputs/ww/*/gpt-5/all_at_once/*.json) rm -- "$path" ;;
      *) echo "Refusing unexpected path: $path" >&2; exit 1 ;;
    esac
  done

  for subset in algorithm-generated hand-crafted; do
    python3 -m baselines.prompting.predict \
      --backend openai \
      --model gpt-5 \
      --model-name gpt-5 \
      --api-base-url https://llm-hub.apodex.app/v1 \
      --api-key-env OPENAI_API_KEY \
      --api-header X-Llmhub-Channel=1 \
      --api-concurrency "$CONCURRENCY" \
      --api-max-retries 6 \
      --input "data/ww/$subset" \
      --output "outputs/ww/$subset/gpt-5" \
      --method all_at_once \
      --api-param max_completion_tokens=16384
  done
done

mapfile -t remaining < <(empty_files)
echo "remaining_empty=${#remaining[@]}"
[[ ${#remaining[@]} -eq 0 ]]
