#!/usr/bin/env bash
# One-trajectory-at-a-time latency of every prompt-based baseline on Who&When with the
# Qwen3.5-9B judge, without GT, on the in-budget samples in reports/latency/ids_*.json.
# Writes reports/latency/<method>_<subset>.tsv; predictions go to a scratch root.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=.venv/bin/python
COMMON=(--backend vllm --model ../hub/Qwen/Qwen3.5-9B --dtype bfloat16 --seed 0
        --enable_thinking False --tensor_parallel_size 1 --model-name qwen3.5-9b --gt without --overwrite)
for s in algorithm-generated hand-crafted; do
  IDS=reports/latency/ids_$s.json
  ROOT=outputs-cost-nogt/ww/latency/$s/qwen3.5-9b
  IN=data/ww/$s
  run() { # label module argv...
    local label=$1 module=$2; shift 2
    echo "=== $label $s $(date)"
    $PY scripts/cost_latency.py --module $module --ids $IDS --out reports/latency/${label}_$s.tsv --label $label -- "$@" 2>&1 | grep -v "^INFO\|^WARNING\|Processed prompts\|it/s\]" | tail -60
  }
  P=(--temperature 0.6 --top_p 0.95 --gen_max_tokens 1024 --gpu_memory_utilization 0.9 --max_model_len 131072 --input $IN --output $ROOT/prompting)
  run all_at_once  baselines.prompting.predict "${COMMON[@]}" "${P[@]}" --method all_at_once
  run binary_search baselines.prompting.predict "${COMMON[@]}" "${P[@]}" --method binary_search
  run step_by_step_early baselines.prompting.predict "${COMMON[@]}" "${P[@]}" --method step_by_step --step-mode early_stop
  run step_by_step_batch baselines.prompting.predict "${COMMON[@]}" --temperature 0.6 --top_p 0.95 --gen_max_tokens 1024 --gpu_memory_utilization 0.9 --max_model_len 131072 --input $IN --output $ROOT/prompting-batch --method step_by_step --step-mode batch
  run correct baselines.correct.predict "${COMMON[@]}" --temperature 0.0 --top_p 1.0 --gen_max_tokens 1024 --gpu_memory_utilization 0.9 --max_model_len 131072 --input $IN --output $ROOT/correct --method correct --schemata-dir artifacts/ww/$s/schemagen/gpt-4o --similarities artifacts/ww/$s/similarities/bge-m3.json --num-schemata 10 --schema-model gpt-4o
  run chief baselines.chief.predict "${COMMON[@]}" --temperature 0.0 --top_p 1.0 --gen_max_tokens 2048 --gpu_memory_utilization 0.9 --max_model_len 131072 --input $IN --output $ROOT/chief --rag-texts artifacts/ww/$s/rag/all-MiniLM-L6-v2.json
  run errorprobe_paper baselines.errorprobe.predict "${COMMON[@]}" --temperature 1.0 --top_p 0.95 --gen_max_tokens 512 --gpu_memory_utilization 0.6 --max_model_len 16384 --truncate_prompt_tokens 15872 --input $IN --output $ROOT/errorprobe --mode paper --max-hypotheses 3 --chunk-chars 12000 --condensed-chars 20000 --include-mast-examples False --sequential-edges fallback
done
echo "=== ALL DONE $(date)"
