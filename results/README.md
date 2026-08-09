# GPT-4o and GPT-5 prompting results

This directory contains the complete prompting-baseline sweep run on
2026-08-09. It includes per-trajectory raw model responses, parsed predictions,
run parameters, per-seed evaluation tables, and aggregate metrics.

## Completeness

- Models: `gpt-4o`, `gpt-5`
- Methods: `all_at_once`, `step_by_step`, `binary_search`
- Dataset subsets: 11
- Experiment cells: 66 / 66 complete
- Trajectory result files: 15,516 / 15,516
- Result files per model: 7,758
- Empty responses after targeted repair: 0
- Unparseable structured responses: 0
- Invalid JSON files: 0

Eight GPT-5 `all_at_once` responses explicitly predict `Agent Name: None` and
`Step Number: N/A`. These are valid model responses, not missing runs; because
the datasets contain a labeled error, they are scored as incorrect.

## Layout

```text
results/
  gpt-4o/<dataset>/<subset>/<method>/<id>.json
  gpt-5/<dataset>/<subset>/<method>/<id>.json
  gpt-*/<dataset>/<subset>/<method>/_run.json
  gpt-*/<dataset>/<subset>/metrics_by_seed.tsv
  metrics/<dataset>/completion_status.tsv
  metrics/<dataset>/summary_mean_over_seeds.tsv
  run_metadata.json
```

Each trajectory JSON contains its parsed prediction, gold attribution, raw
model response, and the complete call log. `_run.json` records the actual model
parameters and execution mode. `metrics_by_seed.tsv` contains validation/test
results for every evaluation seed. The tables below use the split-independent
full-corpus metrics from `summary_mean_over_seeds.tsv`.

## Full-corpus metrics (%)

`AoA` = all-at-once, `SBS` = step-by-step, and `BS` = binary search.

| Model | Dataset | Subset | N | AoA step | AoA agent | SBS step | SBS agent | BS step | BS agent |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt-4o | correct-error | arc | 304 | 50.33 | 88.49 | 71.05 | 72.70 | 25.33 | 22.70 |
| gpt-4o | correct-error | gaia | 50 | 26.00 | 74.00 | 18.00 | 32.00 | 0.00 | 4.00 |
| gpt-4o | correct-error | hotpot | 578 | 13.84 | 75.09 | 43.94 | 52.60 | 11.07 | 16.96 |
| gpt-4o | correct-error | math500 | 157 | 34.39 | 63.06 | 34.39 | 31.21 | 8.92 | 8.28 |
| gpt-4o | correct-error | mmlu_pro | 92 | 43.48 | 85.87 | 60.87 | 65.22 | 9.78 | 3.26 |
| gpt-4o | correct-error | musique | 312 | 11.22 | 76.28 | 40.06 | 51.60 | 6.73 | 9.62 |
| gpt-4o | correct-error | wikimqa | 733 | 11.87 | 77.90 | 37.11 | 52.39 | 7.09 | 14.19 |
| gpt-4o | traceelephant | captain | 85 | 10.59 | 65.88 | 18.82 | 55.29 | 7.06 | 25.88 |
| gpt-4o | traceelephant | magentic | 91 | 6.59 | 67.03 | 21.98 | 71.43 | 6.59 | 42.86 |
| gpt-4o | ww | algorithm-generated | 126 | 15.87 | 53.97 | 22.22 | 31.75 | 27.78 | 55.56 |
| gpt-4o | ww | hand-crafted | 58 | 1.72 | 63.79 | 15.52 | 65.52 | 3.45 | 6.90 |
| gpt-5 | correct-error | arc | 304 | 73.36 | 89.80 | 87.17 | 88.49 | 89.80 | 89.80 |
| gpt-5 | correct-error | gaia | 50 | 48.00 | 66.00 | 48.00 | 58.00 | 72.00 | 68.00 |
| gpt-5 | correct-error | hotpot | 578 | 62.80 | 78.55 | 61.59 | 72.66 | 62.63 | 73.70 |
| gpt-5 | correct-error | math500 | 157 | 71.34 | 66.24 | 49.04 | 41.40 | 79.62 | 63.69 |
| gpt-5 | correct-error | mmlu_pro | 92 | 68.48 | 85.87 | 78.26 | 76.09 | 83.70 | 85.87 |
| gpt-5 | correct-error | musique | 312 | 56.09 | 80.77 | 58.01 | 77.88 | 63.46 | 79.17 |
| gpt-5 | correct-error | wikimqa | 733 | 60.57 | 82.95 | 57.30 | 71.76 | 63.03 | 75.03 |
| gpt-5 | traceelephant | captain | 85 | 8.24 | 61.18 | 18.82 | 49.41 | 25.88 | 60.00 |
| gpt-5 | traceelephant | magentic | 91 | 9.89 | 68.13 | 10.99 | 73.63 | 24.18 | 68.13 |
| gpt-5 | ww | algorithm-generated | 126 | 16.67 | 69.84 | 26.98 | 52.38 | 44.44 | 63.49 |
| gpt-5 | ww | hand-crafted | 58 | 8.62 | 36.21 | 18.97 | 46.55 | 20.69 | 58.62 |

Who&When (`ww`) and TraceElephant prompts include the reference answer and are
therefore oracle-assisted. CORRECT-Error prompts do not include it. Agent@1 is
the repository's normalized agent-name match; Step@1 is exact integer equality.

## Run configuration and cost

- API concurrency: 64 (targeted repair used 25)
- GPT-4o: `max_tokens=1024`, `temperature=0.6`, `top_p=0.95`, `seed=0`
- GPT-5: `max_completion_tokens=16384`
- Account-delta requests: 54,076
- Prompt tokens: 185,898,119
- Completion tokens: 24,172,852
- Cached tokens: 18,014,592
- Total measured cost: **US$374.872852**

Cost is calculated from stable account snapshots immediately before the first
successful compatibility probe and after the repaired sweep. It includes the
small compatibility probes and the 25-request repair (`US$1.554154`). No API
keys, session tokens, or account usage responses are included here.
