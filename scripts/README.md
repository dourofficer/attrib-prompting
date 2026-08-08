# scripts/

Front doors for running one attribution method at a time, plus utilities.

## Per-method scripts

```bash
MODEL=<name> DATASET=<ww|correct-error|traceelephant> [SUBSET=<subset>] bash scripts/<method>.sh
```

where `<method>.sh` is one of `all_at_once.sh`, `step_by_step.sh`, `binary_search.sh`.

Examples:

```bash
# GPT-4o on one subset (needs OPENAI_API_KEY exported):
MODEL=gpt-4o DATASET=ww SUBSET=hand-crafted bash scripts/all_at_once.sh

# ...on every subset of a dataset (omit SUBSET):
MODEL=gpt-4o DATASET=correct-error bash scripts/step_by_step.sh

# Local vLLM model on one GPU:
MODEL=qwen3.5-9b DATASET=traceelephant SUBSET=captain GPU=0 bash scripts/binary_search.sh

# Quick 10-sample trial, preview first:
MODEL=gpt-4o DATASET=ww SUBSET=hand-crafted END_IDX=10 DRY_RUN=1 bash scripts/all_at_once.sh
MODEL=gpt-4o DATASET=ww SUBSET=hand-crafted END_IDX=10 bash scripts/all_at_once.sh
```

### Environment knobs

| var | meaning |
|---|---|
| `MODEL` (required) | model name — must be declared in the dataset's config `model_specs` |
| `DATASET` (required) | `ww`, `correct-error`, or `traceelephant` |
| `SUBSET` | one subset; omit to run all subsets of the dataset |
| `GPU` | sets `CUDA_VISIBLE_DEVICES` (local vLLM models) |
| `START_IDX` / `END_IDX` | slice of the (numerically sorted) trajectories |
| `DRY_RUN=1` | print the predict command(s) without running |
| `OVERWRITE=1` | redo trajectories that already have an output file |
| `EXTRA_SET` | extra sweep overrides, e.g. `EXTRA_SET="--set step_mode=batch"` |
| `CONFIG` | explicit config path, bypassing the resolution below |

### Config resolution

Closed-source and local models live in separate configs. The script picks
`baselines/prompting/configs/<DATASET>-api.yaml` when it declares `MODEL` in
its `model_specs` (API models: `gpt-4o`, `gpt-5`, ...), and
`configs/<DATASET>.yaml` otherwise (local vLLM models: `qwen3.5-9b`,
`deepseek-8b`). To add a new API model, add a spec block to the `-api` config
and put it in `models:` — no code changes.

Runs are **idempotent per trajectory**: outputs land as
`outputs/<dataset>/<subset>/<model>/<method>/<id>.json` the moment each
trajectory completes, and rerunning the same command executes only the missing
ones — so a crashed or interrupted (API) run resumes where it left off.

## Other scripts

- `import_legacy_jsonl.py` — convert a legacy `predictions_method-*.jsonl`
  tree into the per-trajectory output layout (see its docstring).
- `../baselines/prompting/scripts/run_qwen.sh`, `run_deepseek.sh` — per-model
  wrappers that sweep all datasets × methods for one local model on one GPU.
