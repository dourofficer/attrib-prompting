# scripts/

Front doors for running the baselines. One subdirectory per baseline family —
`prompting/` and `correct/` today; `chief/` gets its own when it is adapted.
This README stays at `scripts/` and covers all of them.

## prompting/

```bash
MODEL=<name> DATASET=<ww|correct-error|correct-error-nogt|traceelephant> \
  [SUBSET=<subset>] [GT=with|without] bash scripts/prompting/<method>.sh
```

`<method>.sh` is `all_at_once.sh`, `step_by_step.sh`, or `binary_search.sh`.

```bash
# GPT-4o on one subset, with-GT (default; needs OPENAI_API_KEY exported):
MODEL=gpt-4o DATASET=ww SUBSET=hand-crafted bash scripts/prompting/all_at_once.sh

# ...without-GT, every subset of a dataset (omit SUBSET):
MODEL=gpt-4o DATASET=correct-error GT=without bash scripts/prompting/step_by_step.sh

# Local vLLM model on one GPU:
MODEL=qwen3.5-9b DATASET=traceelephant SUBSET=captain GPU=0 bash scripts/prompting/binary_search.sh

# Quick 10-sample trial, preview first:
MODEL=gpt-4o DATASET=ww SUBSET=hand-crafted END_IDX=10 DRY_RUN=1 bash scripts/prompting/all_at_once.sh
```

### Environment knobs

| var | meaning |
|---|---|
| `MODEL` (required) | model name — must be declared in the dataset's config `model_specs` |
| `DATASET` (required) | `ww`, `correct-error`, `correct-error-nogt`, or `traceelephant` |
| `SUBSET` | one subset; omit to run all subsets of the dataset |
| `GT` | `with` (default) or `without` — drops the answer line from prompts and writes to `outputs-nogt/` |
| `GPU` | sets `CUDA_VISIBLE_DEVICES` (local vLLM models) |
| `START_IDX` / `END_IDX` | slice of the (numerically sorted) trajectories |
| `DRY_RUN=1` | print the predict command(s) without running |
| `OVERWRITE=1` | redo trajectories that already have an output file |
| `EXTRA_SET` | extra sweep overrides, e.g. `EXTRA_SET="--set step_mode=batch"` |
| `CONFIG` | explicit config path, bypassing the resolution below |

### Config resolution

Closed-source and local models live in separate configs. The script picks
`baselines/prompting/configs/<DATASET>-api.yaml` when it declares `MODEL` in its
`model_specs` (API models: `gpt-4o`, `gpt-5`, ...), and `configs/<DATASET>.yaml`
otherwise (local vLLM: `qwen3.5-9b`, `deepseek-8b`). To add an API model, add a
spec block to the `-api` config and put it in `models:` — no code changes.

## correct/

One front door for the whole CORRECT pipeline (schemagen → similarity →
schema-guided detection; see
[`baselines/correct/README.md`](../baselines/correct/README.md)):

```bash
DATASET=<ww|correct-error|traceelephant> [MODEL=<name>] [SUBSET=<subset>] bash scripts/correct/run.sh
```

Examples:

```bash
DATASET=ww GPU=0 bash scripts/correct/run.sh                       # everything, local models
DATASET=ww SUBSET=hand-crafted MODEL=gpt-4o bash scripts/correct/run.sh
DATASET=correct-error STAGES=schemagen MODEL=gpt-5 bash scripts/correct/run.sh   # schemata only
DATASET=ww MODEL=qwen3.5-9b END_IDX=10 DRY_RUN=1 bash scripts/correct/run.sh    # preview
```

Same env knobs as the prompting scripts (`GT`, `GPU`, `START_IDX`/`END_IDX`,
`DRY_RUN`, `OVERWRITE`, `EXTRA_SET`, `CONFIG`; same config resolution against
`baselines/correct/configs/`), plus:

| var | meaning |
|---|---|
| `MODEL` | optional here — omit to run every model in the config |
| `METHOD` | `correct` (schema-guided) or `correct_baseline` (k=0) — omit for both |
| `STAGES` | comma-list of `schemagen,similarity,predict` (default: all) |

Two CORRECT-specific notes: the default GT setting is **`without`** (the
vendored cloud path never puts the answer in the detection prompt — the paper
setting), so detection results land in `outputs-nogt/` unless `GT=with`; and
stage-1/2 artifacts (`<subset>/<schema_model>/schemagen/`, `_similarities/`)
are GT-independent and always live under `outputs/`.

## I/O — what each operation reads and writes

Paths are repo-root relative. `<gt-root>` is `outputs` when `GT=with` and
`outputs-nogt` when `GT=without`; the inner layout is identical in both.

| operation | reads | writes |
|---|---|---|
| `scripts/prompting/<method>.sh` | `baselines/prompting/configs/<DATASET>[-api].yaml` | nothing itself — execs the sweep |
| `… → baselines.prompting.predict` | `data/<DATASET>/<SUBSET>/<id>.json`; existing `<gt-root>/…/<id>.json` (resume ledger); `$OPENAI_API_KEY` for API models; `../hub/<checkpoint>` for vLLM | `<gt-root>/<DATASET>/<SUBSET>/<MODEL>/<METHOD>/<id>.json` (one per trajectory, atomic) and `…/<METHOD>/_run.json` (run snapshot) |
| `baselines.prompting.report --config configs/report_<ds>.yaml [--gt without]` | `data/<ds>/<subset>/*.json` (split universe only); `<gt-root>/<ds>/<subset>/<model>/<method>/[0-9]*.json` | `<gt-root>/<ds>/reports/completion_status.tsv`, `…/reports/<model>/<subset>/comparison_by_seed.tsv`, `…/reports/summary_mean_over_seeds.tsv` |

## TODO — the full sweep, both GT settings

Produce {all_at_once, step_by_step, binary_search} × {ww, traceelephant,
correct-error} × {with-GT, without-GT}. Target models: **GPT-4o and GPT-5
first**; extend to others afterwards.

- [ ] **1. Setup.** `pip install -e ".[api]"`, `export OPENAI_API_KEY=sk-...`.

- [ ] **2. Smoke test** (10 trajectories, preview then run, both settings):

  ```bash
  MODEL=gpt-4o DATASET=ww SUBSET=hand-crafted END_IDX=10 DRY_RUN=1 bash scripts/prompting/all_at_once.sh
  MODEL=gpt-4o DATASET=ww SUBSET=hand-crafted END_IDX=10 bash scripts/prompting/all_at_once.sh
  MODEL=gpt-4o DATASET=ww SUBSET=hand-crafted END_IDX=10 GT=without bash scripts/prompting/all_at_once.sh
  ```

  Check `outputs/ww/hand-crafted/gpt-4o/all_at_once/1.json` and its
  `outputs-nogt/…` twin: `gt_in_prompt` must be `true` / `false` respectively,
  with sane `raw` and parsed `predicted_*`.

- [ ] **3. GPT-4o, everything.** Runs resume, so just rerun after any crash:

  ```bash
  for gt in with without; do
    for ds in ww traceelephant correct-error; do
      for m in all_at_once step_by_step binary_search; do
        MODEL=gpt-4o DATASET=$ds GT=$gt bash scripts/prompting/${m}.sh
      done
    done
  done
  ```

  Cost note: `step_by_step` is the expensive method (one call per step, though
  it early-stops at the first "Yes") and `correct-error` the big dataset (2,226
  trajectories). Run `ww` and `traceelephant` first.

- [ ] **4. GPT-5, everything.** Same loop with `MODEL=gpt-5` — its spec is
  already in the `-api` configs, so no config edit is needed.

- [ ] **5. Check completion & evaluate**, once per dataset per setting. Add the
  finished models to `models:` in `baselines/prompting/configs/report_<ds>.yaml`
  (shared by both settings), then:

  ```bash
  for gt in "" "--gt without"; do
    for ds in ww traceelephant correct-error; do
      python -m baselines.prompting.report --config baselines/prompting/configs/report_${ds}.yaml $gt --check-only
      python -m baselines.prompting.report --config baselines/prompting/configs/report_${ds}.yaml $gt
    done
  done
  ```

  Every row must be `DONE`. For unparsed rows (`fmt_fail`), try
  `python -m baselines.prompting.reparse` before re-running anything. Tables
  land in `outputs/<ds>/reports/` and `outputs-nogt/<ds>/reports/`.

- [ ] **6. Extend to other models** (optional). Add a spec block to
  `configs/<ds>-api.yaml` (any OpenAI-compatible provider via `base_url`) and
  repeat steps 3–5. Reasoning models: use `max_completion_tokens`, omit
  `temperature`/`top_p` (see the `gpt-5` spec).

- [ ] **7. Commit the results** — outputs are part of the repo:

  ```bash
  git add outputs/ outputs-nogt/ && git commit -m "GPT-4o/GPT-5 prompting results, both GT settings"
  ```

## Other scripts

- `prompting/run_apodex_full.sh`, `prompting/rerun_empty_gpt5.sh` — the
  collaborator's full-sweep driver and its targeted repair pass. Both are
  environment-specific (hardcoded `REPO_ROOT`, `.env` and archive paths); read
  them before reuse.
- `../misc/build_correct_error_gt.py` — one-off: restore the task answer
  CORRECT-Error ships empty by re-joining each record to its source benchmark on
  the row index in `question_id`. Needs `pip install -e ".[data]"`.
- `../misc/import_legacy_jsonl.py` — convert a legacy
  `predictions_method-*.jsonl` tree into the per-trajectory output layout.
- `../baselines/prompting/scripts/run_qwen.sh`, `run_deepseek.sh` — per-model
  wrappers that sweep all datasets × methods for one local model on one GPU.
