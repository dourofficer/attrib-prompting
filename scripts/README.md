# scripts/

Front doors for running the baselines. One subdirectory per baseline family —
`prompting/`, `correct/`, `chief/`, `raffles/`, `errorprobe/`, `oat/` and
`stepfinder/`. This
README stays at `scripts/` and covers all of them.

The first five are *prompting-based*: they ask a model to read a log and name
the step that went wrong, they live in `baselines/`, and they write to
`outputs/` or `outputs-nogt/`. `oat/` and `stepfinder/` are the
*representation-based*
baseline: it reads a model's hidden states instead of prompting it, it lives in
`baselines-rp/`, and it writes to `outputs-rb-gt/` or `outputs-rb-nogt/`.

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
DATASET=ww bash scripts/correct/run.sh                             # everything in the config
DATASET=ww SUBSET=hand-crafted MODEL=gpt-4o bash scripts/correct/run.sh
DATASET=correct-error STAGES=schemagen MODEL=gpt-5 bash scripts/correct/run.sh   # schemata only
DATASET=ww MODEL=gpt-4o END_IDX=10 DRY_RUN=1 bash scripts/correct/run.sh        # preview
```

Same env knobs as the prompting scripts (`GT`, `GPU`, `START_IDX`/`END_IDX`,
`DRY_RUN`, `OVERWRITE`, `EXTRA_SET`, `CONFIG`), plus:

| var | meaning |
|---|---|
| `MODEL` | optional here — omit to run every model in the config |
| `METHOD` | `correct` (schema-guided) or `correct_baseline` (k=0) — omit for both |
| `STAGES` | comma-list of `schemagen,similarity,predict` (default: all) |

Config resolution is the same shape as prompting's, but only the closed-source
configs ship: `baselines/correct/configs/<DATASET>-api.yaml` (`gpt-4o`,
`gpt-5`). Add `<DATASET>.yaml` for local vLLM models and the script picks it up
— [`baselines/correct/configs/README.md`](../baselines/correct/configs/README.md)
has the template.

Two CORRECT-specific notes: the default GT setting is **`without`** (the
vendored cloud path never puts the answer in the detection prompt — the paper
setting), so detection results land in `outputs-nogt/` unless `GT=with`; and
the stage-1/2 artifacts are GT-independent, so they live outside both output
trees, in **`artifacts/`**:

```
artifacts/<ds>/<subset>/schemagen/<schema_model>/<id>.json
artifacts/<ds>/<subset>/similarities/<embed_model>.json
```

## chief/

One front door for the whole CHIEF pipeline (exemplar retrieval → six-call
causal-graph detection; see
[`baselines/chief/README.md`](../baselines/chief/README.md)):

```bash
DATASET=<ww|correct-error|traceelephant> [MODEL=<name>] [SUBSET=<subset>] bash scripts/chief/run.sh
```

Examples:

```bash
DATASET=ww STAGES=ragprep bash scripts/chief/run.sh                # exemplars only (CPU)
DATASET=ww bash scripts/chief/run.sh                               # everything in the config
DATASET=ww SUBSET=hand-crafted MODEL=gpt-4o bash scripts/chief/run.sh
DATASET=ww MODEL=gpt-4o END_IDX=10 DRY_RUN=1 bash scripts/chief/run.sh   # preview
```

Same env knobs as the prompting scripts (`GT`, `GPU`, `START_IDX`/`END_IDX`,
`DRY_RUN`, `OVERWRITE`, `EXTRA_SET`, `CONFIG`), plus:

| var | meaning |
|---|---|
| `MODEL` | optional here — omit to run every model in the config |
| `STAGES` | comma-list of `ragprep,predict` (default: both) |

Config resolution is the same shape as prompting's, but only the closed-source
configs ship: `baselines/chief/configs/<DATASET>-api.yaml` (`gpt-4o`, `gpt-5`).
Add `<DATASET>.yaml` for local vLLM models and the script picks it up —
[`baselines/chief/configs/README.md`](../baselines/chief/configs/README.md) has
the template.

Three CHIEF-specific notes. The default GT setting is **`with`** (every vendored
stage prompt carries the answer), so results land in `outputs/` unless
`GT=without`. Stage 1 is GT-independent and lives in **`artifacts/`**:

```
artifacts/<ds>/<subset>/rag/<embed_model>.json
```

It is committed, so you only rerun it to change the encoder — and only that
stage needs `pip install -e ".[rag]"` (faiss + sentence-transformers).
Finally, detection is **six LLM calls per trajectory** with the causal graph
inlined in the last two; budget roughly 2.5–3× an all-at-once run.

## raffles/

One front door for the RAFFLES Judge-Evaluator loop (see
[`baselines/raffles/README.md`](../baselines/raffles/README.md)):

```bash
DATASET=<ww|correct-error|traceelephant> [MODEL=<name>] [SUBSET=<subset>] bash scripts/raffles/run.sh
```

Examples:

```bash
DATASET=ww bash scripts/raffles/run.sh                              # everything in the config
DATASET=ww SUBSET=hand-crafted MODEL=gpt-4o bash scripts/raffles/run.sh
DATASET=ww MODEL=gpt-4o MAX_ITERS=5 bash scripts/raffles/run.sh     # the paper's K=5
DATASET=ww MODEL=gpt-4o END_IDX=10 DRY_RUN=1 bash scripts/raffles/run.sh  # preview
```

Same env knobs as the prompting scripts (`GT`, `GPU`, `START_IDX`/`END_IDX`,
`DRY_RUN`, `OVERWRITE`, `EXTRA_SET`, `CONFIG`), plus:

| var | meaning |
|---|---|
| `MODEL` | optional here — omit to run every model in the config |
| `MAX_ITERS` | K, the extra Judge-Evaluator iterations after the first pass (default 2 from the config) |
| `THRESHOLD` | early-stop confidence, of 400 (default 350) |

Config resolution is the same shape as prompting's, but only the closed-source
configs ship: `baselines/raffles/configs/<DATASET>-api.yaml` (`gpt-4o`,
`gpt-5`). Add `<DATASET>.yaml` for local vLLM models and the script picks it up
— [`baselines/raffles/configs/README.md`](../baselines/raffles/configs/README.md)
has the template.

Two RAFFLES-specific notes. The default GT setting is **`without`** (the paper
evaluates Who&When without ground truth), so results land in `outputs-nogt/`
unless `GT=with`. And one trajectory costs up to `4 × (K+1)` LLM calls — a
Judge plus three concurrent Evaluators per iteration — though early
termination usually stops sooner. There is no offline stage and nothing in
`artifacts/`. For a side-by-side K=5 run that doesn't overwrite the default,
add `EXTRA_SET="--set max_iters=5 --set method_dir=raffles.k5"`.

## errorprobe/

One front door for the ErrorProbe Analyzer→Verifier diagnosis (see
[`baselines/errorprobe/README.md`](../baselines/errorprobe/README.md)):

```bash
DATASET=<ww|correct-error|traceelephant> [MODEL=<name>] [SUBSET=<subset>] bash scripts/errorprobe/run.sh
```

Examples:

```bash
DATASET=ww bash scripts/errorprobe/run.sh                               # everything in the config
DATASET=ww SUBSET=hand-crafted MODEL=gpt-4o bash scripts/errorprobe/run.sh
DATASET=ww MODEL=gpt-4o MODE=truncated bash scripts/errorprobe/run.sh   # the cheap mode only
DATASET=ww MODEL=gpt-4o MODE=paper bash scripts/errorprobe/run.sh       # the paper pipeline (opt-in)
DATASET=ww MODEL=gpt-4o END_IDX=10 DRY_RUN=1 bash scripts/errorprobe/run.sh  # preview
```

Same env knobs as the prompting scripts (`GT`, `GPU`, `START_IDX`/`END_IDX`,
`DRY_RUN`, `OVERWRITE`, `EXTRA_SET`, `CONFIG`), plus:

| var | meaning |
|---|---|
| `MODEL` | optional here — omit to run every model in the config |
| `MODE` | `truncated` (vendored default: 2 calls per trajectory), `backward` (backward tracing: ~2–3 calls per examined turn) or `paper` (the paper's own pipeline: tagger, dependency graph, diagnosis team; `2C + 2 + H` calls, about one chunk `C` per ten steps and `H ≤ 3` hypotheses) — omit for the config's `modes` list, which never includes `paper` |

Config resolution is the same shape as prompting's, with the local config
checked first: `baselines/errorprobe/configs/<DATASET>.yaml` holds the local
vLLM models (`qwen3.5-9b`, `deepseek-8b`) and `<DATASET>-api.yaml` the
closed-source ones (`gpt-4o`, `gpt-5`). The script takes whichever declares
`MODEL`; with no `MODEL` it runs the local config's `models:` list.
[`baselines/errorprobe/configs/README.md`](../baselines/errorprobe/configs/README.md)
documents the keys, including the handicapped `qwen3.5-9b` spec.

Two ErrorProbe-specific notes. The default GT setting is **`without`** (the
vendored prompts never carry the task answer), so results land in
`outputs-nogt/` unless `GT=with`. And the three modes write to distinct method
directories (`errorprobe/`, `errorprobe_bt/`, `errorprobe_paper/`), so they
never collide; the shipped `correct-error` config enables `truncated` only —
enable `backward` there deliberately, it multiplies the call count by the
trace length. The paper mode is opt-in everywhere (`MODE=paper`); its knobs
live under the configs' `paper:` key and reach the child as
`--max-hypotheses` and friends (`EXTRA_SET="--set paper.max_hypotheses=5"`).
There is no offline stage and nothing in `artifacts/`.
The remaining GPT-4o / GPT-5 sweep is tracked in
[`errorprobe/TODO.md`](errorprobe/TODO.md).

## oat/

One front door for the OAT representation-based baseline — extract hidden
states, train a neural CDE on successful trajectories, score the failures (see
[`baselines-rp/oat/README.md`](../baselines-rp/oat/README.md)):

```bash
DATASET=<ww|correct-error|traceelephant> [MODEL=<extractor>] [SUBSET=<subset>] bash scripts/oat/run.sh
```

Examples:

```bash
DATASET=ww MODEL=qwen3.5-9b GPU=0 bash scripts/oat/run.sh              # both subsets
DATASET=ww SUBSET=hand-crafted MODEL=qwen3.5-9b GPU=0 bash scripts/oat/run.sh
DATASET=ww MODEL=qwen3.5-9b STAGES=states-train,train bash scripts/oat/run.sh  # train only
DATASET=ww DRY_RUN=1 bash scripts/oat/run.sh                          # preview
```

Same env knobs as the other scripts (`GT`, `GPU`, `START_IDX`/`END_IDX`,
`DRY_RUN`, `OVERWRITE`, `EXTRA_SET`, `CONFIG`), plus:

| var | meaning |
|---|---|
| `MODEL` | the *extractor* — a local checkpoint to read hidden states from, not a model to prompt. Omit to run every one in the config |
| `STAGES` | subset of `states-train,train,states-test,score`; each is idempotent and resumable |
| `SEEDS` | training seeds (default `42,43,44,45,46`), one method directory each |
| `LAYER` | which layer to read (default `-1`, the last) |
| `AGGREGATION` | `mean` (default, the paper's) or `last` |
| `TOP_K`, `ALPHA` | detection-set size and conformal miscoverage rate |
| `EPOCHS`, `PATIENCE` | training budget |

Config resolution is simpler than the others': one file per dataset,
`baselines-rp/oat/configs/<DATASET>.yaml`, because there is no API variant —
hidden states are not something a chat endpoint exposes. See
[`baselines-rp/oat/configs/README.md`](../baselines-rp/oat/configs/README.md)
to add an extractor.

Four OAT-specific notes. The default GT setting is **`without`** (the vendored
document never carries the task answer), so results land in `outputs-rb-nogt/`
unless `GT=with`. Training uses the successful MCP-Atlas trajectories shipped
in `vendored/OAT/dataset/`, read in place — every corpus in `data/` is failures
only. Cached states and checkpoints live *inside* the `outputs-rb-*` roots, in
`_oat-states/` and `_oat-ckpt/`, not in `artifacts/`; the leading underscore
keeps them out of the report's way. And the front door needs
`baselines-rp` on `PYTHONPATH`, which the script exports for you.

## stepfinder/

The second representation-based baseline: embed every step, train a temporal
scorer on labelled failures, score the corpus (see
[`baselines-rp/stepfinder/README.md`](../baselines-rp/stepfinder/README.md)).

```
DATASET=<ww|correct-error|traceelephant> [MODEL=<encoder>] [SUBSET=<subset>] bash scripts/stepfinder/run.sh
```

```bash
DATASET=ww MODEL=qwen3-embedding-0.6b GPU=0 bash scripts/stepfinder/run.sh          # both subsets
DATASET=ww SUBSET=hand-crafted MODEL=qwen3-embedding-0.6b GPU=0 bash scripts/stepfinder/run.sh
DATASET=ww MODEL=qwen3-embedding-0.6b STAGES=feats-train,train bash scripts/stepfinder/run.sh
DATASET=ww MODEL=qwen3-embedding-0.6b PROTOCOL=in-corpus GPU=1 bash scripts/stepfinder/run.sh
DATASET=ww DRY_RUN=1 bash scripts/stepfinder/run.sh                                # preview
```

### Environment knobs

| variable | effect |
|---|---|
| `DATASET` | required — picks `baselines-rp/stepfinder/configs/<DATASET>.yaml` |
| `MODEL`, `SUBSET` | narrow the grid to one encoder / one subset |
| `GT` | `with` appends the answer to the first step's content; default `without` |
| `PROTOCOL` | `regen` (default, the paper's) or `in-corpus`, comma-separated |
| `SEEDS` | training seeds → `stepfinder.s<seed>` |
| `EVAL_SEEDS` | split seeds for `in-corpus` → `stepfinder.e<seed>` |
| `MODEL_SELECTION` | `val` (default) or `vendored` (parity check; refuses to score) |
| `AGENT_NORMALIZE` | `standardize` (default) or `raw` |
| `STAGES` | subset of `feats-train,train,feats-test,score` |
| `EPOCHS`, `BATCH_SIZE`, `PATIENCE`, `TOP_K` | training and decoding budget |
| `GPU`, `START_IDX`, `END_IDX`, `OVERWRITE`, `DRY_RUN`, `CONFIG`, `EXTRA_SET` | as elsewhere |

Config resolution matches OAT's: one file per dataset, no API variant, because
embeddings are not something a chat endpoint exposes. See
[`baselines-rp/stepfinder/configs/README.md`](../baselines-rp/stepfinder/configs/README.md)
to add an encoder.

Four StepFinder-specific notes. It is **supervised**, so it ships two training
protocols — `regen` trains on the regenerated failures vendored with the paper's
code, `in-corpus` on the 30% partition of each corpus the evaluation protocol
reserves. The in-corpus family predicts only each seed's val and test ids, so
`report --check-only` reads `PARTIAL` for it by design and its table is
meaningful on the diagonal (`--diagonal`). Checkpoints always live in the
without-GT tree, so one model serves both GT settings. And scoring runs one
trajectory at a time on purpose: the vendored position prior divides by the
batch's padded width, which would make a prediction depend on its batch.

## I/O — what each operation reads and writes

Paths are repo-root relative. `<gt-root>` is `outputs` when `GT=with` and
`outputs-nogt` when `GT=without`; the inner layout is identical in both. The
representation-based family uses its own pair, `<rb-root>` = `outputs-rb-gt` or
`outputs-rb-nogt`, with the same inner layout again.

| operation | reads | writes |
|---|---|---|
| `scripts/prompting/<method>.sh` | `baselines/prompting/configs/<DATASET>[-api].yaml` | nothing itself — execs the sweep |
| `scripts/correct/run.sh` | `baselines/correct/configs/<DATASET>[-api].yaml` | nothing itself — execs the 3-stage sweep |
| `… → baselines.correct.schemagen` | `data/<DATASET>/<SUBSET>/<id>.json` (incl. gold labels) | `artifacts/<DATASET>/<SUBSET>/schemagen/<SCHEMA_MODEL>/<id>.json` (+ `_run.json`) |
| `… → baselines.correct.similarity` | `data/<DATASET>/<SUBSET>/<id>.json`; the BGE-M3 checkpoint | `artifacts/<DATASET>/<SUBSET>/similarities/<EMBED_MODEL>.json` (+ `.meta.json`) |
| `… → baselines.correct.predict` | the corpus + **both** artifacts above; existing outputs (resume ledger) | `<gt-root>/<DATASET>/<SUBSET>/<MODEL>/<METHOD>/<id>.json` (+ `_run.json`) |
| `scripts/chief/run.sh` | `baselines/chief/configs/<DATASET>[-api].yaml` | nothing itself — execs the 2-stage sweep |
| `… → baselines.chief.ragprep` | `data/<DATASET>/<SUBSET>/<id>.json` (questions only); `vendored/CHIEF/rag/{index,kb}` | `artifacts/<DATASET>/<SUBSET>/rag/<EMBED_MODEL>.json` (+ `.meta.json`) |
| `… → baselines.chief.predict` | the corpus + the RAG artifact above; existing outputs (resume ledger) | `<gt-root>/<DATASET>/<SUBSET>/<MODEL>/chief/<id>.json` (+ `_run.json`), each with all six stage responses in `calls` |
| `scripts/raffles/run.sh` | `baselines/raffles/configs/<DATASET>[-api].yaml` | nothing itself — execs the sweep |
| `… → baselines.raffles.predict` | `data/<DATASET>/<SUBSET>/<id>.json`; existing outputs (resume ledger) | `<gt-root>/<DATASET>/<SUBSET>/<MODEL>/raffles/<id>.json` (+ `_run.json`), each with the full Judge/Evaluator transcript in `calls` and a per-iteration audit in `iterations` |
| `scripts/errorprobe/run.sh` | `baselines/errorprobe/configs/<DATASET>[-api].yaml` | nothing itself — execs the sweep |
| `… → baselines.errorprobe.predict` | `data/<DATASET>/<SUBSET>/<id>.json`; existing outputs (resume ledger) | `<gt-root>/<DATASET>/<SUBSET>/<MODEL>/<errorprobe\|errorprobe_bt\|errorprobe_paper>/<id>.json` (+ `_run.json`), each with the full transcript in `calls` (Analyzer/Verifier, the backward walk, or the paper mode's tagger/dependency/Strategist/Investigator/Arbiter rounds) |
| `… → baselines.prompting.predict` | `data/<DATASET>/<SUBSET>/<id>.json`; existing `<gt-root>/…/<id>.json` (resume ledger); `$OPENAI_API_KEY` for API models; `../hub/<checkpoint>` for vLLM | `<gt-root>/<DATASET>/<SUBSET>/<MODEL>/<METHOD>/<id>.json` (one per trajectory, atomic) and `…/<METHOD>/_run.json` (run snapshot) |
| `scripts/oat/run.sh` | `baselines-rp/oat/configs/<DATASET>.yaml` | nothing itself — execs the sweep |
| `… → oat.predict` (`states-train`) | `vendored/OAT/dataset/MCP-atlas/Qwen3.5-27B/*.json`; `../hub/<checkpoint>` | `outputs-rb-nogt/mcp-atlas/train/<MODEL>/_oat-states/<id>.pt` (+ `_manifest.json`) |
| `… → oat.predict` (`train`) | those cached states | `outputs-rb-nogt/mcp-atlas/train/<MODEL>/_oat-ckpt/{projector.pt, source_latents.pt, s<SEED>/model.pt}` |
| `… → oat.predict` (`states-test`) | `data/<DATASET>/<SUBSET>/<id>.json`; `../hub/<checkpoint>` | `<rb-root>/<DATASET>/<SUBSET>/<MODEL>/_oat-states/<id>.pt` |
| `… → oat.predict` (`score`) | the cached test states + the checkpoint; existing outputs (resume ledger) | `<rb-root>/<DATASET>/<SUBSET>/<MODEL>/oat.s<SEED>/<id>.json` (+ `_run.json`), each with the per-step anomaly scores and the top-k/conformal sets |
| `oat.report --config baselines-rp/oat/configs/report_<ds>.yaml` | `data/<ds>/<subset>/*.json`; `<rb-root>/<ds>/<subset>/<model>/oat.s<seed>/[0-9]*.json` | `<rb-root>/<ds>/reports/oat/…` — step@1 and agent@1, same tables as every other baseline |
| `scripts/stepfinder/run.sh` | `baselines-rp/stepfinder/configs/<DATASET>.yaml` | nothing itself — execs the sweep |
| `… → stepfinder.predict` (`feats-train`) | `vendored/StepFinder/data/<TRAIN-SET>/train/*.json`; `../hub/<checkpoint>` | `outputs-rb-nogt/stepfinder-regen/<TRAIN-SET>/<MODEL>/_sf-feats/<id>.pt` (+ `_manifest.json`) |
| `… → stepfinder.predict` (`train`) | those cached features | `outputs-rb-nogt/stepfinder-regen/<TRAIN-SET>/<MODEL>/_sf-ckpt/<PRESET>/<SELECTION>/s<SEED>/model.pt` (in-corpus: `<rb-nogt>/<DATASET>/<SUBSET>/<MODEL>/_sf-ckpt/…/e<SEED>/`) |
| `… → stepfinder.predict` (`feats-test`) | `data/<DATASET>/<SUBSET>/<id>.json`; `../hub/<checkpoint>`; under `GT=with`, the without-GT cache | `<rb-root>/<DATASET>/<SUBSET>/<MODEL>/_sf-feats/<id>.pt` |
| `… → stepfinder.predict` (`score`) | the cached test features + the checkpoint; existing outputs (resume ledger) | `<rb-root>/<DATASET>/<SUBSET>/<MODEL>/stepfinder.{s,e}<SEED>/<id>.json` (+ `_run.json`), each with the per-step distribution and the top-k set |
| `stepfinder.report --config baselines-rp/stepfinder/configs/report_<ds>[_incorpus].yaml` | `data/<ds>/<subset>/*.json`; `<rb-root>/<ds>/<subset>/<model>/stepfinder.*/[0-9]*.json` | `<rb-root>/<ds>/reports/stepfinder[-incorpus]/…` — step@1 and agent@1, plus `diagonal.tsv` with `--diagonal` |
| `rb_shared.rb_metrics --pred-root <rb-root>/<ds>` | the same prediction files | `<rb-root>/<ds>/reports/rb_metrics.tsv` — every method directory found: OAT's set metrics, StepFinder's Acc@K / MRR@3 / tolerance accuracy, and AUROC/AUPRC |
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
