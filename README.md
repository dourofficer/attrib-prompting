# attrib-prompting

Standalone reproduction of **failure-attribution baselines** for LLM
multi-agent systems: given a *failed* trajectory, predict which agent, at which
step, made the decisive mistake.

The baselines come in two families, scored the same way so they can be
compared:

- **Prompting-based** — ask a frontier model to read the log and name the step.
  These live in `baselines/` and write to `outputs/` and `outputs-nogt/`.
  The three original methods come from the Who&When paper (vendored under
  [`vendored/Agents_Failure_Attribution/`](vendored/Agents_Failure_Attribution)):
  **all_at_once** (one-shot agent+step), **step_by_step** (per-step Yes/No,
  earliest "Yes" wins), **binary_search** (recursive halving); then **CORRECT**,
  **CHIEF**, **RAFFLES** and **ErrorProbe**.
- **Representation-based** — read a model's vectors instead of prompting it,
  and train a small model on top. These live in `baselines-rp/` and write to
  `outputs-rb-gt/` and `outputs-rb-nogt/`: **OAT**, which learns the shape of
  success from unlabelled successful runs, and **StepFinder**, which learns
  from labelled failures.

Prompts, regexes and decision rules are byte-identical to the vendored code,
enforced by execution-level parity tests; so are the representation-based
methods' serialization, pooling and model arithmetic.

## Datasets

| dataset | subsets (size) | answer in prompt? | eval seeds |
|---|---|---|---|
| `ww` (Who&When) | algorithm-generated (126), hand-crafted (58) | yes | 1–20 |
| `correct-error` | arc (304), gaia (50), hotpot (578), math500 (157), mmlu_pro (92), musique (312), wikimqa (733) | yes | 1–3 |
| `correct-error-nogt` | the same 2,226 trajectories as upstream ships them | no | 1–3 |
| `traceelephant` | magentic (91), captain (85) | yes | 1–20 |

Corpora live at `data/<dataset>/<subset>/<id>.json`; agent identity is
`history[t]["role"]`, gold labels are `mistake_agent`/`mistake_step`.

The "answer in prompt?" column is whether the record carries a non-empty
`ground_truth`, which the prompt builders interpolate. CORRECT-Error ships that
field empty upstream, so `data/correct-error` restores it by re-joining each
record to its source benchmark (`question_id` = `task<N>_<K>`, where N is a row
index into the source split) — see
[`misc/build_correct_error_gt.py`](misc/build_correct_error_gt.py);
`correct-error-nogt` preserves the upstream copy. Every dataset therefore
supports both GT settings through the `--gt` flag below.

## Install

```bash
pip install -e .            # CPU core: evaluation, tests, dry-runs
pip install -e ".[api]"     # + OpenAI-compatible API inference
pip install -e ".[vllm]"    # + local-checkpoint inference (CUDA-matched vLLM)
pip install -e ".[rag]"     # + CHIEF's offline retrieval stage only (faiss, MiniLM)
pip install -e ".[rb]"      # + the representation-based family (torch, transformers, torchcde)
```

Run everything from the repo root. Local checkpoints are expected under `../hub/`.

## Running

One method at a time (see [`scripts/README.md`](scripts/README.md) for all knobs):

```bash
export OPENAI_API_KEY=sk-...
MODEL=gpt-4o     DATASET=ww SUBSET=hand-crafted bash scripts/prompting/all_at_once.sh
MODEL=qwen3.5-9b DATASET=traceelephant GPU=0    bash scripts/prompting/binary_search.sh   # all subsets
```

The CORRECT baseline (3-stage pipeline: schema generation → similarity →
schema-guided detection; see [`baselines/correct/README.md`](baselines/correct/README.md)):

```bash
DATASET=ww bash scripts/correct/run.sh                        # full pipeline, all config models
DATASET=ww SUBSET=hand-crafted MODEL=gpt-4o bash scripts/correct/run.sh
```

Its `configs/` ships the closed-source configs only (`gpt-4o`, `gpt-5`);
[`baselines/correct/configs/README.md`](baselines/correct/configs/README.md)
documents the keys and templates a local-vLLM config.

The CHIEF baseline (exemplar retrieval → six-call causal-graph detection; see
[`baselines/chief/README.md`](baselines/chief/README.md)):

```bash
DATASET=ww STAGES=ragprep bash scripts/chief/run.sh           # CPU, needs the [rag] extra
DATASET=ww MODEL=gpt-4o bash scripts/chief/run.sh
```

The RAFFLES baseline (iterative Judge-Evaluator loop, built from the paper —
it releases no code; see [`baselines/raffles/README.md`](baselines/raffles/README.md)).
Its paper setting is without-GT, so default results land in `outputs-nogt/`:

```bash
DATASET=ww MODEL=gpt-4o bash scripts/raffles/run.sh
DATASET=ww MODEL=gpt-4o MAX_ITERS=5 bash scripts/raffles/run.sh   # the paper's K=5
```

The ErrorProbe baseline (Analyzer→Verifier diagnosis from the authors'
simplified reproduction, vendored under `vendored/ERRORPROBE/`; three modes —
`truncated` reads the last 15 turns, `backward` walks the full trace from the
symptom, and `paper` rebuilds the paper's own pipeline, which the vendored
code omits: a MAST tagger, a dependency graph walked backward from the
failure, and a Strategist/Investigator/Arbiter team; see
[`baselines/errorprobe/README.md`](baselines/errorprobe/README.md)).
Its vendored prompts carry no task answer, so default results land in
`outputs-nogt/`:

```bash
DATASET=ww MODEL=gpt-4o bash scripts/errorprobe/run.sh
DATASET=ww MODEL=gpt-4o MODE=truncated bash scripts/errorprobe/run.sh   # cheap mode only
DATASET=ww MODEL=gpt-4o MODE=paper bash scripts/errorprobe/run.sh       # the paper pipeline (opt-in)
```

The OAT baseline (hidden states → PCA → a neural controlled differential
equation trained on *successful* trajectories, scoring each step of a failure by
how far it strays; see [`baselines-rp/oat/README.md`](baselines-rp/oat/README.md)).
It prompts nothing and costs no tokens, but it must be trained first — one
command runs all four stages, and the later ones resume:

```bash
DATASET=ww MODEL=qwen3.5-9b GPU=0 bash scripts/oat/run.sh
DATASET=ww MODEL=qwen3.5-9b STAGES=states-train,train bash scripts/oat/run.sh   # train only
```

Training uses the successful MCP-Atlas trajectories vendored with the paper's
code — every corpus in `data/` is failures only. Its default GT setting is
without, so results land in `outputs-rb-nogt/`.

The StepFinder baseline (the second representation-based one: step embeddings →
BiLSTM → agent-aware attention → one score per step, trained on *labelled
failures*; see [`baselines-rp/stepfinder/README.md`](baselines-rp/stepfinder/README.md)).
It also costs no tokens, and it uses a language model only to embed — never to
generate:

```bash
DATASET=ww MODEL=qwen3-embedding-0.6b GPU=0 bash scripts/stepfinder/run.sh
DATASET=ww MODEL=qwen3-embedding-0.6b PROTOCOL=in-corpus GPU=0 bash scripts/stepfinder/run.sh
```

Being supervised, it ships two training protocols. The default reproduces the
paper: it trains on the regenerated failure trajectories vendored with the code,
which share no task with `data/ww`. The second (`PROTOCOL=in-corpus`) trains on
the 30% partition of each corpus that the evaluation protocol reserves and no
other baseline uses, so it covers only the val and test ids of each seed.

Or the full grid per dataset:

```bash
GPU=0 bash baselines/prompting/scripts/run_qwen.sh        # local models, all datasets
python -m baselines.prompting.sweep --config baselines/prompting/configs/ww-api.yaml
```

Configs are split by backend: `configs/<ds>.yaml` for local vLLM models,
`configs/<ds>-api.yaml` for closed-source APIs (`gpt-4o` active, `gpt-5` spec
ready). An API spec's `params` are sent verbatim and recorded in `_run.json`;
any OpenAI-compatible provider works via `base_url` — adding one is a config
entry, not code.

### Outputs & resume

```
outputs/<dataset>/<subset>/<model>/<method>/<id>.json   # prediction + gold + raw + call log
outputs/<dataset>/<subset>/<model>/<method>/_run.json   # run snapshot (params actually sent)
```

One file per trajectory, written atomically as it completes; `calls` logs every
model response (each step_by_step judgment, each binary_search round). File
existence is the resume ledger: **rerun the same command and only missing
trajectories execute** — a crash costs at most the trajectories in flight. On
API backends step_by_step early-stops at the first "Yes" (the vendored control
flow, ~50% fewer calls, provably identical predictions). Inference always covers
all trajectories; no split is applied at inference time.

### GT settings

`--gt with` (default) keeps prompts byte-identical to the vendored code, which
interpolates the task answer. `--gt without` removes the
`The Answer for the problem is: ...` line and nothing else — the removal the
vendored comments themselves sanction. Without-GT results mirror into
**`outputs-nogt/`** with the identical inner layout, so the two settings never
collide; `gt_in_prompt` is recorded in `_run.json` and in every output file.

```bash
GT=without MODEL=gpt-4o DATASET=ww bash scripts/prompting/all_at_once.sh   # or --gt on predict/sweep
python -m baselines.prompting.report --config .../report_ww.yaml --gt without
```

Because the two settings differ only by that line and share the corpus (hence
the same per-seed splits), `outputs/` vs `outputs-nogt/` is an exact paired
comparison. The representation-based family uses the same axis under its own
pair of roots, `outputs-rb-gt/` and `outputs-rb-nogt/`. Note that `outputs/correct-error/` predates the corpus's restored
answers — see the warning in [`scripts/README.md`](scripts/README.md).

## Evaluation

Decoupled from inference; mirrors the attribscope protocol (verified
cell-for-cell over 2,424 table cells):

```bash
python -m baselines.prompting.report --config baselines/prompting/configs/report_ww.yaml [--check-only]
```

Per-seed val/test splits are reproduced from the corpus file list alone (sorted
stems, seeded shuffle, 0.3/0.2/0.5). `step@1` = integer equality; `agent@1` =
normalized match (gold may be a substring of the prediction); missing
predictions count as wrong. Tables land in `outputs/<ds>/reports/`
(`comparison_by_seed.tsv` per model/subset + `summary_mean_over_seeds.tsv`,
with split-independent `*_full` columns over the whole corpus); `--gt without`
reads and writes the `outputs-nogt/` mirror instead.

The representation-based baselines are scored by the same report, over the same
seeded splits, so a hidden-state method and a prompting one land in comparable
tables:

```bash
python -m oat.report        --config baselines-rp/oat/configs/report_ww.yaml
python -m stepfinder.report --config baselines-rp/stepfinder/configs/report_ww.yaml
python -m rb_shared.rb_metrics --pred-root outputs-rb-nogt/ww
```

The last command adds the views the two papers report — precision, recall, F1
and hit rate over OAT's top-k and conformal step *sets*, and StepFinder's
Acc@K, MRR@3 and tolerance accuracy — plus AUROC and AUPRC, all recomputed from
the scores already stored in each prediction file. Columns a method has no
machinery for stay empty rather than borrowing another's.

StepFinder's in-corpus family predicts only each seed's val and test ids, so its
shared-report table is meaningful on its diagonal; `--diagonal` writes that out:

```bash
python -m stepfinder.report --config baselines-rp/stepfinder/configs/report_ww_incorpus.yaml --diagonal
```

Utilities: `python -m baselines.prompting.reparse` re-derives all_at_once
predictions from stored `raw` (no GPU); `misc/import_legacy_jsonl.py`
imports legacy `predictions_method-*.jsonl` trees.

## Layout

```
baselines/shared/              method-agnostic infra: common.py helpers,
                               backends/ (vllm|openai|dummy), runner.py (drivers, writer)
baselines/prompting/           the three methods (verbatim prompts), predict/sweep/report,
                               configs/ (<ds>.yaml vLLM, <ds>-api.yaml APIs), per-model scripts
baselines/correct/             CORRECT baseline: schemagen → similarity → detection
baselines/chief/               CHIEF baseline: ragprep → six-call causal-graph detection
baselines/raffles/             RAFFLES baseline: iterative Judge-Evaluator loop
baselines/errorprobe/          ErrorProbe baseline: Analyzer→Verifier (truncated, backward) and the paper pipeline (paper/)
baselines-rp/                  representation-based family (own source root, own
                               output roots): rb_shared/ (state cache, encoder
                               loading, the papers' metrics), oat/, stepfinder/
data/                          corpora   ·  vendored/  upstream code, verbatim
outputs/, outputs-nogt/        committed results, with-GT and without-GT
outputs-rb-gt/,                the same for the representation-based family,
outputs-rb-nogt/               including its cached vectors (_oat-states/,
                               _sf-feats/) and trained models (_oat-ckpt/,
                               _sf-ckpt/)
artifacts/                     committed inputs a run consumes, not results:
                               CORRECT's schemata and similarities, CHIEF's exemplars
scripts/                       front doors (one subdir per baseline family)
misc/                          one-off corpus/format utilities
tests/                         CPU-only, keyless (pytest)
```

## Tests

```bash
python -m pytest tests/ -q
```

Vendored prompt/control-flow parity (drives the vendored code itself), bit-exact
split fixtures, driver batching parity, resume/atomic writes, OpenAI
payload/retry/concurrency (fake client), metric rules.
