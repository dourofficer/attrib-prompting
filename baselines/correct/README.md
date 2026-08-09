# CORRECT baseline

Reproduction of **CORRECT** (COndensed eRror RECognition via knowledge
Transfer; the paper is vendored at
[`vendored/CORRECT/correct.pdf`](../../vendored/CORRECT/correct.pdf)) — a
training-free, retrieval-based failure-attribution method. Instead of judging a
trajectory in isolation, an LLM detector is shown **error schemata** distilled
offline from *other* annotated failures and retrieved by trajectory similarity.

## The variant this repo implements

The vendored code has two prompt families: a local-vLLM path used for the
paper's CORRECT-Error table, and the **cloud path**
([`src/Lib/cloud_paper.py`](../../vendored/CORRECT/src/Lib/cloud_paper.py)) the
authors keep byte-identical to their paper runs — the only path ever used with
closed-source detectors. Per project decision this repo implements the **cloud
variant for all datasets and backends**: the "THOUGHT TEMPLATE FOR GUIDANCE"
schema injection, the cloud schema-generation prompt (step-numbered history +
a trailing block carrying the source trajectory's gold agent/step — retrieved
schemata deliberately show labeled exemplars), and the cloud unicode scrubbing.

[`IMPLEMENTATION.md`](IMPLEMENTATION.md) records the byte-level details that
bite and every deliberate deviation.

## Pipeline

The idea in one sentence: **a detector judging trajectory *q* is shown, in its
prompt, condensed error patterns distilled offline from the trajectories most
similar to *q***. Nothing is trained; the "knowledge transfer" is entirely
prompt-level, and the transferred unit is the *error schema*.

Three stages, all resumable, orchestrated by `sweep.py` (each stage is also a
directly-invocable module):

| stage | module | one unit of work | writes |
|---|---|---|---|
| 1. schemagen | `schemagen.py` | one LLM call per **annotated** trajectory: given its history, question, answer **and its gold mistake agent/step/reason**, write a reusable schema — error signatures, error context, detection heuristics | `artifacts/<ds>/<subset>/schemagen/<schema_model>/<id>.json` |
| 2. similarity | `similarity.py` | embed every trajectory (BGE-M3, question + role-labelled turns, mean-pooled, L2-normalised) and rank all others by cosine, self excluded | `artifacts/<ds>/<subset>/similarities/bge-m3.json` (+ `.meta.json`) |
| 3. detection | `predict.py` | one LLM call per trajectory: the vendored all-at-once prompt with the **top-k neighbours' schemata** spliced in as "THOUGHT TEMPLATES FOR GUIDANCE", answered as `Agent Name:` / `Step Number:` / `Reason for Mistake:` | `<gt-root>/<ds>/<subset>/<model>/<method>/<id>.json` |

Detecting trajectory `7` therefore means: read `bge-m3.json["7"]` → take the
first k ids → load whichever of those have a schema (fewer than k is fine, and
silent) → inject their schema text → one call → parse.

Stages 1–2 are **offline and corpus-scoped**: they run once per (subset,
`schema_model`) and every detector, GT setting and k reuses them. One
`schema_model` per config distills the schemata and **all detectors share
them** — the paper's design of one strong generator serving many detectors.

That is why their output lives in **`artifacts/`**, a sibling of `outputs/`:
`outputs/` holds predictions, one dir per detector; `artifacts/` holds the
precomputed inputs a run consumes, stage first and then producing model. Note
which model each stage is keyed by — the schemata by the **LLM** that wrote
them, the similarities by the **embedder** alone (they are built from corpus
text only, so they are identical whichever detector or schema model you use):

```
artifacts/<ds>/<subset>/schemagen/gpt-4o/1.json      # stage 1, LLM-dependent
artifacts/<ds>/<subset>/similarities/bge-m3.json     # stage 2, embedder-dependent
```

Two things are worth being explicit about, because they look like leakage and
are not quite:

- Schemata are built *from* gold labels, and the retrieved ones carry their
  source trajectory's gold agent/step in the injected text. That is CORRECT's
  premise (an annotated pool of past failures), not an accident.
- The query's *own* schema is never retrieved — stage 2 drops self-similarity —
  so no trajectory sees its own answer.

Methods: **`correct`** (schema-guided, needs stages 1–2) and
**`correct_baseline`** (the vendored k=0 baseline prompt, no artifacts) — the
paper's baseline rows, isolating the effect of the schemata.

Retrieved-schema count `num_schemata` (paper §A.3): Who&When
algorithm-generated **1**, hand-crafted **10**, CORRECT-Error **5**;
TraceElephant is not in the paper — we default to **10** (its trajectories are
long GAIA-style runs like hand-crafted).

## GT settings

The repo-wide GT axis (GUIDE.md) applies, with one inversion: **the vendored
cloud path never puts the task answer in the detection prompt**, so for this
baseline `--gt without` is the parity-tested paper setting **and the default**
(prompting is the opposite: its vendored prompt carries the answer). `--gt
with` inserts the vendored answer line `The Answer for the problem is: ...`
(bytes from the vendored local path) after the problem line. As everywhere:
with-GT detection lands under `outputs/`, without-GT mirrors into
`outputs-nogt/`.

**The flag changes exactly one line of one prompt** — everything else is
identical, so the two settings are an exact paired comparison and share all
offline artifacts:

| stage | affected by `--gt`? | why |
|---|---|---|
| 1. schemagen | no | the schemagen prompt always interpolates `Ground Truth: {ground_truth}` (vendored, gold-conditioned); what fills that slot is a *corpus* property |
| 2. similarity | no | trajectory text is question + turns; the answer never enters the embedding |
| 3. detection | **yes** | `--gt with` adds `The Answer for the problem is: ...` after the problem line, for the query trajectory only |

So schema building is the same under both settings, and both read the same
files under `artifacts/` — which is why that root has no `-nogt` mirror. Only
detection outputs split across `outputs/` vs `outputs-nogt/`. (The retrieved schemata
still carry their neighbours' gold agent/step in either setting; that is stage
1's design, not the flag.)

To vary stage 1's ground truth you change the corpus, not the flag: on
`data/correct-error` the restored answers fill the `Ground Truth:` slot as the
paper's Fig. 9 intends, while `data/correct-error-nogt` reproduces the
as-released corpus where the slot renders blank.

## Running

```bash
# Full pipeline, one dataset, every model in the config (paper setting):
export OPENAI_API_KEY=sk-...
DATASET=ww bash scripts/correct/run.sh

# One model / subset / stage:
DATASET=ww SUBSET=hand-crafted MODEL=gpt-4o bash scripts/correct/run.sh
DATASET=correct-error STAGES=schemagen MODEL=gpt-5 bash scripts/correct/run.sh
GT=with DATASET=ww MODEL=gpt-4o bash scripts/correct/run.sh                  # with-GT extension

# Or the sweep directly:
python -m baselines.correct.sweep --config baselines/correct/configs/ww-api.yaml [--dry-run]
```

Everything is idempotent per trajectory (file existence = resume ledger). API
detectors can consume locally-generated schemata and vice versa: run the
schemagen stage from one config, name the same `schema_model:` in the other.

Only closed-source inference configs ship (`configs/<ds>-api.yaml`, `gpt-4o`
and `gpt-5`); [`configs/README.md`](configs/README.md) documents every key and
has a drop-in template for local vLLM models.

## Evaluation

Shared report, correct methods, per-seed splits (ww/te seeds 1–20, ce 1–3):

```bash
python -m baselines.correct.report --config baselines/correct/configs/report_ww.yaml [--check-only]
# --gt with evaluates the with-GT tree; default (without) reads outputs-nogt/.
```
