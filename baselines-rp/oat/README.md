# OAT baseline

Reproduction of **OAT** (One-class Agent Tracing; Yeh, Zhu, Deep & Li,
UW–Madison and Microsoft Research — "Tracing Agentic Failure from the Flow of
Success", arXiv:2607.12747, paper at [`../../vendored/OAT/oat.pdf`](../../vendored/OAT/oat.pdf)).
The authors release both code and data under
[`vendored/OAT/`](../../vendored/OAT), and that code — not the paper's prose —
is the faithfulness reference here. Every place this adaptation had to choose
is recorded in [`IMPLEMENTATION.md`](IMPLEMENTATION.md).

OAT is one of this repo's two **representation-based** baselines — the other is
[StepFinder](../stepfinder/README.md). It does not prompt a model. It reads a
model's internal vectors and does arithmetic on them. Where StepFinder learns
from labelled failures, OAT never sees one.

## The idea in one sentence

A multi-agent system that succeeds moves through the model's internal state in
a smooth, characteristic way; OAT learns that motion from successful runs
alone, then flags the steps of a failed run that depart from it.

## Why it is unlike every other baseline here

The prompting baselines all ask a frontier model to read a log and name the
step that went wrong. OAT asks nothing. Three consequences follow, and they are
the reason the paper exists:

- **It never sees a labelled failure.** Training uses successful trajectories
  only — about a hundred of them — so the expensive part of building a failure
  attribution system, annotating which step of a broken run was to blame,
  disappears.
- **It costs nothing at inference.** No tokens, no API. The paper measures 16 ms
  per trajectory against 4.5 seconds for GPT-4o, and the trained model is small
  enough to run in under a gigabyte of memory.
- **It is trained, not prompted.** That also means it must be trained *here*,
  on this machine, before it can predict anything — which is what makes its
  pipeline four stages instead of one.

## How it works

### 1. Turn each step into a vector

A trajectory becomes one long document: the question, then every turn of the
log, each turn headed by its step number, role and agent. That document goes
through a frozen open-weights model once — a plain forward pass, no generation.

As the model reads, it maintains a *hidden state* for every token: a vector
that is the model's running internal summary of what it has read. OAT takes the
hidden states of the tokens belonging to a step and averages them. That average
is the step's vector.

Averaging matters. The paper ablates it against using only the step's final
token and loses about 20 points of F1 (Fig. 9): what went wrong in a step is
usually visible somewhere in the middle of it — in a fabricated fact, a
malformed argument — not in how it happened to end.

The vectors are as wide as the model is deep — 4,096 numbers for the 8-to-9B
extractors shipped here, 5,120 for the paper's 27B. PCA compresses them to 64,
fitted on the successful trajectories so the surviving directions are the ones
along which *success* varies. That also means the extractor's width stops
mattering after this step: every later stage works in 64 dimensions.

### 2. Learn the flow of success

Now a trajectory is a sequence of 64-dimensional points. OAT treats that
sequence not as a list but as samples from a continuous curve, and learns the
curve's motion with a **neural controlled differential equation** — a small
network that answers "given where the trajectory is now, and which way it is
bending, where does it go next?"

The word *controlled* carries the weight. An ordinary neural differential
equation is told only where the curve starts, and from there its path is fixed;
that suits a system with one correct way to behave. Agent systems have many —
several different plans can all solve the task — so the model is instead
continuously steered by the observed trajectory itself, which a cubic spline
turns from discrete points into a smooth path it can differentiate. The paper
measures the difference and the controlled version wins on every metric
(Fig. 3).

One refinement guards against the situation this repo is always in. The steering
signal from an unfamiliar trajectory can be wildly larger than anything seen in
training, and a big enough signal drags the model somewhere meaningless. So the
signal passes through a **gate**: a small network that recognizes unfamiliar
steering and turns it down. On out-of-domain data the gate is worth +0.17 AUROC
(Fig. 4), at a cost of 0.028 in-domain — exactly the trade you want when
training and test data come from different worlds.

Training is one-class: predict the next step's vector, minimize the squared
error, and only ever look at successful trajectories.

### 3. Score a failed run

Run the trained model along a failed trajectory. At each step it predicts where
that step should have landed; the **anomaly score** is the squared distance
between the prediction and where the step actually landed. A step that
continues a normal-looking task scores low. A step that hallucinates a fact, or
calls a tool with the wrong arguments, lands somewhere the flow of success does
not go, and scores high.

### 4. Turn scores into an answer

Three decoders, all computed and stored for every trajectory:

| decoder | picks | used for |
|---|---|---|
| argmax | the single highest-scoring step | this repo's `step@1` and `agent@1` |
| top-k | the `k` highest-scoring steps (`k = 3`) | the paper's set metrics |
| conformal | every step above a calibrated threshold | the paper's set metrics |

The conformal threshold is the interesting one. Rather than guessing a cutoff,
OAT scores a held-out set of *successful* trajectories, and takes the value that
only the top `α = 20%` of those normal steps exceed. Anything above it in a
failed run is unusual by a measured standard rather than an assumed one. If
nothing clears the bar, the top-scoring step stands in, so every trajectory
always gets an answer.

## The pipeline

Four stages. The first two are shared by every subset and every GT setting, so
they run once per extractor; the last two run per subset.

| stage | does | writes |
|---|---|---|
| `states-train` | one forward pass per successful training trajectory | `_oat-states/*.pt` under the training root |
| `train` | PCA, then one model per seed plus its conformal threshold | `_oat-ckpt/` under the training root |
| `states-test` | one forward pass per trajectory under test | `_oat-states/*.pt` beside the predictions |
| `score` | project, align, score, decode | `oat.s<seed>/<id>.json` |

Between projection and scoring sits **CORAL**, which the paper applies whenever
training and test data come from different sources — as they do here, always.
It recentres and rescales the test vectors so their spread matches the training
set's, using no labels at all.

CORAL is fitted over every cached trajectory of the subset, not over the
trajectories a particular invocation was asked to score, so `--start_idx` and
`--end_idx` do not change the alignment. What *would* change it is scoring
before extraction has finished, since the alignment is only as complete as the
cache. A run says so when that happens, and records `coral_n_trajectories` in
every file it writes. To split extraction across machines, extract in slices
and score once at the end:

```bash
END_IDX=1000   STAGES=states-test bash scripts/oat/run.sh
START_IDX=1000 STAGES=states-test bash scripts/oat/run.sh
               STAGES=score       bash scripts/oat/run.sh
```

## Where the training data comes from

OAT trains on successes, and every corpus in `data/` is failures only. The
successes are the 103 MCP-Atlas tool-use trajectories that ship inside
`vendored/OAT/dataset/`, read in place. That is the paper's own
out-of-distribution protocol: Table 2 trains on exactly these and tests on
Who&When without ever seeing a Who&When success.

Training is therefore identical for every corpus, and the checkpoint is shared:
one `(extractor, seed)` model scores Who&When, CORRECT-Error and TraceElephant
alike.

## Extractors

Any local checkpoint works — the method needs hidden states, which an API does
not expose. The shipped configs use `qwen3.5-9b` and `deepseek-8b`; add
`qwen3.5-27b`, the paper's own, as a config entry (see
[`configs/README.md`](configs/README.md)). Each extractor gets its own PCA and
its own trained models, and is its own `<model>` directory in the output tree.

The paper's proxy-LLM ablation (Fig. 10) is the reason substitution is
reasonable: swapping the extractor away from the model that generated the
trajectories costs only a little accuracy.

## Seeds

The vendored code trains five models, seeds 42 to 46, and reports the mean and
spread. This repo keeps them separate: each seed is its own method directory,
`oat.s42` through `oat.s46`, and all five appear in the report so the variation
across training runs is visible rather than averaged away.

## GT settings

The vendored document never contains the task's answer, so `--gt without` is
both the faithful setting and the default; results land in **`outputs-rb-nogt/`**.
`--gt with` appends the repo's standard `The Answer for the problem is: ...`
line to the question before extraction and writes to **`outputs-rb-gt/`**.

Only the test-side vectors are extracted twice. The trained model is shared:
the training corpus has no answer field, so a with-GT run trains on exactly the
same tensors.

## What the paper reports

Who&When, trained on MCP-Atlas successes, Qwen3.5-27B representations
(Table 2 — the row to compare against, since Who&When is this repo's corpus
too):

| approach | precision | recall | F1 | hit | AUROC |
|---|---|---|---|---|---|
| GPT-4o (prompted) | 0.129 | 0.198 | 0.151 | 0.198 | 0.566 |
| GPT-5 (prompted) | 0.111 | 0.275 | 0.152 | 0.275 | 0.584 |
| OAT (top-k) | 0.150 | 0.451 | **0.225** | **0.451** | **0.758** |
| OAT (conformal) | **0.184** | 0.330 | 0.211 | 0.330 | 0.758 |

Read those numbers with two caveats. The paper's benchmark annotates *every*
step that contributed to a failure; this repo's corpora annotate one decisive
step, which caps the precision of a three-step prediction at 1/3 and makes
recall binary. And a hit rate at `k = 3` is not `step@1` — it is a strictly
easier question.

## How this repo scores it

Two views of the same prediction files, because the two questions are both
worth answering:

- **`python -m oat.report`** — the repo's shared protocol: `step@1` and
  `agent@1` over the same seeded val/test splits every other baseline is scored
  on. This is what makes OAT comparable to the prompting baselines. `step@1` is
  the argmax step; `agent@1` is the role of that step, since OAT predicts no
  agent of its own.
- **`python -m rb_shared.rb_metrics`** — the paper's view: precision, recall,
  F1 and hit rate over the top-k and conformal sets, plus AUROC and AUPRC over
  the raw scores. Nothing is re-run; it reads the scores already stored in each
  output file.

## Layout

- `serialize.py` — the document format: per-step headers, char spans, the
  question prefix, and the answer line under `--gt with`.
- `extract.py` — tokenizer offsets to step token spans, mean pooling, and
  loading a checkpoint as a bare decoder.
- `model.py` — the neural CDE, transcribed from `vendored/OAT/models/oat.py`.
- `train.py` — PCA, normalization, CORAL, the training loop, and the three
  decoders.
- `predict.py` — the four stages, resumable at each.
- `sweep.py` — the grid; shells out one child per extractor × subset.
- `report.py` — thin alias of `baselines.prompting.report`.
- `configs/` — `<ds>.yaml` (what to run) and `report_<ds>.yaml` (what to
  score); see [`configs/README.md`](configs/README.md).

## Running

From the repo root. The front door is `scripts/oat/run.sh`:

```bash
DATASET=ww MODEL=qwen3.5-9b GPU=0 bash scripts/oat/run.sh              # both subsets
DATASET=ww SUBSET=hand-crafted MODEL=qwen3.5-9b GPU=0 bash scripts/oat/run.sh
DATASET=ww MODEL=qwen3.5-9b STAGES=states-train,train bash scripts/oat/run.sh   # train only
DATASET=ww MODEL=qwen3.5-9b GT=with GPU=1 bash scripts/oat/run.sh
DATASET=ww DRY_RUN=1 bash scripts/oat/run.sh                          # preview
```

Every stage resumes on file existence — a cached tensor, a saved checkpoint, a
written prediction. Rerun the same command after a crash and only the missing
work executes. Then build the tables (CPU only):

```bash
python -m oat.report --config baselines-rp/oat/configs/report_ww.yaml --check-only
python -m oat.report --config baselines-rp/oat/configs/report_ww.yaml
python -m rb_shared.rb_metrics --pred-root outputs-rb-nogt/ww
```

Installing with `pip install -e ".[rb]"` puts `oat` and `rb_shared` on the path;
without it, prefix commands with `PYTHONPATH=baselines-rp` as `run.sh` does.

## Cost

One forward pass per trajectory, and nothing else. On a shared H200 with a 9B
extractor, a Who&When trajectory takes about a second, and the corpus's worst
case — 121 steps and 81k tokens — takes 15 seconds at 31 GiB of peak memory.
Training is minutes on a GPU and runs once per extractor, however many subsets
follow. Scoring is milliseconds.

The cached vectors are the bulk of the storage: one float16 row per step, which
across all three corpora (38,993 rows) comes to about 305 MB for one layer, one
extractor and one GT setting — so roughly 1.2 GB for the two shipped extractors
in both settings. Sweeping `correct-error` (2,226 trajectories) costs a few
GPU-hours of extraction once and is essentially free thereafter, which is the
opposite of the prompting baselines' cost profile: they pay again on every
rerun.
