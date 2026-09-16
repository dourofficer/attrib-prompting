# StepFinder baseline

Reproduction of **StepFinder** (Zhu, Wu, Jin, Li & Huang, Peking University —
"StepFinder: A Temporal Semantic Framework for Failure Attribution in
Multi-Agent Systems", KDD '26, arXiv:2606.03467, paper at
[`../../vendored/StepFinder/stepfinder.pdf`](../../vendored/StepFinder/stepfinder.pdf)).
The authors release both code and training data under
[`vendored/StepFinder/`](../../vendored/StepFinder), and that code — not the
paper's prose — is the faithfulness reference here. Every place this adaptation
had to choose is recorded in [`IMPLEMENTATION.md`](IMPLEMENTATION.md).

StepFinder is the second of this repo's two **representation-based** baselines.
Like OAT it reads vectors rather than prompting a model. Unlike OAT, it is
**supervised**: it is shown thousands of failures with the guilty step circled,
and learns to circle it itself.

## The idea in one sentence

Turn a log into a sequence of vectors, one per step, and train a small network
to pick out the point in such a sequence where the run went wrong.

## Why it is unlike OAT

The two representation-based baselines answer opposite questions about the same
data.

- **OAT learns what success looks like** and flags whatever departs from it. It
  never sees a labelled failure, which is its whole point — no annotation
  needed.
- **StepFinder learns what blame looks like.** It needs labelled failures, and
  a lot of them, so the paper's real contribution is as much the training set
  as the network: it manufactures one.

That difference propagates. OAT trains once and scores everything. StepFinder's
model is tied to the kind of agent system it was trained on, which is why this
package ships two training protocols rather than one, and why every prediction
records which corpus taught it.

## How it works

### 1. Turn each step into two vectors

An **embedding model** — a model whose job is to compress a passage into a
single vector standing for its meaning — reads each step twice: once for the
step's text, once for the name of the agent that took it. The text keeps 128
numbers, the agent's name 32.

This is the only place a language model is involved, and it generates nothing.
Zero output tokens, which is where the paper's headline efficiency comes from:
0.61 seconds per trajectory against 26 seconds for step-by-step prompting
(Table 6).

### 2. Read the sequence forwards and backwards

A **BiLSTM** walks the sequence of step vectors in both directions, so each
step ends up carrying both what led to it and what came after.

The backward pass is doing real work here. A step can look perfectly reasonable
on its own and turn out to be the mistake only in hindsight — a plausible-looking
search query that returned nothing useful, a calculation that was fine
arithmetic on the wrong number. Reading the sequence in reverse is how a later
contradiction reaches back to the step that caused it. The paper's ablation
removes this module and loses more than removing any other (Table 5).

### 3. Let every step look at every other step

The BiLSTM passes information along a chain, so step 40 hears about step 3
only through thirty-six intermediaries. **Attention** lets them talk directly:
every step compares itself against every other and pulls in what looks
relevant.

StepFinder adds one nudge. Two steps taken by agents with similar names get a
small bonus to how strongly they attend to each other, so one agent's chain of
reasoning holds together against the interleaved turns of everyone else. A
second, gentler version of the same idea scales every step's features by what
the trajectory's agent line-up looks like overall.

### 4. Score, correct, and pick

A small network gives every step a score. Two corrections follow, and the
paper's ablation says both earn their place:

- **Multi-scale differencing** adds a bonus where the sequence *lurches* —
  where a step's vector sits far from where its neighbours' trend pointed. It
  looks for that at two time scales at once, so both an abrupt swerve and a
  slower drift register.
- **Position bias** subtracts a little more the later a step falls. The
  decisive mistake is usually upstream of the symptom, and without this prior a
  model trained on end-of-run failures learns to blame whatever it saw last.

The highest final score is the answer.

## The training signal

Two losses, added together.

The main one is cross-entropy **across the time axis**: the model is asked
"which one of these steps" rather than "is this step bad", so the steps compete
and exactly one wins. That matches how the corpora are annotated — one decisive
step per failure — and it means the model never has to learn a threshold.

The second is self-supervised: from each step's representation, predict the
*next* step's. Nothing about the labels enters it. It teaches the model what an
ordinary continuation looks like, which sharpens its sense of an extraordinary
one. Removing it costs more accuracy than removing any single architecture
block (Table 5), which is the paper's most interesting negative result.

## Where the training data comes from

Every corpus in `data/` is a test set. StepFinder needs labelled failures to
train on, and the paper manufactures them: take one annotated failure, ask a
model to fail the *same task* a different way — same question, same answer, same
agent line-up, a different wrong path and a different guilty step — and repeat
about eighteen times per task.

That yields 1,564 trajectories for Algorithm-Generated and 2,604 for
Hand-Crafted, and both ship inside
[`vendored/StepFinder/data/`](../../vendored/StepFinder/data). The prompt that
generated them ships too; the code that ran it does not, so the data is the
artefact, not the recipe.

The split is clean where it matters. Checked here by exact question match:
**zero** of the training questions appear in either `data/ww` subset, which are
the paper's own test sets. Elsewhere it is not clean, and this package says so
per file — see [Leakage](#leakage).

### Which corpus a subset trains on

The two training corpora come from two different agent systems: Who&When's
Algorithm-Generated half was produced by **CaptainAgent**, its Hand-Crafted half
by **Magentic-One** (`vendored/Agents_Failure_Attribution/README.md:39`). That
matters more than it sounds, because the agent embedding is a text embedding of
the agent's *name*, and the attention bias is the cosine between two such names.
Train on one vocabulary and score on another and that whole signal is noise. So
each subset is paired with the corpus built from the same system:

| subset | agent system | trains on | preset |
|---|---|---|---|
| `ww/algorithm-generated` | CaptainAgent | Algorithm-Generated | `alg` |
| `traceelephant/captain` | CaptainAgent | Algorithm-Generated | `alg` |
| `ww/hand-crafted` | Magentic-One | Hand-Crafted | `hc` |
| `traceelephant/magentic` | Magentic-One | Hand-Crafted | `hc` |
| `correct-error/*`, `traceelephant/swe` | neither | Hand-Crafted | `hc` |
| `tracertraj/code` | MetaGPT (neither) | both, as two families | `hc` / `alg` |

The vocabularies confirm the pairing. Of `traceelephant/captain`'s 1,746 steps,
82.8% carry an agent name that appears in Algorithm-Generated against 57.3% for
Hand-Crafted — its `*_Expert` names are in the former's 205-name vocabulary and
absent from the latter's five. `traceelephant/magentic` runs the other way:
97.2% against 31.9%. `correct-error` matches neither system — its agents are
`Planner`, `WebSurfer`, `CodeExecutor` — so its assignment is a fallback, not a
match, and its numbers should be read as out-of-domain transfer.

The preset follows the corpus, since the paper tuned Table 1's two columns for
these two systems. `protocol.py:48-54` holds the mapping; `--train-set` and
`--preset` override it per subset from the config.

## Two protocols

Both ship, and both write into the same tree under different method names.

| | family A — `regen` (default) | family B — `in-corpus` |
|---|---|---|
| trains on | the vendored regenerated failures | the corpus under test |
| which part | all of it | the 30% the evaluation protocol reserves |
| seeds | five training seeds | one model per evaluation seed |
| method dirs | `stepfinder.s42` … `stepfinder.s46` | `stepfinder.e1` … `stepfinder.e20` |
| covers | every trajectory | that seed's val + test only (~70%) |
| `--check-only` | `DONE` | `PARTIAL`, by construction |

**Family A is the paper.** It reproduces the published protocol and, because it
predicts everything, it drops straight into the shared report beside every
prompting baseline.

**Family B answers a different question**: what StepFinder does with in-domain
supervision. It spends the 30% training partition that the evaluation protocol
sets aside and no other baseline in this repo touches.

Its training budget needs care, and this is the sharpest lesson the
reproduction turned up. The vendored recipe is 50 epochs at batch 16, which
over the 2,604-trajectory training corpus is about 8,000 optimizer steps. Over
a 91-trajectory in-corpus partition the same recipe is about **300** — far
fewer updates than the learning rate was tuned for, leaving the network barely
trained. `--min-train-steps` (5,000 in the shipped configs) raises the epoch
count until the budget is comparable; family A already clears the floor, so the
vendored protocol is untouched.

Given that, family B leads on **ten of twelve** subsets — though not all
equally, and the table is not a scoreboard:

| subset | family A | family B | B trains on |
|---|---|---|---|
| `correct-error/mmlu_pro` | 0.332 | **0.697** | 27 |
| `correct-error/arc` | 0.530 | **0.815** | 91 |
| `correct-error/hotpot` | 0.245 | **0.503** | 173 |
| `correct-error/gaia` | 0.297 | **0.552** | 15 |
| `ww/algorithm-generated` | 0.218 | 0.291 | 37 |
| `correct-error/wikimqa` | 0.196 | 0.246 | 219 |
| `ww/hand-crafted` | 0.114 | 0.140 | 17 |
| `traceelephant/magentic` | **0.178** | 0.095 | 27 |

Six of the twelve are decisive, all on `correct-error`, where the margin is
several times family B's own spread — fifteen in-domain trajectories beat 2,604
out-of-domain ones on `gaia`. Four are inside the noise, and family B is the
noisier column by construction: family A averages five trained models per split
seed, family B has one. The two losses are the two subsets where family A has
seen half the tasks already.

So the honest claim is that **on `correct-error` a few dozen in-domain
trajectories beat the paper's whole regenerated corpus, and elsewhere the two
are close** — a result about the training-data strategy rather than the
architecture. `IMPLEMENTATION.md` has all twelve rows with spreads, and the two
earlier, wrong versions of this table.

Because family B's models are seed-specific, its shared-report table is only
meaningful on its diagonal — the cell where the method's seed and the split's
seed agree. `stepfinder.report --diagonal` writes that out beside the raw
table.

## Leakage

Family A transfers to `correct-error` and `traceelephant` zero-shot, and that
transfer is not clean everywhere. Those benchmarks reuse GAIA questions that the
training corpora also cover, so for some trajectories the model has
already seen a labelled failure on the same task — a different agent system, a
different path, a different guilty step, but the same question and answer.

| subset | trajectories whose task is in training |
|---|---|
| `ww/hand-crafted`, `ww/algorithm-generated` | 0 of 58, 0 of 126 |
| `traceelephant/captain` | 22 of 85 |
| `traceelephant/magentic` | 51 of 91 |
| `correct-error/gaia` | 30 of 50 |
| every other subset | 0 |

Nothing is dropped and nothing is hidden. Every prediction file carries
`train_task_overlap`, every `_run.json` carries `n_train_task_overlap`, and the
flag is always recomputed against the training set actually used — the mapping
matters, since pairing `ww/algorithm-generated` with the *Hand-Crafted* set
instead of its own would put 69 of its 126 tasks into training.

## Checkpoint selection

The vendored loop evaluates on the test directory every epoch and keeps the
best-scoring epoch (`main.py:231-251`). There is no validation split anywhere in
that repository, so every published number is a maximum over fifty epochs
measured on the data being reported.

This package defaults to `--model-selection val`, which holds out part of the
*training* data — split by question, so no task straddles the boundary — and
never touches the corpus under test. `--model-selection vendored` reproduces the
original rule for a parity check and **refuses to write predictions**, so a
number produced that way cannot end up in `outputs-rb-*` by accident.

`--model-selection test` is the paper's rule kept as a reportable, labeled
variant. Each model early-stops on Who&When test accuracy — the paper's only
benchmark — and every other subset that maps to the same training corpus reuses
that checkpoint unchanged. Read the result accordingly: the WW columns are
selected on the data they report, exactly as the published numbers are, while
the CE and TE columns are selected on data disjoint from what they report.
Predictions land under `stepfinder-tsel.s<seed>/` (never the plain family), the
checkpoint sits with the WW subset that picked it, per GT setting, and a CE/TE
run refuses to train its own selection — it demands the WW run first. Because
the selection watches the scored features, and those change with the GT
setting, test-selected checkpoints are GT-specific, unlike val-selected ones.

## Encoders

Any local checkpoint works; the method needs embeddings, which a chat endpoint
does not expose. The shipped configs run three, and the `<model>` axis is
deliberately the same as OAT's so the two representation-based methods land in
comparable rows:

| name | checkpoint | why |
|---|---|---|
| `qwen3-embedding-0.6b` | `../hub/Qwen/Qwen3-Embedding-0.6B` | the paper's own |
| `qwen3.5-9b` | `../hub/Qwen/Qwen3.5-9B` | shared with OAT |
| `deepseek-8b` | `../hub/deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | shared with OAT |

One caveat travels with every decoder column. The vendored code takes the
*first* 128 coordinates of the embedding. For Qwen3-Embedding that is
principled — the model is trained so its leading coordinates carry the most
information. A plain decoder has no such ordering, so for the decoder backbones
the slice is arbitrary, and a weak result in those columns says as much about
the reduction as about the method.

`--reduce pca` is the principled alternative: encode at the encoder's full
width, project onto the top 128 principal directions of the *training*
features, and hand the network the most informative 128 numbers the backbone
has instead of the first 128. The basis is fitted once per (encoder, training
corpus) by uncentered SVD — no mean subtraction, so the agent cosines the
attention bias depends on keep their geometry (`reduce.py`). Test features
never touch the fit. Predictions land under `stepfinder-pca.s<seed>/`; the two
axes compose, so the paper-faithful and the principled pipeline are always
separate rows, never a silent replacement. Full-width features are cached in
`_sf-featsw/`, and a later `--reduce slice` run derives its 128/32 inputs from
that cache bit-identically instead of re-running the encoder.

Across all twelve subsets the paper's own 0.6B encoder wins seven, against four
for `deepseek-8b` and one for `qwen3.5-9b`, with the best mean (0.251) and
median (0.210) of the three.

The interesting part is that **validation accuracy ranks them backwards.** On
both training corpora the 0.6B encoder scores *lowest* on held-out training
tasks — 0.257 against 0.396 and 0.329 on Hand-Crafted — yet wins most subsets;
`qwen3.5-9b` has the highest validation accuracy on Algorithm-Generated (0.467)
and the worst mean. Choosing an encoder the way the training pipeline chooses an
epoch would pick the worst of the three.

That is the same overfitting that inflates the paper's headline, seen on another
axis: more representational capacity fits the eighteen-variations-on-87-tasks
training corpus more closely and transfers worse. Three encoders is a
demonstration, not an estimate, and the per-subset picture is heterogeneous —
`deepseek-8b` is 0.23 ahead on `mmlu_pro` — but the aggregate ordering is
consistent. `IMPLEMENTATION.md` has the full table and the confounds.

Sharing the axis with OAT pays off too: on `deepseek-8b` both methods write
into the same directory, so one `rb_shared.rb_metrics` call scores them
together. On `ww/hand-crafted` OAT takes it — Acc@1 0.141 against StepFinder's
0.097 — which is not the ordering either paper would predict, and worth the
caveats in
[`IMPLEMENTATION.md`](IMPLEMENTATION.md#oat-and-stepfinder-on-the-same-checkpoint).

## GT settings

The vendored features carry no task answer, so `--gt without` is both the
faithful setting and the default; results land in **`outputs-rb-nogt/`**.
`--gt with` appends the repo's standard `The Answer for the problem is: ...`
line to the **first step's content** before encoding, and writes to
**`outputs-rb-gt/`**.

"First step" is precise and deliberate. StepFinder has no question row — the
model sees only `history` — so there is no natural home for the answer. In
`ww/hand-crafted` and `correct-error` that first step is the user turn carrying
the question; in `traceelephant` it is an orchestrator turn or a tool call, and
the placement is a convention. It is the only placement that leaves every other
step's vector untouched, which is what lets a with-GT run reuse the without-GT
cache and re-encode a single row instead of the whole corpus.

The trained model is shared between the two settings: it always learns from
without-GT features, so only what it reads at scoring time changes.

Measured across all twelve subsets, the answer does not help: mean Δ `step@1`
is **−0.016**, and it improves only 3 of 12. That is the opposite of its effect
on the prompting baselines, and it follows from the method. StepFinder never
reasons about the answer; appending it perturbs one of `T` content vectors,
embedded by a model that never saw such text in training. The damage duly
scales with how much of the trajectory that one step represents — correlation
−0.72 against `1 / mean T`, with the two shortest corpora losing nine and eight
points and the longest losing almost nothing.

## What the paper reports

Who&When, step-level accuracy, mean ± sd over runs (Table 2):

| method | Algorithm-Generated | Hand-Crafted |
|---|---|---|
| Random | 15.08 ± 1.71 | 4.60 ± 3.54 |
| GPT-4o all-at-once | 11.90 ± 1.59 | 2.87 ± 1.99 |
| GPT-4o step-by-step (with answer) | 20.11 ± 2.55 | 6.90 ± 1.73 |
| BiGRU | 23.28 ± 0.99 | 12.64 ± 2.15 |
| **StepFinder** | **29.63 ± 1.50** | **22.99 ± 2.15** |

Its test sets *are* this repo's `data/ww/algorithm-generated` and
`data/ww/hand-crafted` — the Hand-Crafted corpus is byte-identical, all 58
records — so the comparison is direct rather than approximate.

## What this reproduction gets

Five training seeds, `qwen3-embedding-0.6b`, `--gt without`, step-level
accuracy over the whole subset:

| | Algorithm-Generated | Hand-Crafted |
|---|---|---|
| paper, Table 2 | 0.2963 ± 0.0150 | 0.2299 ± 0.0215 |
| here, `--model-selection vendored` | 0.2778 ± 0.0332 | 0.2172 ± 0.0154 |
| here, `--model-selection val` (default) | 0.2159 ± 0.0283 | 0.1138 ± 0.0197 |

The middle row is the one that says the port is faithful: run under the paper's
own rule, both subsets land inside one standard deviation of the published
figure.

The gap between the middle and bottom rows is the cost of that rule. Both come
from the same models, the same data and the same seeds — they differ only in
whether the epoch to keep was chosen on the trajectories being reported or on
data held out of training. It is worth +0.06 on Algorithm-Generated and +0.10
on Hand-Crafted, which nearly doubles the latter.

That the inflation is twice as large on Hand-Crafted is not a coincidence.
Selecting on the test set pays in proportion to how far the test set is from
training, and Hand-Crafted is the further of the two: Magentic-One traces up to
130 steps long, against a training corpus the model has 2,604 near-variations
of.

Ranking quality survives the honest protocol far better than top-1 does:

| | Acc@1 | Acc@3 | MRR@3 | AUROC |
|---|---|---|---|---|
| Algorithm-Generated (paper) | 0.2884 | 0.6031 | ~0.47 | — |
| Algorithm-Generated (here) | 0.2159 | **0.5714** | 0.3707 | 0.686 |
| Hand-Crafted (paper) | 0.2126 | 0.3046 | ~0.28 | — |
| Hand-Crafted (here) | 0.1138 | 0.2414 | 0.1649 | 0.773 |

An AUROC near 0.77 on Hand-Crafted, where Acc@1 is 0.11, is the shape of the
result: the model ranks the guilty step well above chance and often inside the
top three; it is picking it *first* that the distribution shift costs.

One caution when carrying these metrics to other corpora. Acc@3 only means
something when trajectories are longer than three steps, and much of
`correct-error` is not — 76% of `arc` and 58% of `mmlu_pro` are three steps or
fewer, so "the top three" is the whole trajectory and Acc@3 is 1.0 by
construction. Both `ww` subsets are safe; see `IMPLEMENTATION.md` for the
per-subset table.

## How this repo scores it

Two views of the same prediction files:

- **`python -m stepfinder.report`** — the repo's shared protocol: `step@1` and
  `agent@1` over the same seeded val/test splits every other baseline is scored
  on. `step@1` is the highest-scoring step; `agent@1` is the role of that step,
  since StepFinder predicts no agent of its own. Add `--diagonal` for the
  in-corpus family.
- **`python -m rb_shared.rb_metrics`** — the paper's view: Acc@K, MRR@3 and
  tolerance accuracy, plus AUROC and AUPRC. Nothing is re-run; it reads the
  scores already stored in each file.

The conformal columns in that second table stay empty for StepFinder, and that
is correct: it has no calibration set, no nonconformity score and no threshold.
Synthesising one would read as a coverage guarantee and carry none.

## Layout

- `encode.py` — the frozen embedder: last-token pooling, the width slice, and a
  memo so a repeated string costs one forward pass.
- `features.py` — a record to the three arrays the network eats, including
  which field holds the agent and where the answer goes.
- `model.py` — the network, transcribed from `vendored/StepFinder/model.py`.
- `collate.py` — padding, transcribed from `vendored/StepFinder/collate_fn.py`.
- `train.py` — the epoch loop and metrics lifted out of `main.py`, plus the
  paper's Table 1 presets.
- `protocol.py` — the two training families; the only file that knows they
  differ.
- `predict.py` — the four stages, resumable at each.
- `sweep.py` — the grid; shells one child per encoder × subset.
- `report.py` — the shared report, plus the diagonal table.
- `configs/` — `<ds>.yaml` (what to run) and `report_<ds>[_incorpus].yaml`
  (what to score); see [`configs/README.md`](configs/README.md).

## The pipeline

| stage | does | writes |
|---|---|---|
| `feats-train` | embed the trajectories the model learns from | `_sf-feats/*.pt` under the training root |
| `train` | fit one model per seed | `_sf-ckpt/<preset>/<selection>/<seed>/` |
| `feats-test` | embed the trajectories to be scored | `_sf-feats/*.pt` beside the predictions |
| `score` | run the model, one trajectory at a time | `stepfinder.<seed>/<id>.json` |

One trajectory at a time is not an implementation detail. The vendored position
prior divides by the *batch's* padded width (`model.py:242`), so a short
trajectory batched beside a long one gets a flatter prior than the paper
describes. Scoring alone makes the divisor the trajectory's own length — the
paper's equation exactly — and makes a prediction independent of whatever else
happened to be in flight beside it, which is what this repo's per-trajectory,
resumable, sliceable outputs require. Training keeps the vendored batch of 16.

## Running

From the repo root. The front door is `scripts/stepfinder/run.sh`:

```bash
DATASET=ww MODEL=qwen3-embedding-0.6b GPU=0 bash scripts/stepfinder/run.sh    # both subsets
DATASET=ww SUBSET=hand-crafted MODEL=qwen3-embedding-0.6b GPU=0 bash scripts/stepfinder/run.sh
DATASET=ww MODEL=qwen3-embedding-0.6b STAGES=feats-train,train bash scripts/stepfinder/run.sh
DATASET=ww MODEL=qwen3-embedding-0.6b PROTOCOL=in-corpus GPU=1 bash scripts/stepfinder/run.sh
DATASET=ww MODEL=qwen3-embedding-0.6b GT=with GPU=1 bash scripts/stepfinder/run.sh
DATASET=ww DRY_RUN=1 bash scripts/stepfinder/run.sh                          # preview
```

Every stage resumes on file existence — a cached tensor, a saved checkpoint, a
written prediction. Rerun the same command after a crash and only the missing
work executes. Then build the tables (CPU only):

```bash
python -m stepfinder.report --config baselines-rp/stepfinder/configs/report_ww.yaml --check-only
python -m stepfinder.report --config baselines-rp/stepfinder/configs/report_ww.yaml
python -m stepfinder.report --config baselines-rp/stepfinder/configs/report_ww_incorpus.yaml --diagonal
python -m rb_shared.rb_metrics --pred-root outputs-rb-nogt/ww
```

Installing with `pip install -e ".[rb]"` puts `stepfinder` and `rb_shared` on
the path; without it, prefix commands with `PYTHONPATH=baselines-rp` as
`run.sh` does.

## Cost

No tokens, ever. The whole method is one embedding pass per step plus a network
of 278k parameters — small enough to train on a CPU.

Embedding dominates, and it happens once. The corpora hold 36,363 steps and the
two training sets 81,429 more. Every step needs its content embedded; agent
names collapse to a handful of distinct strings, so the memo removes almost all
of that half. Measured with `qwen3-embedding-0.6b` on a shared H200, the 2,604
Hand-Crafted training trajectories (55,591 steps) took about half an hour, so
the whole corpus plus both training sets is roughly an hour per encoder — an
order of magnitude cheaper than OAT's passes through an 8-to-27B decoder. The
9B and 8B encoders cost proportionally more.

Training is minutes per seed and runs once per (encoder, training set, preset),
however many subsets follow: `correct-error` and `traceelephant` both map to the
Hand-Crafted corpus, so once `ww/hand-crafted` has trained they need only
encoding and scoring. Scoring is 6 ms per trajectory at the median, 28 ms at the
worst.

Keep training on the GPU. Unlike OAT, whose neural CDE is faster on a CPU
because it is bound by kernel-launch overhead, StepFinder's core is a BiLSTM
that cuDNN accelerates heavily — 18.2 optimizer steps per second on an H200
against 0.4 on CPU. That matters most for the in-corpus family, whose small
partitions need many epochs to reach a comparable update count.

Storage is small. A cached trajectory is a 128-wide and a 32-wide float32 matrix
— 57 MB for all 2,604 Hand-Crafted training trajectories, 2.8 MB for
`ww/hand-crafted` — and a checkpoint is 1.1 MB. Rerunning costs nothing, which
is the opposite of the prompting baselines' profile: they pay again every time.
