# StepFinder — implementation notes

What this package changes from `vendored/StepFinder/`, and why. The rule is the
repo's usual one: the vendored code is the reference, and anything that differs
is either forced by this repo's data or belongs to the uniform shell every
baseline shares (I/O, resume, evaluation). Nothing in the method itself was
"improved".

Read [`README.md`](README.md) first for what the method does.

## What is verbatim

| here | vendored source | what it is |
|---|---|---|
| `model.py` | `model.py` (whole file) | the BiLSTM, the agent-aware attention and gate, the scoring head, the multi-scale difference, the position prior, the loss |
| `collate.py` — `sequence_collate_fn` | `collate_fn.py:5-43` | right padding, float32 mask |
| `collate.py` — `SequenceDataset` | `main.py:59-67` | the dataset wrapper |
| `encode.py` — `last_token_pool` | `Q3Emb.py:43-58` | pooling |
| `encode.py` — `Encoder._forward` | `Q3Emb.py:60-92` | tokenize, forward, slice to width |
| `encode.py` — `Encoder.embed` | `feature_construction.py:41-64` | zeros for empty text, pad/truncate to width, degrade to zeros on failure |
| `train.py` — `set_seed` | `main.py:19-30` | seeding, including the CUDA determinism block |
| `train.py` — `evaluate` | `main.py:74-139` | Acc, Acc@2, Acc@3, MRR@3, tolerance accuracy |
| `train.py` — `train_one_epoch` | `main.py:202-227` | AdamW step, clip at 1.0 |
| `train.py` — `PRESETS["alg"]` | `main.py:271-295` CLI defaults | also the paper's Table 1 Alg column |

`tests/test_stepfinder_pipeline.py` proves it rather than asserting it: it
imports the vendored modules and compares. The network's logits and temporal
loss bit-for-bit on a shared state dict across four batch shapes; the loss; the
multi-scale difference at every scale including the nonstandard `s=3`; the
collate; the metric block; and one full training epoch, weight by weight.

The encoder and featurizer were checked once more outside the suite, against the
real `Qwen3-Embedding-0.6B` checkpoint that the tests cannot load: on a full
`ww/hand-crafted` trajectory our payload and the vendored
`TemporalFeatureExtractor.process_log` agree bit-for-bit on both the 29x128
content matrix and the 29x32 agent matrix, maximum absolute difference 0.0.

`PRESETS["hc"]` cannot be checked that way. The Hand-Crafted hyperparameters
appear only in the paper's Table 1 — the shipped CLI defaults are the
Algorithm-Generated column — so the test pins the paper's numbers and pins that
the Alg preset still equals `main.py`'s defaults, which is what would catch a
drift.

## Forced by this repo's data

### Which field holds the agent

The vendored featurizer reads the agent as `step.get("name") or
step.get("role")` (`feature_construction.py:91`). Applying that rule here would
destroy the agent signal on exactly the subset the paper's Algorithm-Generated
model targets.

`data/ww/algorithm-generated` stores the two fields the other way round from the
upstream Who&When copy it came from:

| | `role` | `name` |
|---|---|---|
| `vendored/Agents_Failure_Attribution/Who&When/Algorithm-Generated/1.json` | `assistant` | `Excel_Expert` |
| `data/ww/algorithm-generated/1.json` | `Excel_Expert` | `assistant` |

Across all 1,099 steps of that subset, `name` is only ever `user` (731) or
`assistant` (368), while `role` carries `Computer_terminal`,
`Verification_Expert` and 189 others. The vendored rule would hand the model the
string `"assistant"` as the agent identity for every model turn, collapsing 191
agents into two — and the agent-aware bias and gate are the modules that then
receive it.

The vendored *training* files do follow the upstream convention. So the rule
here is an explicit per-source setting, not a fallback chain: `name` for the
vendored training corpus, `role` for every corpus in `data/`, which is this
repo's documented agent identity. `features.AGENT_FIELD_VENDORED` and
`AGENT_FIELD_REPO` name the two, and
`test_agent_identity_reads_role_not_name_on_repo_corpora` is the gate that keeps
a future edit from quietly reverting to the fallback.

`ww/hand-crafted`, `correct-error/*` and `traceelephant/*` carry no `name` key at
all and are unaffected either way.

### Agent-string normalization, and why it is the default

The agent embedding is a text embedding of the agent's *name*, so the attention
bias `cos(r_i, r_j)` only carries information when the training and test
vocabularies overlap. Under family A they do not:

| training set | vocabulary | scored subset | vocabulary |
|---|---|---|---|
| Hand-Crafted | `Orchestrator, Assistant, WebSurfer, FileSurfer, user` | `ww/hand-crafted` | `Orchestrator (thought)`, `Orchestrator (-> WebSurfer)`, `human`, … |
| Hand-Crafted | same | `correct-error/*` | `Planner, Coder, CodeExecutor, user` |
| Hand-Crafted | same | `traceelephant/swe` | `bash, str_replace_editor, submit` |
| Algorithm-Generated | 206 `*_Expert` + `Computer_terminal` | `ww/algorithm-generated` | 191 `*_Expert` + `Computer_terminal` |

Only the last pairing matches out of the box. `--agent-normalize standardize`
(the default) applies `baselines.shared.common.standardize_role`, collapsing
every `Orchestrator*` variant onto `Orchestrator` — which is the string the
upstream Who&When data, and therefore the training corpus, actually uses.
`--agent-normalize raw` keeps the vendored behaviour, and the setting is stamped
in every prediction file and every `_run.json`.

Measured, as the fraction of a subset's distinct agent strings that appear
verbatim in its training set's vocabulary:

| scored subset | raw | standardized |
|---|---|---|
| `ww/hand-crafted` | 27% | **67%** |
| `traceelephant/magentic` | 43% | 50% |
| `correct-error/gaia` | 50% | 50% |
| `traceelephant/swe` | 14% | 14% |
| `ww/algorithm-generated` | 17% | 17% |

It more than doubles overlap on the subset it was chosen for and is a no-op
everywhere there are no orchestrator variants to collapse — it can only merge
strings, never invent them. The last row understates the real match: the
Algorithm-Generated vocabulary is 205 task-specific `*_Expert` names, so exact
overlap is low by construction while the embeddings of `Algebra_Expert` and
`AnalyticalReasoning_Expert` remain close. Exact overlap is a lower bound on
what the cosine bias actually sees.

### Trajectories whose gold step is unusable

`compute_loss` takes `targets.argmax(dim=1)` (`model.py:336`). On an all-zero
one-hot that silently returns 0, teaching the model that step 0 was to blame.
Two `traceelephant/magentic` records carry a `mistake_step` past the end of
their history, and 184 `ww` records store the gold as a string rather than an
int.

`features.label_vector` returns `None` rather than a zero vector when the gold
cannot be used, and `build_features` records `trainable: False`. Any training
set drops those; they are still **scored**, since scoring needs no label, and
they still get a prediction file so the shared report sees a complete corpus.
Their `gold_on_filtered_step` is `true`.

### No model-step filter

OAT drops human and environment turns before scoring
(`oat/IMPLEMENTATION.md`). StepFinder has no such rule: the vendored model
scores every unpadded step and argmaxes over the whole mask
(`main.py:106-108`). So StepFinder can predict a `user` or `human` turn, and
`score_step_indices` is always the full `range(num_steps)`. Kept as-is —
filtering would be a change to the method, not to the plumbing. It also means
`gold_on_filtered_step` has a narrower meaning here than in OAT: the only
unreachable gold is one pointing past the end of the history.

## Adaptation glue

### Scoring one trajectory at a time

`model.py:242-244` computes the position prior as `-(t / (T - 1 + 1e-6))`, where
`T` is the **batch's padded width**, not the trajectory's own length. A
trajectory therefore scores differently depending on which other trajectories
shared its batch. That is incompatible with this repo's contract, where a
prediction file is per-trajectory, resumable and sliceable: `--start_idx`
would change results.

Every other term is batch-invariant — the LSTM is packed, attention is masked
before the softmax, the gate uses a masked mean, and the difference is
normalized by a masked mean. So the discrepancy has a closed form, verified to
6e-8 in `test_batched_position_bias_differs_from_single_by_a_closed_form`:

```
logits_batched[i, :L] - logits_alone[:L] == gamma * (t/(L-1) - t/(T_max-1))
```

Scoring at batch size 1 makes the divisor the trajectory's own length, which is
the paper's Eq. 9 as written. Training keeps the vendored batch of 16, so the
fitted weights are the vendored ones. Every prediction records
`eval_batch_size: 1` and `position_bias_denominator: "trajectory"`.

The magnitude is not cosmetic: with the Hand-Crafted preset's `gamma = 0.75` and
trajectories up to 130 steps, the shift reaches 0.75 in logit units.

### Checkpoint selection

`main.py:231-251` evaluates on `--test_dir` every epoch and saves the
best-scoring epoch. No validation split exists in the vendored repository, so
every published number is a maximum over fifty epochs on the reported data.

`--model-selection val` (default) holds out `--val-frac` of the *training*
questions — `holdout_by_question` groups by exact question string and moves
whole groups, because the training corpus holds ~18 near-duplicate trajectories
per task and a random split would put near-duplicates on both sides. Family B
uses the same machinery over its own partition.

`--model-selection vendored` restores the original rule and **exits rather than
scoring** if `score` is in `--stages`. Its checkpoints live under a separate
path component so they cannot be confused with real ones.

When a holdout would be smaller than `--val-min-tasks` (default 4), early
stopping is skipped and the final epoch is kept. That is not hypothetical:
family B's partitions run to 13 trajectories on `traceelephant/swe`. The
outcome is recorded as `val(fallback:last)` in the checkpoint, the prediction
files and `_run.json`.

One further fallback: the vendored rule requires a *strict* improvement over a
0.0 start (`main.py:247`), so on a hard subset it saves nothing at all and the
run ends with no checkpoint on disk. This package must always return a model,
so it keeps the last epoch and marks the selection `(fallback:last)`.

### The two protocols

`protocol.py` is the only module that knows the families differ. It builds a
`TrainingSet` and a list of `Job`s; everything after that — encoding, caching,
training, scoring, the output schema, resume — is shared code. The subset →
training-set → preset mapping is config, not code
(`train_set_overrides` / `preset_overrides`).

### Pair each subset with its own agent system

The mapping is not arbitrary and it was wrong once. Who&When's
Algorithm-Generated half was produced by **CaptainAgent** and its Hand-Crafted
half by **Magentic-One** (`vendored/Agents_Failure_Attribution/README.md:39`).
TraceElephant's two subsets were produced by the same two systems. The first
version of this package sent only `ww/algorithm-generated` to the
Algorithm-Generated corpus and everything else to Hand-Crafted, which put
`traceelephant/captain` — CaptainAgent trajectories — on the Magentic-One model.

The agent embedding is a text embedding of the agent's *name* and the attention
bias is the cosine between two such names, so a vocabulary mismatch does not
degrade the signal gracefully; it deletes it. Of `captain`'s 1,746 steps, 82.8%
carry a name in the Algorithm-Generated vocabulary against 57.3% for
Hand-Crafted, because its six `*_Expert` names are in the former's 205-name list
and absent from the latter's five (`Orchestrator, Assistant, WebSurfer,
FileSurfer, user`). `magentic` runs the other way, 97.2% against 31.9%.

Repairing it moved three numbers and was worth doing for a fourth reason:
`captain` shares 52 of 85 questions with Hand-Crafted but only 22 with
Algorithm-Generated, so the correct pairing is also the less contaminated one.

`captain` step@1 on the test split, split seeds 1–3, averaged over the five
training seeds:

| | Hand-Crafted (wrong) | Algorithm-Generated (right) |
|---|---|---|
| `qwen3.5-9b` | 0.140 | **0.169** |
| `deepseek-8b` | 0.115 | **0.143** |
| `qwen3-embedding-0.6b` | **0.195** | 0.153 |
| task overlap | 52 of 85 | **22 of 85** |

Both of the manuscript backbones gain about three points; the paper's own 0.6b
encoder loses four. That split is not evidence the pairing is wrong — the 0.6b
column also carries the Hand-Crafted preset's much lower learning rate, so the
two arms differ in optimizer settings as well as in corpus, and the paper offers
no way to separate them. The pairing rests on provenance and vocabulary, which
are facts about the data, not on which arm scored higher.

`correct-error` matches neither system, so its Hand-Crafted assignment is a
fallback rather than a match, and every `correct-error` number in this document
is out-of-domain transfer.

Family B reproduces the report's split over the **file list**
(`_get_sorted_json_files`), not over `load_records` output, because
`load_records` drops empty histories (`baselines/prompting/predict.py:74-76`)
and a partition built after that drop would not line up with the val and test
ids being scored. `test_in_corpus_partition_matches_the_reports_own_split`
pins that the train partition and the scored ids are disjoint and together
cover the corpus.

### The `--gt with` extension

The answer is appended to `history[0]["content"]` as
`"\nThe Answer for the problem is: {ground_truth}"` — the line every prompting
baseline in this repo interpolates. StepFinder has no question row, so there is
no natural anchor; the first step is the user turn carrying the question in
`ww/hand-crafted` (53 of 58 records are exactly the question) and all of
`correct-error`, but an orchestrator turn in `traceelephant/magentic` and a
tool call in `traceelephant/swe`. The docs say "the first step's content" for
that reason.

Only that one vector changes, so `derive_gt_features` copies the rest from the
without-GT cache: 2,630 forward passes instead of 36,363.

### Where the artifacts live

```
outputs-rb-nogt/stepfinder-regen/<train-set>/<model>/_sf-feats/<id>.pt        training features
outputs-rb-nogt/stepfinder-regen/<train-set>/<model>/_sf-ckpt/<preset>/<selection>/s<seed>/   family A models
outputs-rb-<gt|nogt>/<ds>/<subset>/<model>/_sf-feats/<id>.pt                  test features
outputs-rb-nogt/<ds>/<subset>/<model>/_sf-ckpt/<preset>/<selection>/e<seed>/  family B models
outputs-rb-<gt|nogt>/<ds>/<subset>/<model>/stepfinder.{s,e}<seed>/<id>.json   predictions
```

The leading underscore keeps them out of the report's way: it reads only the
method directories named in its config, and `OutputWriter.done_ids` counts only
digit stems.

`stepfinder-regen/<train-set>/<model>/` is the analogue of OAT's
`mcp-atlas/train/<model>/`: a training corpus with no home in `data/`, given the
same `<ds>/<subset>/<model>` shape so the "find the `outputs…` component and
rebuild" logic ports over. StepFinder ships two training corpora, so the
training-set name occupies the subset slot.

`<preset>/<selection>` in the checkpoint path is new. Family A's checkpoint is
keyed by (training set, hyperparameters, seed) and is independent of the subset
being scored — that is what lets one model serve every subset that maps to it.
But the preset is configurable per subset, so two subsets sharing a training
set with different hyperparameters would otherwise collide silently. A tuned
preset gets `custom-<8 hex of the sorted hyperparameter dict>`.

**Checkpoints always live in the without-GT tree, for both families.** For
family A that is automatic — the regenerated corpus has no answer to inject.
For family B it is a decision: it trains on the corpus's *without-GT* features
even under `--gt with`, so the property "one model, two GT settings" holds for
both families and `outputs-rb-gt/` never contains a `_sf-ckpt`.

### Step and agent predictions

`predicted_step` is the argmax over the softmax across valid steps, as an
absolute index into `history` — StepFinder scores every step, so no remapping
is needed. `predicted_agent` is `history[predicted_step]["role"]`, raw: the
method predicts no agent of its own, and the report's `standardize_role`
handles the `Orchestrator (thought)` variants at scoring time.

`scores` holds the softmax probabilities, so `predicted_step` can be read
straight off the stored numbers; `scores_logit` holds the pre-softmax values.
Both are stored because pooled AUROC over probabilities is length-biased — a
uniform distribution is 0.2 on a 5-step trajectory and 0.008 on a 130-step one,
and `rb_metrics` pools across trajectories. `--score-key scores_logit` gives
the unbiased view.

### No conformal set

`conformal_steps`, `conformal_threshold` and `conformal_alpha` are `null`.
StepFinder has no calibration set, no nonconformity score and no threshold.
Synthesising one — say, every step above `1/T` — would look like a coverage
guarantee and carry none. `rb_shared/rb_metrics.py` was changed to report those
four columns as empty rather than fall back to the top-1 set, which is right
for OAT (a conformal set that selects nothing still means "the top step") and
wrong for a method with no conformal machinery at all.

### The in-corpus family's report

The shared report scores every listed method on every listed seed
(`report.py:89-99, 205-229`), so `stepfinder.e7` — which predicts only seed 7's
ids — gets 19 meaningless rows out of 20, and the mean over them is worse than
meaningless. Rather than fork the shared rules, `stepfinder/report.py` runs them
untouched and adds `diagonal.tsv`, keeping only the cell where the method's
seed and the split's seed agree. `--check-only` reporting `PARTIAL` for the
in-corpus family is expected, not a failure.

## What the protocol change costs

`ww`, `qwen3-embedding-0.6b`, five training seeds, `--gt without`. Every row
below is the *same architecture, same training data, same seeds*, scored on the
same trajectories. Only the rule for picking which epoch to keep differs.

| | Algorithm-Generated | Hand-Crafted |
|---|---|---|
| paper, Table 2 | 0.2963 ± 0.0150 | 0.2299 ± 0.0215 |
| **`--model-selection vendored`** (reproduces `main.py:231-251`) | **0.2778 ± 0.0332** | **0.2172 ± 0.0154** |
| **`--model-selection val`** (this package's default) | **0.2159 ± 0.0283** | **0.1138 ± 0.0197** |

Both vendored-rule figures sit inside one standard deviation of the paper's, so
nothing in the network, the features or the training loop is losing accuracy.

The rule is worth +0.062 on Algorithm-Generated (a 29% relative inflation) and
+0.103 on Hand-Crafted (91%). The asymmetry follows the domain gap: selecting on
the test set pays in proportion to how far the test set is from training, and
Hand-Crafted is the further of the two.

It also stops far earlier — epochs kept were 3, 6, 13, 15, 16 and 4, 4, 11, 13,
13 under the vendored rule, against 12–17 and 13–42 under the default. It is
early-stopping on a 58- or 126-sample accuracy that is mostly noise, and keeping
whichever bounce happened to be highest.

One thing not to conflate: every number above is `step@1` on the scored corpus.
The *validation* accuracy the default rule selects on is a different quantity,
and a higher one — see the next table.

The honest run's other numbers, mean over the five seeds:

| | Algorithm-Generated | Hand-Crafted |
|---|---|---|
| validation accuracy (held-out *training* tasks) | 0.361 | 0.257 |
| `step@1` over the corpus | 0.216 | 0.114 |
| Acc@3 | 0.571 | 0.241 |
| MRR@3 | 0.371 | 0.165 |
| tolerance accuracy, δ=1 | 0.378 | 0.183 |
| AUROC | 0.686 | 0.773 |

Two things are worth sitting with. Validation accuracy, measured on held-out
tasks from the *training* corpus, reproduces or exceeds the paper's range on
both subsets; on the actual Who&When corpus the same models score well below
it. The regenerated training set is eighteen variations on 145 tasks, so a
held-out task from it is still far closer to training than a real Who&When
failure is.

And ranking quality survives the shift much better than top-1 does. Acc@3 on
Algorithm-Generated is 0.571 against the paper's 0.603 — nearly intact — while
Acc@1 falls from 0.288 to 0.216. On Hand-Crafted an AUROC of 0.773 sits beside
an Acc@1 of 0.114. The model puts the guilty step well above chance and often
inside the top three; what the distribution shift costs is putting it *first*.

### Other measurements from the same encoder

`--gt with` on `ww/hand-crafted` moves `step@1` from 0.1034 to 0.1207 for seed
42 — the answer helps a little, as it does for the prompting baselines.

`correct-error`, family A, mean of five seeds over the whole subset. The last
column is the share of trajectories with three steps or fewer, for which "the
top three steps" is the entire trajectory and Acc@3 is 1.0 by construction:

| subset | n | mean T | Acc@1 | Acc@3 | MRR@3 | AUROC | T ≤ 3 |
|---|---|---|---|---|---|---|---|
| `arc` | 304 | 5.3 | 0.534 | *0.841* | 0.684 | 0.846 | **76%** |
| `math500` | 157 | 6.3 | 0.412 | *0.730* | 0.555 | 0.778 | **36%** |
| `mmlu_pro` | 92 | 6.1 | 0.335 | *0.759* | 0.536 | 0.792 | **58%** |
| `gaia` | 50 | 10.9 | 0.292 | 0.436 | 0.349 | 0.623 | 24% |
| `hotpot` | 578 | 12.8 | 0.248 | 0.526 | 0.365 | 0.744 | 3% |
| `musique` | 312 | 14.7 | 0.203 | 0.446 | 0.306 | 0.719 | 3% |
| `wikimqa` | 733 | 14.5 | 0.198 | 0.407 | 0.286 | 0.709 | 0.3% |
| `ww/algorithm-generated` | 126 | 8.7 | 0.216 | 0.571 | 0.371 | 0.686 | 0% |
| `ww/hand-crafted` | 58 | 51.6 | 0.114 | 0.241 | 0.165 | 0.773 | 0% |

Two things to hold onto before comparing any of these rows.

**The italicised Acc@3 figures are mostly bookkeeping.** Three quarters of
`arc` is two or three steps long, so its 0.841 says more about the corpus than
the method. Read Acc@1 and AUROC on those subsets, and Acc@3 only on `hotpot`,
`musique`, `wikimqa` and `ww`.

**Short trajectories also hand the position prior most of the decision.** On a
two-step trajectory the Hand-Crafted preset's `gamma = 0.75` puts step 1 three
quarters of a logit behind step 0 before the network says anything, and the
learned logits on these corpora differ by about 0.3. That is why a
well-trained model and a barely-trained one diverge so sharply on `arc`'s Acc@1
while agreeing on its Acc@3: the prior decides, and only a model confident
enough to overrule it scores.

### The answer does not help, and on short trajectories it hurts

Both GT settings are now complete for `qwen3-embedding-0.6b`, family A, five
seeds — twelve subsets scored twice on the same models, differing only in
whether the task's answer was appended to the first step's content.

| | mean Δ `step@1` | subsets improved |
|---|---|---|
| `--gt with` vs `--gt without` | **−0.016** | 3 of 12 |

The answer is not merely useless here, it is mildly harmful, which is the
opposite of its effect on the prompting baselines. The reason is structural:
StepFinder never reasons about the answer. Appending it changes exactly one of
a trajectory's `T` content vectors, and that vector was embedded by a model
trained on text without it — so what reaches the network is a perturbed feature,
not new information.

That predicts the damage should scale with how large a share of the trajectory
step 0 represents, and it does. Correlating the per-subset delta against
`1 / mean T` gives **−0.72**:

| subset | mean T | Δ |
|---|---|---|
| `correct-error/arc` | 5.3 | −0.092 |
| `correct-error/mmlu_pro` | 6.1 | −0.085 |
| `correct-error/gaia` | 10.9 | −0.016 |
| `correct-error/wikimqa` | 14.5 | −0.000 |
| `traceelephant/captain` | 20.5 | +0.002 |
| `ww/hand-crafted` | 51.6 | −0.007 |

The two worst-hit subsets are the two shortest. On a five-step `arc` trajectory
step 0 is a fifth of everything the model sees, and corrupting it costs nine
accuracy points; on a 52-step `ww/hand-crafted` trace it is 2% of the input and
costs almost nothing.

So the GT axis is close to inert for this method, and where it is not, it is
measuring a distribution shift rather than an oracle signal. Both settings are
kept because the repo's paired comparison depends on every baseline supporting
the flag, not because the answer is expected to inform this one.

### Encoders: validation accuracy ranks them backwards

All three encoders, family A, five seeds, `--gt without`, all twelve subsets.

"Val acc" is what the default selection rule optimizes — accuracy on held-out
*training* tasks. It is a property of the training corpus, so every subset
mapping to the same corpus shares it:

| training corpus | `qwen3-embedding-0.6b` | `deepseek-8b` | `qwen3.5-9b` |
|---|---|---|---|
| Hand-Crafted | 0.257 | **0.396** | 0.329 |
| Algorithm-Generated | 0.360 | 0.454 | **0.467** |

The paper's own encoder is **last on both**. Now the corpus results:

| subset | `0.6b` | `ds8b` | `9b` |
|---|---|---|---|
| `correct-error/arc` | 0.534 | **0.634** | 0.445 |
| `correct-error/math500` | 0.411 | **0.445** | 0.364 |
| `correct-error/mmlu_pro` | 0.335 | **0.563** | 0.374 |
| `correct-error/gaia` | 0.292 | **0.308** | 0.276 |
| `correct-error/hotpot` | **0.248** | 0.191 | 0.222 |
| `ww/algorithm-generated` | **0.216** | 0.190 | 0.179 |
| `correct-error/musique` | **0.203** | 0.183 | 0.200 |
| `correct-error/wikimqa` | **0.198** | 0.129 | 0.162 |
| `traceelephant/magentic` | **0.174** | 0.044 | 0.073 |
| `traceelephant/captain` | **0.162** | 0.146 | 0.155 |
| `ww/hand-crafted` | 0.114 | 0.097 | **0.117** |
| `traceelephant/swe` | **0.086** | 0.036 | 0.068 |
| **subsets won** | **7** | 4 | 1 |
| **mean** | 0.248 | 0.247 | 0.220 |
| **median** | **0.210** | 0.187 | 0.190 |

**The ordering is inverted.** The encoder with the lowest validation accuracy on
both training corpora wins the most subsets and has the best median; on the mean
it and `deepseek-8b` are tied to a thousandth (0.248 against 0.247), so the
claim rests on wins and median, not on the mean.
`qwen3.5-9b`, which has the highest validation accuracy on Algorithm-Generated
(0.467), has the worst mean and wins one subset in twelve. Selecting an encoder
the way the training pipeline selects an epoch would pick the worst of the
three.

The mechanism is the one that inflates the paper's headline. The regenerated
training set is eighteen LLM-written variations on 87 or 145 tasks; more
representational capacity buys a closer fit to those idiosyncrasies and pays
for it on real trajectories. It is the same overfitting, observed on a
different axis.

Three caveats keep this from being stronger than it is. Three encoders is not a
sample, so this is a demonstration rather than an estimate. The per-subset
picture is genuinely heterogeneous — `deepseek-8b` takes four subsets and is
0.228 ahead on `mmlu_pro`, so nothing here says a decoder is always worse. And
the prefix slice (`Q3Emb.encode` keeps the first 128 coordinates, principled
for an MRL-trained embedder and arbitrary for a decoder) is a confound that
cannot be separated without fitting a projection instead; a `--reduce pca`
option was considered and deliberately left out, because it would no longer be
the method the paper describes. Note, though, that the decoders reach *higher*
validation accuracy under that same arbitrary slice — so the slice is not what
limits them on the corpus.

The practical reading: the paper's own encoder wins most subsets, has the best
central tendency, and is by far the cheapest to run.

**How this claim moved.** It is worth recording that this section was rewritten
three times as coverage grew. On `ww` alone it read "validation ranks encoders
backwards"; at eleven subsets, with one decoder partly done, it read "validation
carries no information"; complete, it is back to the inversion — but supported
by aggregates over twelve subsets rather than two, and with the per-subset
heterogeneity stated. Each intermediate reading was honest about the data it
had and wrong about the conclusion.

### OAT and StepFinder on the same checkpoint

Sharing the `<model>` axis with OAT was a deliberate choice, and this is what
it buys. Both methods freeze `deepseek-8b`, read `ww/hand-crafted`, and write
five seeds into the same directory, so `rb_shared.rb_metrics` scores them side
by side with no extra work:

| method | Acc@1 | Acc@3 | MRR@3 | AUROC | conformal F1 |
|---|---|---|---|---|---|
| `oat` | **0.141** | **0.328** | **0.214** | **0.695** | 0.138 |
| `stepfinder` | 0.097 | 0.207 | 0.141 | 0.668 | — |

On this subset the unsupervised method wins, which is not the ordering either
paper would predict. Two things temper it. StepFinder is transferring from a
Magentic-One training corpus to 130-step Who&When traces, its hardest setting;
and `deepseek-8b` is not its own encoder — the vendored prefix slice takes the
first 128 coordinates, which is principled for an MRL-trained embedder and
arbitrary for a decoder. On its own encoder StepFinder scores 0.114 here.

The empty conformal cell is the honest one: StepFinder has no calibration
machinery, and `rb_metrics` now reports that as absent rather than quietly
substituting its top-1 set.

They are not reading the *same vectors*, only the same checkpoint. OAT
mean-pools the hidden states of a step's tokens out of one pass over the whole
serialized document, then projects to 64 dimensions with PCA fitted on
successes. StepFinder pools the last token of each step encoded on its own,
then slices to 128. The comparison is method-against-method at equal model
access, not representation-against-representation.

### The two families, and the budget that decides them

Family B's numbers changed twice during this reproduction, both times because
of a defect rather than a result, and the final table reverses the conclusion
the first two would have supported. The corrections are recorded here because
each is a trap for anyone transplanting a paper's hyperparameters onto a
smaller training set.

**The 50-epoch budget is an update count in disguise.** Fifty epochs over the
vendored 2,604-trajectory corpus is about 8,000 optimizer steps. Over a
91-trajectory in-corpus partition it is about 300 — a fraction of what the
Hand-Crafted preset's learning rate of 1e-5 was tuned for, leaving the network
barely past initialization. The symptom was unmistakable: family B on
`correct-error/arc` predicted step 0 for 79% of trajectories while the gold mode
is step 1, yet still reached Acc@3 0.81 and AUROC 0.77. The ranking was fine;
the untrained network simply let the position prior take the top slot.

`--min-train-steps` (5,000 in the shipped configs) raises the epoch count until
the budget is comparable — 13 trajectories get 5,000 epochs, 173 get 455, and
both land near 5,000 updates. Family A already clears the floor at 50 epochs,
so nothing about the vendored protocol changes.

With that fixed, `step@1` on each seed's val+test ids — family A averaged over
its five training seeds and restricted to those same ids, family B as its
diagonal, so both are scored on identical trajectories. `±` is the spread
across split seeds.

| subset | family A | family B | B − A | B trains on |
|---|---|---|---|---|
| `correct-error/mmlu_pro` | 0.332 ± 0.014 | **0.697 ± 0.032** | +0.365 | 27 |
| `correct-error/arc` | 0.530 ± 0.014 | **0.815 ± 0.010** | +0.285 | 91 |
| `correct-error/musique` | 0.204 ± 0.010 | **0.487 ± 0.023** | +0.283 | 93 |
| `correct-error/hotpot` | 0.245 ± 0.005 | **0.503 ± 0.008** | +0.258 | 173 |
| `correct-error/gaia` | 0.297 ± 0.047 | **0.552 ± 0.072** | +0.255 | 15 |
| `correct-error/math500` | 0.434 ± 0.014 | **0.634 ± 0.014** | +0.199 | 46 |
| `traceelephant/swe` | 0.088 ± 0.019 | 0.210 ± 0.058 | +0.122 | 13 |
| `ww/algorithm-generated` | 0.218 ± 0.016 | 0.291 ± 0.049 | +0.073 | 37 |
| `correct-error/wikimqa` | 0.196 ± 0.005 | 0.246 ± 0.138 | +0.050 | 219 |
| `ww/hand-crafted` | 0.114 ± 0.017 | 0.140 ± 0.059 | +0.027 | 17 |
| `traceelephant/captain` | 0.163 ± 0.018 | 0.169 ± 0.042 | +0.007 | 25 |
| `traceelephant/magentic` | **0.178 ± 0.022** | 0.095 ± 0.028 | −0.083 | 27 |

Family B leads on eleven of twelve, but the twelve rows are not equally strong
and the table should not be read as a scoreboard.

**Six are decisive** — the six `correct-error` subsets above `math500`, where
the margin is four to thirty times family B's own spread. On `mmlu_pro` it more
than doubles family A; on `gaia` fifteen in-domain trajectories beat 2,604
out-of-domain ones. **Four are inside the noise**: `wikimqa` (+0.050 against a
spread of 0.138), `ww/hand-crafted` (+0.027 against 0.059), `captain`, and
`ww/algorithm-generated` at roughly 1.5 spreads. **One loss is decisive**,
`magentic` at three spreads.

Family B is inherently the noisier column — mean across-seed spread 0.044
against family A's 0.017, a factor of 2.6 — and structurally so: family A
averages five trained models per split seed, family B has exactly one. Its
diagonal is a single model's score, and on `correct-error` there are only three
split seeds to average.

The one loss is the subset where family A has the most task overlap with its own
training set: 51 of 91 `magentic` trajectories share a question with the
Hand-Crafted corpus. Family A is unusually strong there for a reason
`train_task_overlap` records in every file.

So the defensible claim is narrower than "in-corpus wins": **on
`correct-error`, a few dozen in-domain trajectories beat 2,604 out-of-domain
ones by a wide and consistent margin; on `ww` and `traceelephant` the two
protocols are close, and the one place family A leads decisively is where it
has seen the task.** That is a result about the paper's training-data strategy
rather than its architecture.

Two earlier versions of this table are worth naming so they are not
rediscovered. The first had family B losing everywhere by two to three fold;
that was the holdout bug below. The second had it losing on the small
Hand-Crafted-preset subsets; that was the budget. `ww/hand-crafted` moved
0.034 → 0.039 → 0.140 across the three, and only the last reflects the method.

### The holdout floor, and why it is 20

An earlier version of this table was much worse for family B, and the cause was
a bug worth recording. The guard that decides whether to early-stop originally
required only four *trajectories* in the holdout. On
`correct-error/math500` that produced a 5-trajectory validation set, on which
the vendored rule — which needs only a strict improvement over 0.0 to latch —
selected **epoch 1**, handing back an all-but-untrained network that scored
`step@1` 0.012, below chance.

Training the full budget instead scored 0.042 on the same data, so selecting on
five samples was actively worse than not selecting at all. `--val-min-size`
(default 20) now sets a floor on holdout *trajectories* alongside
`--val-min-tasks` (default 4) on distinct questions; below either, early
stopping is skipped. Refitting the 32 affected checkpoints moved
`ww/algorithm-generated` from 0.094 to 0.299 and `correct-error/hotpot` from
0.054 to 0.347.

## Infrastructure-level deviations

| deviation | why |
|---|---|
| Loading goes through `rb_shared.extractors.load_extractor`, not `Q3Emb.__init__` | the vendored `AutoModel` call cannot load `../hub/Qwen/Qwen3.5-9B`, a vision-language checkpoint whose text decoder sits a level down and whose `hidden_size` lives under `text_config`. OAT already unwraps it; the decoder it returns emits the same `last_hidden_state` the vendored pooling consumes. |
| `dtype` defaults to `fp16` | `Q3Emb.py:24, 30` pins it; `load_extractor` defaults to `bf16` for OAT. |
| `max_length` is forced to 8192 | the vendored cap (`Q3Emb.py:18`), not the checkpoint's own maximum, so a step embeds identically whichever encoder reads it. |
| Repeated strings are encoded once | a `(text, dim)` memo. Agent identity is drawn from ~30 distinct strings across every corpus here, so encoding each step's agent separately repeats the same deterministic forward pass thousands of times. Bit-identical; `--no-memo` disables it. Batching is *not* done: padding several strings together changes what attention sees. |
| Features are cached as `.pt` through `rb_shared.cache.StateCache` | atomic tmp-and-rename, a `_manifest.json`, and ids that may not be digits — rather than the vendored `<data_dir>/cache/` directory, which writes into the corpus. |
| `torch.use_deterministic_algorithms` is set inside `set_seed` | kept verbatim. It only fires under CUDA, and `CUBLAS_WORKSPACE_CONFIG` is set *after* it (`main.py:30`), which on some builds makes a cuDNN LSTM refuse to run. Checked here on torch 2.11 + CUDA 13: training on GPU works as-is. Export the variable before launching if a future build disagrees. |
| No `tqdm` | stage progress is printed on a fixed cadence instead, so a log tail stays readable. |
| `--train-device` defaults to the extraction device, i.e. the GPU | the opposite of OAT, whose neural CDE is faster on CPU because it is bound by kernel-launch overhead. StepFinder's core is a BiLSTM, which cuDNN accelerates heavily: measured here at 18.2 optimizer steps/s on an H200 against 0.4 on CPU, a factor of 42. Family B's small partitions make this the difference between minutes and hours per model. |

## Deliberately not ported

| vendored piece | why not |
|---|---|
| `prompts.py` — `TRAJECTORY_REGENERATION_PROMPT` | dead code upstream: nothing imports it, and the code that ran it, parsed it and derived `step_id`/`is_mistake` is not in the repository. The 4,168 generated trajectories ship, and those are what we train on. |
| `prompts.py` — `ALL_AT_ONCE_RANKING_PROMPT` | an LLM baseline with no caller, no trajectory serializer and no output parser. This repo already has `all_at_once`. |
| `main.py`'s CLI and `run_experiment` | replaced by `predict.py`/`sweep.py`. Its defaults survive as `PRESETS["alg"]`, pinned by a test. |
| `feature_construction.py`'s `<data_dir>/cache/` writer | it creates directories inside the corpus, including on the read path (`main.py:37-40`). |
| `main.py`'s evaluation-driven checkpointing | see *Checkpoint selection*. |

## Quirks preserved on purpose

- **No L2 normalization** of either embedding. The only normalization in the
  whole pipeline is `F.normalize` on the agent sequence inside the attention
  bias (`model.py:114`).
- **The width slice is a prefix**, not a projection. Principled for
  Qwen3-Embedding, which is trained so leading coordinates carry the most
  information; arbitrary for a plain decoder, which is worth remembering when
  reading the `qwen3.5-9b` and `deepseek-8b` columns.
- **The `s=3` difference operator is nonstandard.** `H_t - s·H_{t-1} +
  (s-1)·H_{t-s}` is a true second difference only at `s=2`; at `s=3` it is
  `H_t - 3·H_{t-2} + 2·H_{t-3}`. The paper's grid only uses `{1, 2}`, but the
  formula is transcribed rather than the intent, and the parity test covers
  `s=3` too.
- **Right truncation.** No `truncation_side` is set upstream, so a step over
  8192 tokens loses its tail and last-token pooling reads the 8192nd token.
  Counted as `n_truncated_steps` rather than fixed.
- **Empty step contents embed as zero vectors** (25 such steps across the
  corpora), counted as `n_empty_content`.
- **The mask is float32, not bool.** Every consumer in `model.py` was written
  against that.
- **Training length mismatch.** Regenerated training trajectories top out at 41
  steps (Alg) and 50 (HC); the corpora reach 130. The position prior, the LSTM
  and the attention have never seen a sequence that long.

## The 2×2: reduction × selection (added 2026-08-24)

The manuscript's audit of family A raised two objections, and rather than
argue either away, both became an axis. Every combination writes its own
method family, so nothing can pose as the faithful reproduction:

| family | reduction | checkpoint rule |
|---|---|---|
| `stepfinder.s*` | slice (paper) | val — a held-out slice of the training data |
| `stepfinder-pca.s*` | PCA | val |
| `stepfinder-tsel.s*` | slice (paper) | test — WW test accuracy (paper) |
| `stepfinder-pca-tsel.s*` | PCA | test |

### The PCA reduction

The prefix slice is principled only for the Matryoshka-trained
`qwen3-embedding-0.6b`; for a decoder it is an arbitrary 2–5% sample of the
hidden state, and it shrinks as the backbone grows — 128 of 2560 at 4B, 128 of
5120 at 27B — so the scale axis was partly measuring the reduction.
`--reduce pca` encodes at full width (cached in `_sf-featsw/`, ~2–4 KB per
step) and projects onto the top principal directions of the training corpus:
128 for content, 32 for agents, fitted by **uncentered** SVD via the Gram
matrix (`reduce.py`). Uncentered because the model consumes agent vectors only
through `cos(r_i, r_j)` and a mean-then-gate: subtracting the mean would shift
every vector by one offset and rewrite every cosine, while an orthonormal
projection preserves the cosines of everything inside the kept subspace. The
fit sees training features only; the reducer is stored beside them
(`_sf-pca/reducer.pt`) with its kept-variance fractions, which every
prediction file stamps (`pca_content_var_kept`, `pca_agent_var_kept`). Sliced
features are the first coordinates of wide ones, so a slice run finding a wide
cache derives its inputs on CPU, bit-identically — pinned by a test, which in
turn required pinning `PYTHONHASHSEED` in the dummy-encoder subprocesses: the
dummy tokenizer builds token ids from Python's salted `hash`, so two processes
only agree when the salt does. Real tokenizers are unaffected.

### Test selection, made reportable

`--model-selection vendored` reproduces the paper's rule and refuses to write.
`--model-selection test` is the same rule allowed to publish, under three
constraints. First, its own method prefix (`-tsel`), stamped in every file.
Second, one selection set: the WW subset matching the training corpus
(Alg → `ww/algorithm-generated`, HC → `ww/hand-crafted`), never the subset
being scored — so WW numbers are test-selected exactly as the paper's are,
while CE and TE numbers are selected on disjoint data and are, from their own
point of view, honestly selected. Third, provenance: the checkpoint lives
under the WW subset that picked it (`_sf-ckpt/<preset>/test[-pca]/s<seed>/` in
that subset's tree, per GT setting, since the watched features change with the
GT edit), and a non-WW run that cannot find it refuses to train a replacement.

### Selection must measure the deployed function (bug found 2026-08-25)

The first test-selected checkpoints were selected with the training loop's
batched evaluation (batch 16) — where the position bias is divided by the
batch's padded width — and then deployed at batch 1, where Eq. 9 applies the
bias at full strength. On `ww/hand-crafted` (trajectories to 130 steps,
γ = 0.75) the two functions disagree completely: the `qwen3.5-9b` seed-42
checkpoint measured 12.1% at selection and scored **0.0%** deployed,
predicting only early steps. The selection was optimizing a metric nobody
reports. Fix: `fit()` takes `eval_batch_size`, and test-selection evaluates at
batch 1 — the same computation scoring runs — while `vendored` keeps the
batched evaluation it exists to reproduce and `val` keeps the batched holdout
its published checkpoints were selected with (its holdout trajectories top out
at 50 steps, where the mismatch is mild). Every test-selected checkpoint and
prediction from before the fix was deleted and refit. A regression test pins
the invariant: a test-selected checkpoint's recorded accuracy equals the
accuracy recomputed from its own prediction files, exactly.

### Model count

Val-selected checkpoints are shared across subsets and GT settings (they never
see either): 6 encoders × 2 corpora × 5 seeds = 60 per reduction. Test-selected
checkpoints add the GT axis but collapse the subset axis onto WW: 6 × 2 × 2 ×
5 = 120 per reduction. 360 models total, 300 of them new — the earlier
per-subset design would have been 1,440.
