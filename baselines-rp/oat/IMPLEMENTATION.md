# OAT — implementation notes

What this package changes from `vendored/OAT/`, and why. The rule is the repo's
usual one: the vendored code is the reference, and anything that differs is
either forced by this repo's data or belongs to the uniform shell every
baseline shares (I/O, resume, evaluation). Nothing in the method itself was
"improved".

Read [`README.md`](README.md) first for what the method does.

## What is verbatim

| here | vendored source | what it is |
|---|---|---|
| `model.py` | `models/oat.py` | the neural CDE, its gate, the ODE bridge, the loss |
| `train.py` — PCA, z-norm, CORAL, `get_time_points` | `data_pipeline.py:317-409` | latent projection and domain alignment |
| `train.py` — `train_epoch`, `validate`, `train_model`, `split_train_val` | `train.py:18-107, 241-250` | the training loop |
| `train.py` — `_topk_detection`, `_normalize_scores`, `conformal_quantile_threshold`, `collect_calibration_scores`, `_conformal_detection`, `_scorable_scores_and_indices` | `train.py:110-186` | the three decoders |
| `serialize.py` — `serialize_trajectory` | `data_pipeline.py:33-57`, hand-crafted branch | the per-step document format |
| `serialize.py` — `serialize_mcp_atlas_trajectory`, `_serialize_content`, `_parse_scope`, `load_mcp_atlas_records` | `data_pipeline.py:60-116, 119-207` | the training corpus |
| `serialize.py` — `build_input_text` | `extract_states.py:118-126` | the question prefix and span shift |
| `extract.py` — `get_step_token_ranges`, `get_step_last_token_indices`, the pooling loop | `data_pipeline.py:220-252`, `extract_states.py:55-115` | tokens to step vectors |

`tests/test_oat_pipeline.py` proves it rather than asserting it: it imports the
vendored modules and compares. Text and char bounds byte-for-byte; the model's
loss and per-step scores bit-for-bit on shared weights; PCA, CORAL,
normalization, the conformal threshold, MAD scaling and all three decoders
numerically; and `rb_metrics` against `evaluate.py`. The vendored constants in
`model.py` and `train.py` are checked against `vendored/OAT/config.py` at test
time, so a restated number cannot drift.

## Forced by this repo's data

### Agent identity, and which serializer branch runs

The vendored loader has two branches. Who&When *hand-crafted* records carry
agent identity in `role`; *algorithm-generated* records carry it in `name`, and
the vendored code reads `name` for the step header and the model-step filter.

No corpus in `data/` has a `name` field. GUIDE.md fixes agent identity at
`history[t]["role"]` for the whole repo, and the corpora follow it — Who&When
algorithm-generated records that upstream stored as `name: Excel_Expert` appear
here as `role: Excel_Expert`. So every corpus goes through the **hand-crafted
branch**, whose header reads exactly the field this repo populates. For
Who&When hand-crafted the bytes are the vendored bytes; for the others it is
the vendored format applied to the field that holds the same information.

`agent_of` keeps the vendored collapse of `Orchestrator (thought)` and
`Orchestrator (-> WebSurfer)` to `Orchestrator`.

### The model-step filter

| | vendored rule | here |
|---|---|---|
| Who&When hand-crafted | `role != "human"` | same |
| Who&When algorithm-generated | `name != "Computer_terminal"` | `role != "Computer_terminal"` — the same turns, via this repo's identity field |
| CORRECT-Error | — | `role != "user"`, the corpus's name for the human |
| TraceElephant | — | `role != "ComputerTerminal"` |

So `NON_MODEL_ROLES = {human, user, Computer_terminal, ComputerTerminal}`.

Roles that name a *tool* rather than the environment — `bash`,
`str_replace_editor`, `submit` in `traceelephant/swe` — are **kept**. In that
corpus the tool call is the agent's action and gold labels land on those turns;
dropping them would make 1,542 of the subset's 1,551 turns unscoreable — very
nearly all of it — and their labels unreachable.

### Trajectories whose gold step is filtered out

The vendored loader **discards** any failure whose annotated step is not a
model step (`data_pipeline.py:303-305`). This package keeps it and writes a
prediction anyway, flagged `gold_on_filtered_step: true`.

The reason is the shared report: it counts a missing prediction as wrong, over
a universe fixed by the corpus file list. Dropping a trajectory would not
excuse it from the denominator; it would only remove the record of what
happened. Roughly 45 trajectories are affected (2 in Who&When
algorithm-generated, 43 in CORRECT-Error), and for those `step@1` is
unreachable by construction. The flag makes that countable instead of invisible.

### The gold step outside the history

Two TraceElephant/magentic records carry a `mistake_step` beyond the end of
their history. Nothing special is done: the prediction is written, and it
cannot match. Same reasoning as above.

`gold_on_filtered_step` covers both cases, since what it actually records is
"the annotated step is not among the steps that received a score". That is the
useful question — whether `step@1` was reachable at all — and it is true both
for a gold label on a non-agent turn and for one past the end of the log.

## Adaptation glue

### Step and agent predictions

| | |
|---|---|
| `predicted_step` | the vendored `argmax` over scorable rows, mapped back through `step_indices` to the absolute, 0-based history index. No shift. |
| `predicted_agent` | `history[predicted_step]["role"]`, raw. The report's `standardize_role` handles the Magentic variants. |

OAT predicts no agent of its own — the vendored code has no agent metric at
all. Deriving the agent from the predicted step is the only honest reading of
"which agent does this method blame", and it is what makes `agent@1` reportable
for a representation-based baseline. It also means `agent@1` can only be right
when `step@1` lands on a turn by the right agent, which is a real property of
the method, not a scoring artefact.

### The `--gt with` extension

The vendored document never carries the task answer, so `--gt without` is the
default and the faithful setting. `--gt with` makes the question
`f"{question}\nThe Answer for the problem is: {ground_truth}"` — the same
sentence the prompting baselines interpolate
(`baselines/prompting/methods.py:127`), placed the same way ErrorProbe places
it (`baselines/errorprobe/prompts.py:80`) — inside the vendored
`Question: {q} Trajectory: ` wrapper. The answer therefore sits in the question
row's own span and in the left context of every step.

This is an extension, not a vendored behaviour. It exists so the two GT
settings can be compared for this baseline as they are for the others.

The training corpus is untouched: MCP-Atlas records have no answer field, so a
with-GT run reuses the without-GT checkpoint and only re-extracts the test side.

### CORAL is fitted per subset, over the whole subset

The vendored `ood` command aligns one whole test *dataset* onto the training
successes, once, before the seed loop (`run_pipeline.py:164-170`). The
analogous unit here is a subset, since that is what one invocation covers.

Fitting is over every cached trajectory of the subset, not over the slice a
particular invocation was asked to score, so `--start_idx/--end_idx` do not
move the alignment. They *can* move it indirectly: a slice that also extracts
leaves the cache incomplete, and CORAL is only as complete as the cache. The
run prints the shortfall and records `coral_n_trajectories` in every output
file, so a partial alignment is countable rather than silent; the recommended
split-across-machines flow is to run `states-test` in slices and `score` once.
CORAL uses no labels, so it does not touch the evaluation protocol.

### Seeds as method directories

Vendored: five seeds, metrics averaged, and `predictions.json` holding **only
the last seed** (`run_pipeline.py:132-140`). Here: every seed's predictions are
kept, in `oat.s42` … `oat.s46`, and the report lists all five. The training
variance the vendored code reports as a standard deviation is visible here as
five rows.

### Where the artifacts live

Cached states and checkpoints sit inside the output roots, not under
`artifacts/`:

```
outputs-rb-<gt|nogt>/<ds>/<subset>/<model>/oat.s<seed>/<id>.json   predictions
outputs-rb-<gt|nogt>/<ds>/<subset>/<model>/_oat-states/<id>.pt     test vectors
outputs-rb-nogt/mcp-atlas/train/<model>/_oat-states/<id>.pt        training vectors
outputs-rb-nogt/mcp-atlas/train/<model>/_oat-ckpt/                 PCA + per-seed models
```

The leading underscore keeps them out of the report's way: it reads only the
method directories named in its config, and `OutputWriter.done_ids` counts only
digit stems. Test vectors are GT-dependent and so exist once per root; training
vectors and checkpoints are not, and live only in the `-nogt` tree.

### `nogt_root` learned one more mapping

`baselines/shared/common.py:nogt_root` mapped `outputs/…` to `outputs-nogt/…`
and rejected everything else. It now also maps `outputs-<family>-gt` to
`outputs-<family>-nogt`, which is what routes this baseline's sweep and report
to the right tree. Every other root still raises, and the existing test that
pins the rejection still passes.

## Infrastructure-level deviations

These belong to the shell, not the method. None changes a number.

| deviation | why |
|---|---|
| **The LM head never runs.** The vendored code loads `AutoModelForCausalLM`; we load the decoder underneath it. | The head's logits on an 81k-token log with a 248k-token vocabulary are ~40 GB, and they are discarded. Hidden states are identical. |
| **Only the requested layers are stored**, `float16`, default `[-1]`. Vendored stores all 65 layers in `float32`. | Tens of gigabytes versus ~300 MB. `layers=None` restores the vendored behaviour and is what the parity test uses. |
| **`layers=(-1,)` reads `last_hidden_state`** instead of asking for the full stack. | Verified bit-identical to `hidden_states[-1]` — both are the final norm's output. Saves ~22 GB on the longest logs; it was the difference between OOM and 31 GiB peak on a shared GPU. |
| **Vision-language checkpoints are unwrapped.** `Qwen3.5-9B` is a `Qwen3_5ForConditionalGeneration`; we load it with `AutoModelForImageTextToText` and take the text decoder. | `AutoModelForCausalLM` has no conversion mapping for that layout and would silently leave weights uninitialized. A load test asserts no initialization warning. |
| **`dtype=` replaces `torch_dtype=`; no `device_map="auto"`.** | `torch_dtype` is deprecated in transformers 5.x; a single GPU holds every extractor, and `device_map` needs `accelerate`. |
| **The truncation cap is `min(262144, max_position_embeddings)`.** | DeepSeek-8B tops out at 131,072. The longest real document is 81k tokens, so this never binds. |
| **`train_model` prints every 10th epoch** (`log_every`), not every epoch. | 300 epochs × 5 seeds × 2 extractors buries every other line. The arithmetic is untouched. |
| **`score_trajectory` scores one trajectory at a time.** | Lifted out of the vendored `score_failure_trajectories` loop so predictions can be written and resumed per file. The batch function is kept, and the parity test drives both. |
| **Extraction bypasses `get_backend()`.** | The shared backends speak chat completions; hidden states are not on that interface. Documented here rather than bent into it. |

## Deliberately not ported

| vendored piece | why not |
|---|---|
| `baselines.py` — random-step, first-step, all-at-once prompting | This repo already has an all-at-once baseline with its own vendored prompt; two would confuse the comparison. |
| `evaluate.py` as the scoring path | The shared report is the repo's non-forkable rule. Its metrics are reimplemented in `rb_shared/rb_metrics.py` as a *second* view over the same files, and tested against `evaluate.py`. |
| `run_pipeline.py` CLI, `RunManager` | Replaced by `predict.py`/`sweep.py`, which resume per file. |
| `extract_states.py` `main()` / `--force` | Replaced by the stage machinery; `--overwrite` is the repo's spelling of `--force`. |

## Quirks preserved on purpose

Each of these looks like something to fix. None was.

- **Learning rate 1e-4, not the paper's 4e-5.** `config.py:48` says `1e-4` and
  Table 6 says `4e-5`. The code is the reference, so `1e-4` it is; the
  discrepancy is noted here so a future reader does not "correct" one to the
  other silently.
- **The question is row 0 with step index −1.** It seeds the hidden state and
  never receives an anomaly score. Kept, including its exclusion from argmax.
- **Who&When hand-crafted repeats the question**, once in the `Question:`
  prefix and again as the `human` turn at step 0. That is what the vendored
  code feeds the model.
- **The document contains steps after the mistake.** OAT reads the whole
  trajectory, not a prefix.
- **The validation set is also the calibration set.** The 20% held out for early
  stopping is the same 20% the conformal threshold is computed on
  (`run_pipeline.py:88, 99-103`). Sharing them makes the threshold slightly
  optimistic; it is the vendored behaviour.
- **Conformal falls back to the top step** when nothing clears the threshold
  (`CONFORMAL_MIN_DETECTIONS = 1`), so the set is never empty.
- **MAD normalization before thresholding.** Scores are centred on their median
  and scaled by median absolute deviation, per trajectory, before comparison to
  the threshold — robust to the single huge score a failure often carries.
- **PCA is fitted on successes only**, and the z-normalization statistics are
  the successes' own. Failures are projected through a basis that never saw
  them, which is the point of a one-class method.
- **`_topk_detection` re-sorts by step index**, so `topk_steps` is ordered by
  position, not by score. Read `scores` for the ranking.
