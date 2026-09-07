# StepFinder configs

Two kinds of file per dataset, and they answer different questions.

- **`<ds>.yaml` — what to run.** Read by `stepfinder.sweep`, which turns it into
  one `stepfinder.predict` command per encoder × subset.
- **`report_<ds>.yaml` / `report_<ds>_incorpus.yaml` — what to score.** Read by
  `stepfinder.report`, which is the shared report every baseline in this repo
  uses. One per training family, because the two write different method
  directories.

## `<ds>.yaml`

| key | what it does |
|---|---|
| `models` | encoders to run; each needs a `model_specs` entry |
| `subsets` | subset directories under `data_dir` |
| `data_dir` | corpus root, e.g. `data/ww` |
| `outputs_root` | always the **`-gt`** spelling; `gt: without` mirrors it to `-nogt` |
| `gt` | `without` (the faithful default) or `with` |
| `protocols` | `[regen]`, `[in-corpus]`, or both |
| `seeds` | training seeds for the vendored protocol → `stepfinder.s<seed>` |
| `eval_seeds` | split seeds for the in-corpus protocol → `stepfinder.e<seed>`; **must equal the report config's `seeds`** |
| `splits` | the evaluation protocol's partition, reproduced to find the training 30% |
| `default_train_set`, `train_set_overrides` | which vendored corpus a subset trains on |
| `default_preset`, `preset_overrides` | which row of the paper's Table 1 it uses |
| `train_data_root` | where the vendored training corpora live |
| `model_selection` | `val` (holds out training questions) or `vendored` (parity check only; refuses to write predictions) |
| `val_frac`, `val_min_tasks` | the holdout size, and the floor below which early stopping is skipped |
| `agent_normalize` | `standardize` (default) or `raw` — see `../IMPLEMENTATION.md` |
| `epochs`, `batch_size`, `patience`, `top_k` | training and decoding knobs |

### The two kinds of seed

`seeds` and `eval_seeds` mean unrelated things and are easy to confuse.

- **`seeds`** are *training* seeds. Family A trains five models on the same
  data with different initializations, and each becomes its own method
  directory so the spread across training runs is visible in the report rather
  than averaged away. They have nothing to do with the evaluation splits.
- **`eval_seeds`** are the *split* seeds the evaluation protocol already uses
  (1–20 for `ww` and `traceelephant`, 1–3 for `correct-error`). Family B trains
  one model per split seed, on that seed's training partition, and scores only
  that seed's val and test ids.

Because of that, `eval_seeds` here must match `seeds` in the matching
`report_<ds>_incorpus.yaml`. The shipped configs agree, and
`tests/test_stepfinder_pipeline.py::test_shipped_configs_parse_and_agree_with_the_corpus`
keeps them agreeing.

### Which training corpus a subset gets

Two corpora ship with the paper, built from two different agent systems:
Algorithm-Generated from **CaptainAgent** (an AutoGen team of `*_Expert` roles),
Hand-Crafted from **Magentic-One**. The agent embedding is a text embedding of
the agent's name, so the pairing matters — it decides whether the attention bias
sees a familiar vocabulary.

Pair each subset with the corpus built from the same system. `ww/hand-crafted`
and `traceelephant/magentic` are Magentic-One; `ww/algorithm-generated` and
`traceelephant/captain` are CaptainAgent. `correct-error` matches neither and
falls back to Hand-Crafted. Each pairing brings its own Table 1 preset:

```yaml
# ww.yaml
default_train_set: hand-crafted
train_set_overrides: {algorithm-generated: algorithm-generated}
default_preset: hc
preset_overrides:    {algorithm-generated: alg}

# traceelephant.yaml
default_train_set: hand-crafted
train_set_overrides: {captain: algorithm-generated}
default_preset: hc
preset_overrides:    {captain: alg}
```

`protocol.py:48-54` holds the same mapping for runs that pass no config.

Changing this changes what leaks. Under the default, `ww` has zero task overlap
with its training corpus; pair `ww/algorithm-generated` with Hand-Crafted
instead and 69 of its 126 tasks are suddenly in training. `train_task_overlap`
is always recomputed from the training set actually used, so a swap shows up in
the output files rather than silently inflating a number.

## Adding an encoder

Add a `model_specs` entry and list it under `models`:

```yaml
model_specs:
  my-encoder:
    model_path: ../hub/org/My-Encoder     # required
    tokenizer: path/to/tokenizer          # optional override
    dtype: fp16                           # the vendored setting
    device: cuda                          # optional
```

Any local checkpoint works; the method needs embeddings, which a chat endpoint
does not expose. The loader unwraps a vision-language wrapper automatically. A
model whose leading embedding coordinates are not the informative ones will be
handicapped by the vendored prefix slice — see the caveat in
[`../README.md`](../README.md#encoders).

The encoder name becomes the `<model>` path component, and it is deliberately
shared with OAT's so the two representation-based methods land in comparable
rows of the same tree.

## `report_<ds>.yaml`

Standard shared-report keys — `models`, `subsets`, `methods`, `data_dir`,
`pred_root`, `out_root`, `gt`, `splits`, `seeds`. Two things to keep in step
with the run config: `methods` must list the method directories the run
actually wrote (`stepfinder.s42`…`s46`, or `stepfinder.e1`…), and `pred_root`
must equal the run config's `outputs_root`.

The `_incorpus` variant additionally needs `--diagonal` to be useful:

```bash
python -m stepfinder.report --config .../report_ww_incorpus.yaml --diagonal
```

Without it you get the raw table, in which nineteen of every twenty cells are
the report scoring a method on ids it never predicted. `--check-only` reporting
`PARTIAL` for that family is expected.
