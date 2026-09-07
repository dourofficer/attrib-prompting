# OAT configs

Two files per dataset. `<ds>.yaml` says what to run; `report_<ds>.yaml` says
what to score. Both are plain YAML and every key is overridable from the
command line with `--set key=value`.

There is no `-api.yaml` variant here, and there cannot be. OAT reads a model's
hidden states, and a chat endpoint does not expose them — every extractor is a
local checkpoint.

## `<ds>.yaml` — what to run

```yaml
models:  [qwen3.5-9b, deepseek-8b]   # extractor labels; the <model> path component
subsets: [hand-crafted, algorithm-generated]
data_dir: data/ww
outputs_root: outputs-rb-gt/ww       # --gt without mirrors this to outputs-rb-nogt/ww

gt: without                          # the vendored setting: no answer in the text
seeds: [42, 43, 44, 45, 46]          # one method directory each: oat.s42 … oat.s46

layer: -1                            # which layer's hidden states to read
aggregation: mean                    # how to pool a step's tokens
latent_dim: 64                       # PCA target dimension

top_k: 3                             # size of the top-k detection set
alpha: 0.2                           # conformal miscoverage rate

train_dir: vendored/OAT/dataset/MCP-atlas/Qwen3.5-27B
epochs: 300
patience: 20

model_specs:
  qwen3.5-9b:
    model_path: ../hub/Qwen/Qwen3.5-9B
    dtype: bf16
```

`model_specs` describes a checkpoint to *read* rather than a server to
generate from, so it takes only:

| key | meaning |
|---|---|
| `model_path` | local checkpoint directory (required), or `dummy` for a keyless CPU run |
| `tokenizer` | tokenizer override, when the checkpoint's own is wrong |
| `dtype` | `bf16` (default), `fp16` or `fp32` |
| `device` | defaults to `cuda` when one is available |
| `attn_implementation` | passed through to `from_pretrained` |

There is nothing to configure about sampling because there is no sampling.

## Adding an extractor

Add an entry and name it in `models`. To run the paper's own extractor, fetch
`Qwen/Qwen3.5-27B` into `../hub/` and add:

```yaml
  qwen3.5-27b:
    model_path: ../hub/Qwen/Qwen3.5-27B
    dtype: bf16
```

Each extractor gets its own PCA, its own trained models and its own `<model>`
directory, so adding one never disturbs another's results. The first run for a
new extractor pays for training once; every subset after that reuses it.

A checkpoint whose tokenizer needs correcting takes the same treatment as the
prompting baselines' — see `deepseek-8b`, which points at
`baselines/prompting/tokenizers/deepseek-8b`.

## `report_<ds>.yaml` — what to score

```yaml
models:  [qwen3.5-9b, deepseek-8b]
subsets: [hand-crafted, algorithm-generated]
methods: [oat.s42, oat.s43, oat.s44, oat.s45, oat.s46]
data_dir: data/ww
pred_root: outputs-rb-gt/ww
out_root:  outputs-rb-gt/ww/reports/oat
gt: without
splits: {train: 0.3, val: 0.2, test: 0.5}
seeds: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
```

Note the two kinds of seed, which are unrelated and easy to confuse:

- `seeds` **here** are *data-split* seeds — the repo's evaluation protocol,
  shared with every other baseline. Each one produces a different val/test
  partition of the corpus.
- `seeds` in `<ds>.yaml` are *training* seeds, which produce different models.
  They appear in this file as the `methods` list.

`gt: without` sends both `pred_root` and `out_root` through the without-GT
mapping, so the report reads `outputs-rb-nogt/` and writes its tables there.
