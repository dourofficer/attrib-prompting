# baselines/raffles/configs/

Two kinds of config, both plain YAML read by the drivers — no code changes are
needed to add a model.

| file | read by | purpose |
|---|---|---|
| `<ds>-api.yaml` | `sweep.py` / `predict.py` | what to run: models × subsets, the RAFFLES knobs, `model_specs` |
| `report_<ds>.yaml` | `report.py` | what to score: models, methods, seeds, split ratios |

`<ds>` is `ww`, `correct-error`, `traceelephant` or `tracertraj`.

**Only closed-source inference configs are shipped** (`gpt-4o`, `gpt-5` — the
`-api.yaml` files), because that is what this repo currently runs. Local-model
configs are a drop-in; see below.

## Key config keys

```yaml
models:  [gpt-4o, gpt-5]
subsets: [algorithm-generated, hand-crafted]

data_dir:     data/ww           # corpus
outputs_root: outputs/ww        # predictions (with-GT; the default without-GT
                                # setting mirrors to outputs-nogt/ww)
gt: without                     # this baseline's default = the paper setting

max_iters: 2                    # K — extra iterations after the first pass
threshold: 350                  # early-stop confidence (of 400)
method_dir: null                # output dir override, e.g. raffles.k5
model_specs:  {...}             # backend + params per model name
```

`max_iters` and `threshold` are the paper's values. To reproduce the paper's
Who&When SOTA row (K=5) next to the default run, override both the knob and the
output directory so the runs never collide:

```bash
DATASET=ww MODEL=gpt-4o EXTRA_SET="--set max_iters=5 --set method_dir=raffles.k5" \
    bash scripts/raffles/run.sh
```

`params` in a spec go to the API verbatim. Every call returns one JSON object,
but the Judge quotes the log in three rationale fields, so give it a generous
budget.

## Adding a closed-source model

Add a spec block and put the name in `models:`. Any OpenAI-compatible provider
works via `base_url`:

```yaml
models: [gpt-4o, gpt-5, my-model]
model_specs:
  my-model:
    backend: openai
    model: provider/model-id
    base_url: https://api.provider.com/v1     # optional
    api_key_env: PROVIDER_API_KEY             # optional (default OPENAI_API_KEY)
    concurrency: 8                            # optional
    params: {max_tokens: 8192}
```

Non-reasoning models should carry `temperature: 0.0` — the paper decodes
greedily, and without it the API's own default (1.0) makes an iterative chain
non-reproducible.

Reasoning models: use `max_completion_tokens`, omit `temperature`/`top_p` (they
reject a non-default temperature), and give a larger cap than you would a plain
model, since reasoning tokens come out of the same budget — see the `gpt-5`
spec. `strip_think` removes any reasoning block before the JSON parsers run.

## Adding local (vLLM) models

Repo convention (same as `baselines/prompting/configs/`): **`<ds>.yaml` is the
local-vLLM config, `<ds>-api.yaml` the closed-source one**. Create `<ds>.yaml`
and `scripts/raffles/run.sh` picks it up automatically — it prefers whichever
config declares `MODEL`, and defaults to `<ds>.yaml` when both exist. Never mix
backends in one config.

Template (Who&When; adjust `subsets`, `data_dir`, `outputs_root` for the other
datasets):

```yaml
# RAFFLES inference config — Who&When (ww), local vLLM models.
models:  [qwen3.5-9b, deepseek-8b]
subsets: [algorithm-generated, hand-crafted]

data_dir:     data/ww
outputs_root: outputs/ww
gt: without

max_iters: 2
threshold: 350

# Sampling — greedy, as the paper ("a greedy search is used").
dtype:          bfloat16
seed:           0
temperature:    0.0
top_p:          1.0
gen_max_tokens: 2048
enable_thinking: false

# The Judge sees the full history plus prior-iteration feedback; the longest
# trajectories in this repo are ~98k tokens on their own.
max_model_len:          131072
truncate_prompt_tokens: null
gpu_memory_utilization: 0.90
tensor_parallel_size:   1

start_idx: 0
end_idx:   null

model_specs:
  qwen3.5-9b:
    backend: vllm
    model_path: ../hub/Qwen/Qwen3.5-9B
  deepseek-8b:
    backend: vllm
    model_path: ../hub/deepseek-ai/DeepSeek-R1-Distill-Llama-8B
    # Corrected tokenizer (see baselines/prompting/configs/ww.yaml).
    tokenizer: baselines/prompting/tokenizers/deepseek-8b
    # R1-Distill always emits <think>; 2048 would truncate the actual answer.
    gen_max_tokens: 8192
```

A vLLM spec may override any top-level sampling/runtime knob (as `deepseek-8b`
does); anything unset falls back to the top-level value.

## Report configs

`report_<ds>.yaml` lists the models and methods that go into the tables; add a
model name there once its runs finish. `gt: without` selects the
`outputs-nogt/` tree (the default for this baseline); `gt_in_prompt` labels the
with-GT tree only. Seeds are 1–20 for ww/traceelephant/tracertraj, 1–3 for correct-error.
