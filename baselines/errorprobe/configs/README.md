# baselines/errorprobe/configs/

Two kinds of config, both plain YAML read by the drivers — no code changes are
needed to add a model.

| file | read by | purpose |
|---|---|---|
| `<ds>-api.yaml` | `sweep.py` / `predict.py` | what to run: models × subsets × modes, `model_specs` |
| `report_<ds>.yaml` | `report.py` | what to score: models, methods, seeds, split ratios |

`<ds>` is `ww`, `correct-error` or `traceelephant`.

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
gt: without                     # this baseline's default = the vendored setting

modes: [truncated, backward]    # which vendored configuration(s) to run
method_dir: null                # output dir override (only with a single mode)
model_specs:  {...}             # backend + params per model name
```

`modes` picks the vendored configuration: `truncated` is the shipped default
(Analyzer over the last 15 turns — 2 calls per trajectory, writes under
`errorprobe/`), `backward` is the config-gated backward-tracing walk (full
trace, roughly 2–3 calls per examined turn plus memory maintenance, writes
under `errorprobe_bt/`). The two modes never collide — each has its own method
directory.

`params` in a spec go to the API verbatim. The shipped
`{max_tokens: 4000, temperature: 0.7}` mirrors the vendored `config.yaml`
model block; the vendored code additionally overrides tokens/temperature per
call (Analyzer 2000, Verifier 1500 at 0.3, tracing calls 200–500), which the
shared backend cannot vary — a documented deviation (`IMPLEMENTATION.md`).

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
    params: {max_tokens: 4000, temperature: 0.7}
```

Reasoning models: use `max_completion_tokens`, omit `temperature`/`top_p`
(they reject a non-default temperature), and give a larger cap than you would
a plain model, since reasoning tokens come out of the same budget — see the
`gpt-5` spec. `strip_think` removes any reasoning block before the JSON
parsers run.

## Adding local (vLLM) models

Repo convention (same as `baselines/prompting/configs/`): **`<ds>.yaml` is the
local-vLLM config, `<ds>-api.yaml` the closed-source one**. Create `<ds>.yaml`
and `scripts/errorprobe/run.sh` picks it up automatically — it prefers
whichever config declares `MODEL`, and defaults to `<ds>.yaml` when both
exist. Never mix backends in one config.

Template (Who&When; adjust `subsets`, `data_dir`, `outputs_root` for the other
datasets):

```yaml
# ErrorProbe inference config — Who&When (ww), local vLLM models.
models:  [qwen3.5-9b]
subsets: [algorithm-generated, hand-crafted]

data_dir:     data/ww
outputs_root: outputs/ww
gt: without

modes: [truncated, backward]

# Sampling — the vendored config decodes at temperature 0.7.
dtype:          bfloat16
seed:           0
temperature:    0.7
top_p:          1.0
gen_max_tokens: 2048
enable_thinking: false

# The truncated mode caps its own context (15 turns × 500 chars); the
# backward mode's prompts are similarly bounded (150-800 chars per quoted
# turn), so a moderate window suffices.
max_model_len:          32768
truncate_prompt_tokens: null
gpu_memory_utilization: 0.90
tensor_parallel_size:   1

start_idx: 0
end_idx:   null

model_specs:
  qwen3.5-9b:
    backend: vllm
    model_path: ../hub/Qwen/Qwen3.5-9B
```

A vLLM spec may override any top-level sampling/runtime knob; anything unset
falls back to the top-level value.

## Report configs

`report_<ds>.yaml` lists the models and methods that go into the tables; add a
model name there once its runs finish. `methods: [errorprobe, errorprobe_bt]`
scores both modes side by side; drop one if you only ran the other. `gt:
without` selects the `outputs-nogt/` tree (the default for this baseline);
`gt_in_prompt` labels the with-GT tree only. Seeds are 1–20 for
ww/traceelephant, 1–3 for correct-error.
