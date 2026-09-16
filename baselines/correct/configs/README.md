# baselines/correct/configs/

Two kinds of config, both plain YAML read by the drivers — no code changes are
needed to add a model.

| file | read by | purpose |
|---|---|---|
| `<ds>-api.yaml` | `sweep.py` / `predict.py` / `schemagen.py` | what to run: models × subsets × methods, the 3-stage pipeline knobs, `model_specs` |
| `report_<ds>.yaml` | `report.py` | what to score: models, methods, seeds, split ratios |

`<ds>` is `ww`, `correct-error`, `traceelephant` or `tracertraj`.

**Only closed-source inference configs are shipped** (`gpt-4o`, `gpt-5` — the
`-api.yaml` files), because that is what this repo currently runs. Local-model
configs are a drop-in; see below.

## Key config keys

```yaml
models:  [gpt-4o, gpt-5]        # detectors (stage 3)
subsets: [algorithm-generated, hand-crafted]
methods: [correct, correct_baseline]      # schema-guided / k=0 baseline

data_dir:     data/ww           # corpus
outputs_root: outputs/ww        # predictions (with-GT; without-GT mirrors to
                                # outputs-nogt/ww)
artifacts_root: artifacts/ww    # stage-1/2 artifacts, GT-independent (default:
                                # outputs_root with its root swapped to artifacts/)
gt: without                     # this baseline's default = the paper setting

schema_model: gpt-4o            # stage 1: who distills the schemata
embed_model:  ../hub/BAAI/bge-m3   # stage 2: similarity encoder
num_schemata: 5                 # stage 3: top-k; scalar or a per-subset map
schema_gen:   {params: {max_tokens: 4096}}   # stage-1 overlay (see below)
model_specs:  {...}             # backend + params per model name
```

`num_schemata` follows the paper (§A.3) — ww algorithm-generated 1, ww
hand-crafted 10, CORRECT-Error 5; TraceElephant is not in the paper and
defaults to 10. `schema_gen` overrides the top-level sampling knobs for stage 1
only; its `params` key *replaces* the spec's `params` there (the vendored cloud
generator sends `max_tokens` only).

## Adding a closed-source model

Add a spec block and put the name in `models:`. `params` are sent to the API
verbatim and recorded in `_run.json`; any OpenAI-compatible provider works via
`base_url`:

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

Reasoning models: use `max_completion_tokens` and omit `temperature`/`top_p`
(see the `gpt-5` spec).

## Adding local (vLLM) models

Repo convention (same as `baselines/prompting/configs/`): **`<ds>.yaml` is the
local-vLLM config, `<ds>-api.yaml` the closed-source one**. Create
`<ds>.yaml` and `scripts/correct/run.sh` picks it up automatically — it prefers
whichever config declares `MODEL`, and defaults to `<ds>.yaml` when both exist.
Never mix backends in one config.

Template (Who&When; adjust `subsets`, `data_dir`, `outputs_root`,
`num_schemata` for the other datasets):

```yaml
# CORRECT inference config — Who&When (ww), local vLLM models.
models:  [qwen3.5-9b, deepseek-8b]
subsets: [algorithm-generated, hand-crafted]
methods: [correct, correct_baseline]

data_dir:       data/ww
outputs_root:   outputs/ww
artifacts_root: artifacts/ww
gt: without

schema_model: qwen3.5-9b            # or gpt-4o, to reuse schemata already generated
embed_model:  ../hub/BAAI/bge-m3
num_schemata:
  algorithm-generated: 1
  hand-crafted: 10

# Stage-1 sampling = the vendored schema generator's SamplingParams.
schema_gen: {temperature: 0.7, top_p: 0.95, gen_max_tokens: 1024}

# Detection sampling — greedy, as the vendored local CORRECT inference.
dtype:          bfloat16
seed:           0
temperature:    0.0
top_p:          1.0
gen_max_tokens: 1024
enable_thinking: false

# Longest full-history prompt across all datasets is ~98k tokens (p99 ~29k);
# retrieved schemata add ~1k each.
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
    # R1-Distill always emits <think>; 1024 would truncate the actual answer.
    gen_max_tokens: 8192
```

A vLLM spec may override any top-level sampling/runtime knob (as `deepseek-8b`
does); anything unset falls back to the top-level value.

**Schemata are shared across configs.** Stage 1 is keyed by `schema_model`, not
by the detector, and lands in `artifacts/`, not in any output tree — so a local
detector can consume GPT-4o schemata: keep `schema_model: gpt-4o` and the same
`artifacts_root` in the local config, and the sweep skips stage 1 as already
complete. Same for the similarities, which depend only on `embed_model`. That
is the paper's design — one strong generator, many detectors.

## Report configs

`report_<ds>.yaml` lists the models and methods that go into the tables; add a
model name there once its runs finish. `gt: without` selects the `outputs-nogt/`
tree (the default for this baseline); `gt_in_prompt` labels the with-GT tree
only. Seeds are 1–20 for ww/traceelephant/tracertraj, 1–3 for correct-error.
