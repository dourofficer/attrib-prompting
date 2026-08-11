# baselines/chief/configs/

Two kinds of config, both plain YAML read by the drivers — no code changes are
needed to add a model.

| file | read by | purpose |
|---|---|---|
| `<ds>-api.yaml` | `sweep.py` / `predict.py` / `ragprep.py` | what to run: models × subsets, the RAG knobs, `model_specs` |
| `report_<ds>.yaml` | `report.py` | what to score: models, methods, seeds, split ratios |

`<ds>` is `ww`, `correct-error` or `traceelephant`.

**Only closed-source inference configs are shipped** (`gpt-4o`, `gpt-5` — the
`-api.yaml` files), because that is what this repo currently runs. Local-model
configs are a drop-in; see below.

## Key config keys

```yaml
models:  [gpt-4o, gpt-5]        # detectors (stage 2)
subsets: [algorithm-generated, hand-crafted]

data_dir:     data/ww           # corpus
outputs_root: outputs/ww        # predictions (with-GT; without-GT mirrors to
                                # outputs-nogt/ww)
artifacts_root: artifacts/ww    # stage-1 artifacts, GT-independent (default:
                                # outputs_root with its root swapped to artifacts/)
gt: with                        # this baseline's default = the vendored setting

rag:                            # stage 1: retrieved decomposition exemplars
  root:  vendored/CHIEF/rag     #   committed FAISS indices + knowledge bases
  kb:    [gaia, assistantbench] #   which indices to search (empty list = no RAG)
  top_k: 2                      #   vendored; the [1:top_k] slice yields 1 exemplar
  embed_model: sentence-transformers/all-MiniLM-L6-v2
model_specs:  {...}             # backend + params per model name
```

Every dataset keeps RAG on. The knowledge base is GAIA + AssistantBench
regardless of corpus, because CHIEF uses the exemplar as a *decomposition
template*, not as domain knowledge — and keeping it on is what makes each prompt
byte-identical to the vendored one. Setting `kb: []` disables retrieval, which
changes the stage-1 prompt (a documented deviation, see `../IMPLEMENTATION.md`).

`params` in a spec go to the API verbatim. CHIEF's stages 5 and 6 inline the
whole causal graph on top of the trajectory, so give them a generous budget.

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

Reasoning models: use `max_completion_tokens` and omit `temperature`/`top_p`
(see the `gpt-5` spec). CHIEF's stage parsers are strict regexes over plain
text, and `strip_think` removes any reasoning block before they run.

## Adding local (vLLM) models

Repo convention (same as `baselines/prompting/configs/`): **`<ds>.yaml` is the
local-vLLM config, `<ds>-api.yaml` the closed-source one**. Create `<ds>.yaml`
and `scripts/chief/run.sh` picks it up automatically — it prefers whichever
config declares `MODEL`, and defaults to `<ds>.yaml` when both exist. Never mix
backends in one config.

Template (Who&When; adjust `subsets`, `data_dir`, `outputs_root` for the other
datasets):

```yaml
# CHIEF inference config — Who&When (ww), local vLLM models.
models:  [qwen3.5-9b, deepseek-8b]
subsets: [algorithm-generated, hand-crafted]

data_dir:       data/ww
outputs_root:   outputs/ww
artifacts_root: artifacts/ww
gt: with

rag:
  root:  vendored/CHIEF/rag
  kb:    [gaia, assistantbench]
  top_k: 2
  embed_model: sentence-transformers/all-MiniLM-L6-v2

# Detection sampling — greedy, as the vendored call_model (temperature 0).
dtype:          bfloat16
seed:           0
temperature:    0.0
top_p:          1.0
gen_max_tokens: 2048
enable_thinking: false

# Stages 5-6 inline the causal graph on top of the full history; the longest
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

**Retrieved exemplars are shared across configs.** Stage 1 is keyed by the
encoder, not by the detector, and lands in `artifacts/`, not in any output tree —
so a local detector reuses the artifact a closed-source run already produced:
keep the same `artifacts_root` and `embed_model`, and the sweep skips stage 1 as
already complete.

## Report configs

`report_<ds>.yaml` lists the models and methods that go into the tables; add a
model name there once its runs finish. `gt: with` selects the `outputs/` tree
(the default for this baseline); `gt_in_prompt` labels the with-GT tree only.
Seeds are 1–20 for ww/traceelephant, 1–3 for correct-error.
