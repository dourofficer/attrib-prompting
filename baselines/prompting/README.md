# Prompting baselines

This is the **prompting baseline** sub-repo for the failure-attribution project (see the
top-level `CLAUDE.md`). Where the main method — Causal Residual Rescoring (CRR) — reads a
proxy model's *internal representations*, this baseline instead **asks an LLM directly**, in
natural language, which step of a failed trajectory was the decisive error. It is the
"just prompt a strong model" comparison point against which CRR / SVD are measured, on the
*identical* per-seed val/test splits.

## What it does

Given the same trajectories (Who&When / `ww`, CORRECT-Error, TraceElephant), it runs the three
attribution prompting strategies from the original Who&When paper — reimplemented here to
**batch over trajectories with VLLM** while keeping the prompts and control-flow verbatim:

- **`all_at_once`** — show the whole conversation, ask for `Agent Name` + `Step Number` in one shot.
- **`step_by_step`** — judge each step "does this contain the decisive error? Yes/No"; the
  prediction is the earliest "Yes".
- **`binary_search`** — recursively ask which half of the segment holds the critical mistake.

Predictions are split-agnostic at inference time; evaluation against the CRR test splits (step@1 /
agent@1) is deferred to `report.py`.

## Layout (mirrors the repo's `src/` ↔ `experiments/` split, in miniature)

- **Core logic (config-free, thin `main()` per stage):**
  - `predict.py` — runner: one `(model, subset, method)` per invocation → `predictions_method-{method}.jsonl`.
  - `engine.py` — `PromptEngine`, the single batched-VLLM `LLM.chat` wrapper; also `strip_think`.
  - `methods.py` — the three methods + their verbatim Who&When prompts and parsers.
  - `report.py` — completion check + per-seed comparison tables placing the three methods next to SVD/CRR.
  - `reparse.py` — re-derive `all_at_once` predictions from stored `raw` text, no GPU (recovers
    e.g. DeepSeek's markdown-bolded labels).
- **Orchestration (chooses arguments, runs nothing itself):**
  - `sweep.py` — grid over models × subsets × methods; shells out one child per combo to `predict`.
  - `configs/<ds>.yaml` — one inference config per dataset; `configs/report_<ds>.yaml` — one report config.
  - `scripts/run_deepseek.sh`, `scripts/run_qwen.sh` — per-model wrappers over the sweep, one GPU each.
- `tokenizers/deepseek-8b/` — corrected tokenizer dir for DeepSeek-R1-Distill (its shipped
  `tokenizer_config` builds the wrong SentencePiece tokenizer; this fixes only `tokenizer_class`).

Outputs land under `outputs-<ds>/prompting/<model>/<subset>/` (predictions) and
`outputs-<ds>/prompting-reports/` (tables).

## Conventions that bite (baseline-specific)

- **`role` is the agent identity for every dataset here.** This repo stores the agent name in
  `history[t]["role"]` and `mistake_agent` matches it — unlike the original Who&When `name`/`role`
  handcrafted switch.
- **Reasoning backbones need token headroom.** DeepSeek-R1-Distill always emits a `<think>` block
  and `gen_max_tokens` caps *thinking + answer combined*; 1024 gets eaten by reasoning before an
  answer appears. `run_deepseek.sh` defaults `gen_max_tokens` to 8192 for this reason.
- **`strip_think` runs before every parse**, regardless of the `enable_thinking` toggle, so
  reasoning traces never reach the Who&When regexes.
- **Report splits are reproduced byte-identically** to `src/svd/reproduce.py` (same `split_data`
  calls, same `split_model` id-source), so the baseline sits on exactly the CRR val/test seeds.

## Running

Everything runs **from the repo root** as `python -m baselines.prompting.<stage>` (the package is
importable as `baselines.…`). The sweep shares the repo's interface —
`--config <yaml> [--set key=value ...] [--dry-run]`; `--set` does dot-path overrides.

The two wrapper scripts are the front door. Each drives **one model across all datasets on one
GPU**; pair them on different GPUs to parallelise. They are controlled by **env knobs**, matching
`scripts/_common.sh`: `GPU`, `DATASETS`, `DRY_RUN`, `EXTRA_SET` (and `GEN_MAX_TOKENS` for deepseek).

```bash
# One model, one GPU, all datasets (ww → traceelephant → correct-error):
GPU=5 bash baselines/prompting/scripts/run_deepseek.sh
GPU=4 bash baselines/prompting/scripts/run_qwen.sh          # run on another GPU in parallel

# Narrow the dataset set, preview commands, or forward extra --set overrides:
GPU=5 DATASETS="traceelephant correct-error" bash baselines/prompting/scripts/run_deepseek.sh
GPU=5 DRY_RUN=1 bash baselines/prompting/scripts/run_qwen.sh
GPU=5 EXTRA_SET="--set overwrite=true" bash baselines/prompting/scripts/run_deepseek.sh
```

The old positional GPU form still works (`... run_deepseek.sh 5`); the `GPU=` env var wins when both are given.

Call the sweep directly for finer control (e.g. one dataset, both models):

```bash
CUDA_VISIBLE_DEVICES=5 uv run python -m baselines.prompting.sweep \
    --config baselines/prompting/configs/ww.yaml --dry-run
```

Predictions are **idempotent** — an existing `predictions_method-{method}.jsonl` is skipped unless
you pass `--set overwrite=true` (or `--overwrite` to `predict`).

Then build the comparison tables (CPU only; no GPU), and optionally re-parse `all_at_once`:

```bash
python -m baselines.prompting.report --config baselines/prompting/configs/report_ww.yaml
python -m baselines.prompting.report --config baselines/prompting/configs/report_ww.yaml --check-only
python -m baselines.prompting.reparse --dry-run     # re-derive all_at_once from stored raw text
```
