# Prompting baselines

The three Who&When attribution strategies — reimplemented to run **batched over
trajectories** on local checkpoints (vLLM) or **per-trajectory with incremental
writes** on OpenAI-compatible APIs, while keeping the vendored prompts and
control flow verbatim (see the top-level [`README.md`](../../README.md) for the
project overview):

- **`all_at_once`** — show the whole conversation, ask for `Agent Name` + `Step Number` in one shot.
- **`step_by_step`** — judge each step "does this contain the decisive error? Yes/No";
  the prediction is the earliest "Yes".
- **`binary_search`** — recursively ask which half of the segment holds the critical mistake.

Predictions cover every trajectory; evaluation against the per-seed splits
(step@1 / agent@1) is deferred entirely to `report.py`.

## Layout

- **Core logic:**
  - `methods.py` — the verbatim Who&When prompts/parsers, plus the three methods
    as **per-trajectory generator programs** (yield one round's prompts, receive
    responses, return prediction + full call log). Also `strip_think`.
  - `runner.py` — the two drivers (`run_batched` lockstep for vLLM,
    `run_streaming` thread-pool for APIs) and `OutputWriter` (atomic
    per-trajectory files; file existence = resume ledger).
  - `backends/` — `vllm` / `openai` / `dummy` behind one
    `generate(message_lists) -> list[str]` protocol; adding a provider that
    speaks the OpenAI protocol is a config entry, not code.
  - `predict.py` — runner: one `(model, subset, method)` per invocation →
    `{output}/{method}/<id>.json` per trajectory + `_run.json` snapshot.
  - `report.py` — completion check + per-seed val/test/full tables.
  - `reparse.py` — re-derive `all_at_once` predictions from stored `raw`, no GPU
    (recovers e.g. DeepSeek's markdown-bolded labels).
- **Orchestration (chooses arguments, runs nothing itself):**
  - `sweep.py` — grid over models × subsets × methods; shells out one child per
    combo to `predict`. Models are declared in the config's `model_specs`
    (backend + paths/params; vLLM specs may override sampling knobs per model).
  - `configs/<ds>.yaml` — inference config per dataset; `configs/report_<ds>.yaml`
    — report config (explicit `seeds`, split ratios, roots).
  - `scripts/run_qwen.sh`, `scripts/run_deepseek.sh` — per-model wrappers over
    the sweep, one GPU each (`GPU`/`DATASETS`/`DRY_RUN`/`EXTRA_SET` env knobs).
- `tokenizers/deepseek-8b/` — corrected tokenizer dir for DeepSeek-R1-Distill
  (its shipped `tokenizer_config` builds the wrong SentencePiece tokenizer;
  this fixes only `tokenizer_class`).

Outputs land under `outputs/<ds>/<subset>/<model>/<method>/` (one JSON per
trajectory) and `outputs/<ds>/reports/` (tables).

## Faithfulness notes (the details that bite)

- **Prompts, regexes and decision rules are verbatim** from
  `vendored/Agents_Failure_Attribution/Automated_FA/Lib/local_model.py` /
  `evaluate.py` — including the intentionally unformatted literal `{idx}` in the
  step_by_step prompt. `tests/test_vendored_parity.py` drives the vendored code
  itself and asserts byte-identical prompts and identical decisions; don't
  "fix" either side.
- **The agent-identity field is `history[t]["role"]` for every dataset** (this
  repo's data stores the agent name in `role`; the vendored `name`/`role`
  switch targeted the original Who&When layout).
- **`strip_think`** removes `<think>` blocks (and dangling closers) before any
  parsing, so reasoning backbones work with the vendored regexes; markdown
  decoration is stripped before the `Agent Name`/`Step Number` regexes.
- **step_by_step early-stop vs batch**: vLLM judges all steps in one giant
  batch; APIs stop at the first "Yes" (the vendored control flow). Predictions
  are identical by construction — only the number of issued calls (and hence
  logged `calls`) differs.
- **binary_search ambiguity** defaults to the upper half, and the lower-half
  recursion clamps with `min(mid+1, end)` — both vendored behaviors.
- **DeepSeek-R1-Distill needs token headroom**: it always thinks, and
  `gen_max_tokens` caps thinking + answer combined; its model spec sets 8192.
- **Step indexing is 0-based** end-to-end; `report.py` compares
  `int(predicted_step) == int(gold_step)` (gold steps are *strings* in ww) with
  no ±1 shifting anywhere.

## Running

```bash
# The grid (idempotent per trajectory; rerun = resume):
CUDA_VISIBLE_DEVICES=0 python -m baselines.prompting.sweep \
    --config baselines/prompting/configs/ww.yaml [--dry-run] [--set overwrite=true]

# Reports (CPU only):
python -m baselines.prompting.report --config baselines/prompting/configs/report_ww.yaml
```
