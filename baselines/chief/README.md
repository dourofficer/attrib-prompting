# CHIEF baseline

This is the **CHIEF baseline** sub-repo for the failure-attribution project (see the top-level
`CLAUDE.md`). CHIEF — *"From Flat Logs to Causal Graphs: Hierarchical Failure Attribution for
LLM-based Multi-Agent Systems"* — attributes failure by first reconstructing the trajectory into a
**hierarchical causal graph** (subtask → agent → step) and then reasoning over that graph, rather
than reading the log flat. The authors' implementation is **vendored verbatim** at
`baselines/CHIEF/` (with `chief.pdf`, the RAG index/KB, and their Who&When copy); this directory is
the local-vLLM re-implementation that runs it over **our** backbones and datasets on the identical
per-seed val/test splits as CRR / SVD / prompting.

## What it does

CHIEF is exactly **six sequential LLM calls per trajectory** (the vendored `CHIEF.py`
`step1…step6`), all greedy (`temperature 0.0`):

1. **Subtask decomposition** — split the history into contiguous, non-overlapping step ranges with
   a name, a virtual *oracle*, evidence, and loop info; on Who&When, stage 1 is few-shot seeded by
   **RAG** over the vendored GAIA + AssistantBench knowledge bases.
2. **Subtask edges** — causal edges between consecutive subtasks (data transfer, failure modes).
3. **Agent nodes** — per-subtask OTAR parsing (Action/Observation/Thought/Result) + step-level data flows.
4. **Agent edges** — intra-subtask dependencies between agents, with failure modes.
5. **Candidate set** — reason over the assembled DAG with the paper's three localization rules;
   emit ≥5 candidate error steps annotated with loop / data / irrecoverability issues.
6. **Final attribution** — pick the single most responsible `(agent, step)` from the candidates.

The prediction is the stage-6 `(Agent Name, Step Number)` pair; evaluation (step@1 / agent@1 on the
CRR test splits) is deferred to `report.py`, exactly like the prompting baselines.

## Faithfulness to the vendored implementation

**Every prompt string and every parsing regex is lifted verbatim from
`baselines/CHIEF/CHIEF.py`** — `stages.py` only splits each vendored `stepN` (which interleaved
*build prompt → call API → parse*) into side-effect-free `build_stepN` / `parse_stepN` pairs.
A stage-by-stage parity test (identical dummy inputs + canned LLM outputs through both codebases)
confirms byte-identical prompts and deep-equal parsed structures for all six stages, the DAG
assembly, and the `normalize_agent`/`normalize_step` helpers. The deliberate differences, all
infrastructure:

- **Local vLLM instead of the OpenAI API.** The vendored code calls an OpenAI-compatible endpoint
  one prompt at a time; here `engine.py` reuses the prompting baseline's `PromptEngine` (same chat
  templates, same `enable_thinking` toggle) so any local checkpoint is a backbone. The vendored
  system prompt and greedy decoding are preserved.
- **`strip_think` before every parse.** Reasoning backbones emit `<think>` blocks the vendored
  parsers never saw; `engine.strip_think` (a hardened superset of prompting's) removes them so the
  verbatim regexes see only the answer.
- **Batched execution.** `pipeline.py` runs the six stages *columnar* — one batched `generate` per
  stage across all trajectories. Each trajectory's stage-N prompt depends only on its own parsed
  stage-(N−1) output, so this is identical in effect to the per-sample loop; `reference.py` is that
  literal per-sample loop (shared builders/parsers) kept for correctness checks.
- **Linux-safe, lazy RAG.** The vendored `RAGRetriever` hardcodes Windows backslash paths and loads
  at import time; `rag.py` wraps the same FAISS + MiniLM logic (including the quirky
  `combined_sorted[1:top_k]` slice, which returns a single example at the default `top_k=2`) with
  configurable roots/KBs. With both KBs selected the retrieved examples match the vendored search.
- **RAG is a config axis, not hardcoded.** Who&When is CHIEF's native domain → `rag.enabled: true`
  (faithful). CORRECT-Error / TraceElephant are off-domain for the GAIA/AssistantBench KBs →
  disabled there; stage 1 then omits the retrieved-example block (the only prompt deviation, and
  only when RAG is off).

## Layout (mirrors the repo's `src/` ↔ `experiments/` split, in miniature)

- **Core logic (config-free):**
  - `stages.py` — the six `build_stepN`/`parse_stepN` pairs + `build_dag_graph`; the single source
    of truth both execution paths share.
  - `pipeline.py` — columnar/batched runner (default). `reference.py` — per-sample faithful runner.
  - `engine.py` — re-export of prompting's `PromptEngine` + hardened `strip_think`.
  - `rag.py` — `ChiefRetriever` over the vendored `baselines/CHIEF/rag/{index,kb}`.
  - `predict.py` — runner: one `(model, subset)` per invocation → `predictions_method-chief.jsonl`
    in the prompting schema, so the whole report stack applies unchanged.
  - `report.py` — thin alias of `baselines.prompting.report` (schema-compatible predictions).
- **Orchestration (chooses arguments, runs nothing itself):**
  - `sweep.py` — grid over models × subsets; shells out one child per combo to `predict`.
  - `configs/<ds>.yaml` — one inference config per dataset (models, paths, RAG toggle, vLLM knobs);
    `configs/report_<ds>.yaml` — one report config (same split sources as the prompting reports).
  - `scripts/run_chief.sh` — the front door: models × datasets sequentially on one GPU.

Outputs land under `outputs-<ds>/chief/<model>/<subset>/` (predictions) and
`outputs-<ds>/chief-reports/` (tables).

## Conventions that bite (baseline-specific)

- **Stage parsers are lenient by design — never "fix" them.** The vendored regexes silently drop
  malformed blocks, default missing floats to `0.0`, and truncate subtasks to the shortest parsed
  field list (`min(len(names), …)`). Tightening any of this changes the method.
- **Model paths are `../hub/...` relative to the repo root** (matching the dataset manifests);
  run everything from the repo root or the checkpoints won't resolve. Only `qwen3.5-9b` and
  `deepseek-8b` are currently downloaded — fetch the rest into `../hub/` before sweeping them.
- **Reasoning backbones need token headroom.** DeepSeek-R1-Distill always thinks and
  `gen_max_tokens` caps thinking + answer combined; CHIEF's structured stage outputs are long, so
  `run_chief.sh` bumps deepseek-8b to 8192 (`DEEPSEEK_GEN_MAX_TOKENS` to override) — same fix as
  prompting's `run_deepseek.sh`.
- **`history` step indexing is 0-based** and the stage-6 `Step Number` is compared to
  `mistake_step` as an integer by the shared report; no ±1 shifting anywhere.
- **RAG assets are the vendored ones** (`baselines/CHIEF/rag/{index,kb}`); the FAISS indices were
  built with `sentence-transformers/all-MiniLM-L6-v2` — don't swap the embedder without rebuilding.
- **`--mode per_sample` exists for auditing, not sweeping.** It issues one `generate` per call
  (GPU-starved); use it to spot-check that batched results match the literal vendored control flow.

## Running

Everything runs **from the repo root** as `python -m baselines.chief.<stage>`. The sweep shares the
repo interface — `--config <yaml> [--set key=value ...] [--dry-run]`; `--set` does dot-path
overrides. Predictions are **idempotent** — an existing `predictions_method-chief.jsonl` is skipped
unless you pass `--set overwrite=true` (or `--overwrite` to `predict`).

```bash
# The front door: all models × all datasets on one GPU (sequential, idempotent):
bash baselines/chief/scripts/run_chief.sh 4

# Narrow it, or forward overrides:
MODELS="qwen3.5-9b"  bash baselines/chief/scripts/run_chief.sh 2
MODELS="deepseek-8b"  bash baselines/chief/scripts/run_chief.sh 3
MODELS="qwen3.5-9b" DATASETS="ww" bash baselines/chief/scripts/run_chief.sh 0
bash baselines/chief/scripts/run_chief.sh 4 --set overwrite=true

# One dataset directly through the sweep:
CUDA_VISIBLE_DEVICES=4 uv run python -m baselines.chief.sweep \
    --config baselines/chief/configs/ww.yaml --dry-run
```

Then build the comparison tables (CPU only), which place CHIEF next to SVD/CRR per seed:

```bash
python -m baselines.chief.report --config baselines/chief/configs/report_ww.yaml
python -m baselines.chief.report --config baselines/chief/configs/report_ww.yaml --check-only
```
