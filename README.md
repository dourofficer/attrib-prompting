# attrib-prompting

Standalone reproduction of **prompting-based failure-attribution baselines** for
LLM multi-agent systems: given a *failed* trajectory, predict which agent, at
which step, made the decisive mistake.

The three methods come from the Who&When paper (vendored under
[`vendored/Agents_Failure_Attribution/`](vendored/Agents_Failure_Attribution)):
**all_at_once** (one-shot agent+step), **step_by_step** (per-step Yes/No, earliest
"Yes" wins), **binary_search** (recursive halving). Prompts, regexes and decision
rules are byte-identical to the vendored code, enforced by execution-level
parity tests.

## Datasets

| dataset | subsets (size) | answer in prompt? | eval seeds |
|---|---|---|---|
| `ww` (Who&When) | algorithm-generated (126), hand-crafted (58) | yes | 1–20 |
| `correct-error` | arc (304), gaia (50), hotpot (578), math500 (157), mmlu_pro (92), musique (312), wikimqa (733) | no | 1–3 |
| `traceelephant` | magentic (91), captain (85) | yes | 1–20 |

Corpora live at `data/<dataset>/<subset>/<id>.json`; agent identity is
`history[t]["role"]`, gold labels are `mistake_agent`/`mistake_step`.

## Install

```bash
pip install -e .            # CPU core: evaluation, tests, dry-runs
pip install -e ".[api]"     # + OpenAI-compatible API inference
pip install -e ".[vllm]"    # + local-checkpoint inference (CUDA-matched vLLM)
```

Run everything from the repo root. Local checkpoints are expected under `../hub/`.

## Running

One method at a time (see [`scripts/README.md`](scripts/README.md) for all knobs):

```bash
export OPENAI_API_KEY=sk-...
MODEL=gpt-4o     DATASET=ww SUBSET=hand-crafted bash scripts/all_at_once.sh
MODEL=qwen3.5-9b DATASET=traceelephant GPU=0    bash scripts/binary_search.sh   # all subsets
```

Or the full grid per dataset:

```bash
GPU=0 bash baselines/prompting/scripts/run_qwen.sh        # local models, all datasets
python -m baselines.prompting.sweep --config baselines/prompting/configs/ww-api.yaml
```

Configs are split by backend: `configs/<ds>.yaml` for local vLLM models,
`configs/<ds>-api.yaml` for closed-source APIs (`gpt-4o` active, `gpt-5` spec
ready). An API spec's `params` are sent verbatim and recorded in `_run.json`;
any OpenAI-compatible provider works via `base_url` — adding one is a config
entry, not code.

### Outputs & resume

```
outputs/<dataset>/<subset>/<model>/<method>/<id>.json   # prediction + gold + raw + call log
outputs/<dataset>/<subset>/<model>/<method>/_run.json   # run snapshot (params actually sent)
```

One file per trajectory, written atomically as it completes; `calls` logs every
model response (each step_by_step judgment, each binary_search round). File
existence is the resume ledger: **rerun the same command and only missing
trajectories execute** — a crash costs at most the trajectories in flight. On
API backends step_by_step early-stops at the first "Yes" (the vendored control
flow, ~50% fewer calls, provably identical predictions). Inference always covers
all trajectories; no split is applied at inference time.

## Evaluation

The complete GPT-4o/GPT-5 sweep, including raw responses and metrics, is
published under [`results/`](results/README.md).

Decoupled from inference; mirrors the attribscope protocol (verified
cell-for-cell over 2,424 table cells):

```bash
python -m baselines.prompting.report --config baselines/prompting/configs/report_ww.yaml [--check-only]
```

Per-seed val/test splits are reproduced from the corpus file list alone (sorted
stems, seeded shuffle, 0.3/0.2/0.5). `step@1` = integer equality; `agent@1` =
normalized match (gold may be a substring of the prediction); missing
predictions count as wrong. Tables land in `outputs/<ds>/reports/`
(`comparison_by_seed.tsv` per model/subset + `summary_mean_over_seeds.tsv`,
with split-independent `*_full` columns over the whole corpus).

Utilities: `python -m baselines.prompting.reparse` re-derives all_at_once
predictions from stored `raw` (no GPU); `scripts/import_legacy_jsonl.py`
imports legacy `predictions_method-*.jsonl` trees.

## Layout

```
baselines/shared/              method-agnostic infra: common.py helpers,
                               backends/ (vllm|openai|dummy), runner.py (drivers, writer)
baselines/prompting/           the three methods (verbatim prompts), predict/sweep/report,
                               configs/ (<ds>.yaml vLLM, <ds>-api.yaml APIs), per-model scripts
baselines/{chief,correct}/     further baselines, not yet adapted (see GUIDE.md)
data/                          corpora   ·  vendored/  upstream code, verbatim
outputs/                       committed results  ·  scripts/  per-method front doors
tests/                         CPU-only, keyless (pytest)
```

## Tests

```bash
python -m pytest tests/ -q
```

Vendored prompt/control-flow parity (drives the vendored code itself), bit-exact
split fixtures, driver batching parity, resume/atomic writes, OpenAI
payload/retry/concurrency (fake client), metric rules.
