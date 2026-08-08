# attrib-prompting

Standalone reproduction of **prompting-based failure-attribution baselines** for
LLM multi-agent systems: given a *failed* multi-agent trajectory, predict which
agent, at which step, made the decisive mistake.

The three methods come from the Who&When paper (vendored verbatim under
[`vendored/Agents_Failure_Attribution/`](vendored/Agents_Failure_Attribution)):

- **all_at_once** — show the whole conversation, ask for `Agent Name` + `Step Number`.
- **step_by_step** — judge each step "does this contain the decisive error? Yes/No";
  the prediction is the earliest "Yes".
- **binary_search** — recursively ask which half of the segment holds the critical mistake.

Prompt strings, parse regexes and decision rules are **byte-identical** to the
vendored implementation (enforced by execution-level parity tests that drive the
vendored code itself — see [`tests/test_vendored_parity.py`](tests/test_vendored_parity.py)).

## Datasets

| dataset | subsets (size) | answer in prompt? | eval seeds |
|---|---|---|---|
| `ww` (Who&When) | algorithm-generated (126), hand-crafted (58) | yes | 1–20 |
| `correct-error` | arc (304), gaia (50), hotpot (578), math500 (157), mmlu_pro (92), musique (312), wikimqa (733) | no (`ground_truth` empty) | 1–3 |
| `traceelephant` | magentic (91), captain (85) | yes | 1–20 |

Corpus files live at `data/<dataset>/<subset>/<id>.json`; the agent identity is
`history[t]["role"]` for every dataset, and `mistake_agent`/`mistake_step` are
the gold labels. (`data/correct-full` and `data/synthetic` are unused here.)

## Install

```bash
pip install -e .            # CPU core: evaluation, tests, dry-runs, dummy backend
pip install -e ".[api]"     # + OpenAI-compatible API inference
pip install -e ".[vllm]"    # + local-checkpoint inference (needs a CUDA-matched vLLM)
```

Everything runs from the repo root as `python -m baselines.prompting.<stage>`.
Local checkpoints are expected under `../hub/` (see `model_specs` in
`baselines/prompting/configs/*.yaml`).

## Running inference

One `(model, subset, method)` per `predict` invocation; the `sweep` runs the
grid from a config; the shell scripts wrap the sweep per model:

```bash
# Local vLLM (one GPU each):
GPU=0 bash baselines/prompting/scripts/run_qwen.sh
GPU=1 bash baselines/prompting/scripts/run_deepseek.sh
GPU=0 DATASETS="ww" DRY_RUN=1 bash baselines/prompting/scripts/run_qwen.sh   # preview

# API model (set the key, add the model to `models:` in the config, or run directly):
export OPENAI_API_KEY=sk-...
python -m baselines.prompting.predict \
    --backend openai --model gpt-4o \
    --api-param max_tokens=1024 --api-param temperature=0.6 \
    --input data/ww/hand-crafted --output outputs/ww/hand-crafted/gpt-4o \
    --method all_at_once
```

API models are declared in `model_specs` (see `configs/ww.yaml` for worked
`gpt-4o` / `gpt-5` examples). The spec's `params` dict is sent to the API
**verbatim** — reasoning models like GPT-5 get `max_completion_tokens` and no
temperature; what was actually sent is recorded in `_run.json`. Any
OpenAI-compatible provider works via `base_url`; a differently-shaped API means
one new module in `baselines/prompting/backends/`.

### Outputs, incremental writes, resume

Every trajectory gets its own inspectable file the moment it completes:

```
outputs/<dataset>/<subset>/<model>/<method>/<id>.json   # prediction + gold + raw + call log
outputs/<dataset>/<subset>/<model>/<method>/_run.json   # run snapshot (params actually sent)
```

Each file records `predicted_agent` / `predicted_step`, the gold labels, the
decisive `raw` response, and `calls` — the full response log (every
step_by_step judgment with its step/agent, every binary_search round with its
`[start, end]` range). File existence is the resume ledger: **rerun the same
command and only missing trajectories are executed** (`--overwrite` redoes the
method directory). A crash mid-run costs at most the trajectories in flight —
this matters for API runs, where step_by_step fans out one call per step.

On API backends step_by_step **early-stops** at the first "Yes" (the vendored
control flow; ~50% fewer paid calls). The prediction is provably identical to
judging every step, because the earliest "Yes" wins and each step's prompt
depends only on the deterministic accumulated history; `--step-mode` overrides.

Inference covers **all** trajectories — no split is applied at inference time.

## Evaluation

Evaluation is fully decoupled from inference and mirrors the attribscope
project's protocol (verified cell-for-cell against its evaluation code over
2,424 table cells):

```bash
python -m baselines.prompting.report --config baselines/prompting/configs/report_ww.yaml
python -m baselines.prompting.report --config baselines/prompting/configs/report_ww.yaml --check-only
```

- **Splits**: per-seed val/test partitions are reproduced from the corpus file
  list (filenames numerically sorted, seeded shuffle, `{train: .3, val: .2, test: .5}`)
  — bit-identical to the original experiments, no external artifacts needed.
- **Metrics**: `step@1` is integer equality; `agent@1` normalizes with
  `standardize_role`/strip/lower and accepts the gold name as a substring of
  the prediction. A missing prediction counts as wrong.
- **Tables**: `outputs/<ds>/reports/<model>/<subset>/comparison_by_seed.tsv`
  (one row per seed, val/test plus split-independent `*_full` columns over the
  whole corpus) and `outputs/<ds>/reports/summary_mean_over_seeds.tsv`.

Post-hoc utilities: `python -m baselines.prompting.reparse` re-derives
all_at_once predictions from stored `raw` (recovers markdown-bolded labels) —
no GPU; `scripts/import_legacy_jsonl.py` converts old
`predictions_method-*.jsonl` trees into the per-trajectory layout.

## Layout

```
baselines/common.py            vendored helpers (file listing, split_data, standardize_role)
baselines/prompting/
  methods.py                   verbatim prompts/parsers + the three method programs
  runner.py                    batched (vLLM) & streaming (API) drivers, output writer
  backends/                    vllm | openai | dummy behind one generate() protocol
  predict.py  sweep.py         inference CLI + config-driven grid
  report.py   reparse.py       evaluation + raw re-parsing
  configs/                     per-dataset inference & report configs
  scripts/                     per-model wrappers (GPU/DATASETS/DRY_RUN env knobs)
baselines/{chief,correct}/     further baselines, not yet adapted to this layout
data/                          the corpora (never regenerated)
vendored/                      upstream baseline codebases, kept verbatim
outputs/                       committed inference outputs + report tables
tests/                         CPU-only, keyless (pytest)
```

## Tests

```bash
python -m pytest tests/ -q
```

Covers: execution-level prompt/control-flow parity against the vendored code,
bit-exact split reproduction against frozen fixtures, batch-composition parity
of the lockstep driver, resume/atomic-write behavior, OpenAI payload/retry/
concurrency semantics (fake client, no key needed), and the metric rules.
