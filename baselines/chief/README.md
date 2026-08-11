# CHIEF baseline

Reproduction of **CHIEF** (Causal HIErarchical Failure attribution; the paper is
vendored at [`vendored/CHIEF/chief.pdf`](../../vendored/CHIEF/chief.pdf)) — a
training-free method that stops reading an execution log as a flat sequence.
Instead it rebuilds the trajectory into a **hierarchical causal graph**
(subtask → agent → step), then walks that graph top-down to find the step that
caused the failure.

## The idea in one sentence

A long multi-agent log hides *why* one step went wrong, because the evidence
that condemns it is scattered across dozens of other turns; CHIEF spends five
LLM calls turning the log into a structure that makes those links explicit, then
asks the sixth call a much narrower question.

## Pipeline

Two stages, both resumable, orchestrated by `sweep.py` (each is also a
directly-invocable module). Stage 2 is six sequential LLM calls per trajectory.

| stage | module | one unit of work | writes |
|---|---|---|---|
| 1. ragprep | `ragprep.py` | embed each trajectory's *question* (MiniLM) and search the committed GAIA + AssistantBench index for a worked decomposition example | `artifacts/<ds>/<subset>/rag/<embed_model>.json` (+ `.meta.json`) |
| 2. detection | `predict.py` | the six calls below | `<gt-root>/<ds>/<subset>/<model>/chief/<id>.json` |

The six calls, all greedy (`temperature 0.0`), each parsed before the next is
built:

| # | asks the model to | consumes |
|---|---|---|
| 1 | split the history into contiguous, non-overlapping step ranges, each with a name, a one-line *oracle*, evidence and loop info | history + the retrieved exemplar |
| 2 | draw causal edges between **consecutive** subtasks, with the data transferred and how it could fail | stage 1 |
| 3 | for each subtask, summarize every acting agent as **OTAR** (Observation, Thought, Action, Result) plus the step-level data flow | stage 1 |
| 4 | draw agent→agent edges *within* each subtask | stage 3 |
| — | *(no LLM call)* assemble the graph from stages 1–4 | — |
| 5 | walk the graph and shortlist ≥5 candidate error steps, tagged with loop / data / irrecoverability issues | the graph |
| 6 | pick the **single** most responsible `(agent, step)` and say why | candidates + graph |

Stage 6's answer — `Agent Name:` / `Step Number:` / `Reason for Mistake:` — is
the prediction. All six responses are kept in the output file's `calls` list, so
a finished trajectory carries its whole reasoning transcript and the graph can
be reconstructed without re-running anything.

## Where retrieval fits, and why it is precomputed

Stage 1 is few-shot seeded: the model sees one worked example of a task being
decomposed into steps, retrieved from a knowledge base of 165 GAIA and 33
AssistantBench tasks that ships with the vendored code
(`vendored/CHIEF/rag/`). The exemplar is a *decomposition template*, not domain
knowledge — which is why every dataset keeps it on, even where the tasks are
unrelated to web search.

That lookup depends only on the question, the knowledge base and the encoder —
not on the detector, and not on the GT setting. So it runs once per subset and
lands in **`artifacts/`**, the sibling of `outputs/` that holds precomputed
*inputs* rather than predictions:

```
artifacts/<ds>/<subset>/rag/all-MiniLM-L6-v2.json    # stage 1, encoder-keyed
outputs/<ds>/<subset>/<model>/chief/<id>.json        # stage 2, predictions
```

Detection then reads that JSON. No FAISS search and no embedding model run
during the six calls, so the machine holding the API key needs neither
`faiss` nor `sentence-transformers` installed (they are the optional `[rag]`
extra), every rerun injects byte-identical exemplars, and the artifact is
committed so nobody recomputes it.

## The one prompt deviation

Each stage prompt gains a single sentence — *"Steps are indexed from 0, so the
first entry is step 0."* — because the vendored prompts never say where the
numbering starts, and this corpus lacks the per-turn `step` labels the original
Who&When files carried. Without it models count from 1 and every prediction
lands one past the gold index. `--step-hint off` restores the vendored bytes,
and the parity test proves that sentence is the only difference.
[`IMPLEMENTATION.md`](IMPLEMENTATION.md) has the evidence.

One quirk worth knowing before reading the artifact: the vendored search returns
`combined_sorted[1:top_k]`, **dropping the best hit**, so the default `top_k: 2`
injects exactly one exemplar — the runner-up. It is reproduced, not fixed; see
[`IMPLEMENTATION.md`](IMPLEMENTATION.md), which also records how the vendored
code differs from the paper's appendix (no separate oracle-synthesis stage, no
step-edge prompt).

## GT settings

Every vendored stage prompt carries the task answer, so with-GT is this
baseline's default — the inverse of CORRECT, the same as prompting.

| stage | affected by `--gt`? | why |
|---|---|---|
| 1. ragprep | no | keyed by the question, the knowledge base and the encoder; one artifact serves both settings |
| 2. detection | **yes** | `--gt without` drops `The correct answer for the problem is: ...` from **all six** prompts |

With-GT results land in `outputs/`, without-GT in `outputs-nogt/`, identical
inner layout.

## Layout

- `stages.py` — the six `build_stepN`/`parse_stepN` pairs + `build_dag_graph`,
  lifted verbatim from `vendored/CHIEF/CHIEF.py`.
- `methods.py` — `chief_program`: the six stages as a generator, so the shared
  runner supplies either columnar batching (vLLM) or per-trajectory concurrency
  (API). Also the hardened `strip_think`.
- `ragprep.py` / `rag.py` — stage 1 and the FAISS retriever it wraps.
- `predict.py` — one `(model, subset)` per invocation.
- `sweep.py` — the grid; shells out one child per stage/combo.
- `report.py` — thin alias of `baselines.prompting.report`.
- `configs/` — `<ds>-api.yaml` (what to run) and `report_<ds>.yaml` (what to
  score); see [`configs/README.md`](configs/README.md) to add local models.

## Running

Everything runs from the repo root. The front door is `scripts/chief/run.sh`:

```bash
# Precompute the exemplars once per dataset (CPU, needs pip install -e ".[rag]"):
DATASET=ww STAGES=ragprep bash scripts/chief/run.sh

# Detect (both models in the config), or narrow it:
DATASET=ww bash scripts/chief/run.sh
DATASET=ww SUBSET=hand-crafted MODEL=gpt-4o bash scripts/chief/run.sh
DATASET=ww MODEL=gpt-4o GT=without bash scripts/chief/run.sh
DATASET=ww MODEL=gpt-4o END_IDX=10 DRY_RUN=1 bash scripts/chief/run.sh   # preview
```

Runs resume on file existence — rerun the same command after a crash and only
the missing trajectories execute. Then build the tables (CPU only):

```bash
python -m baselines.chief.report --config baselines/chief/configs/report_ww.yaml --check-only
python -m baselines.chief.report --config baselines/chief/configs/report_ww.yaml
```

## Cost

Six calls per trajectory, and stages 5–6 inline the causal graph on top of the
full history. The paper measures ~55k tokens per hand-crafted case and ~20k per
algorithm-generated one — 2.5–3× a single all-at-once prompt. Budget
accordingly before sweeping `correct-error`, which is 2,226 of the repo's 2,586
trajectories.
