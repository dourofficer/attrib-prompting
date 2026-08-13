# RAFFLES baseline

Reproduction of **RAFFLES** (Reasoning-based Attribution of Faults for LLM
Systems; Zhu et al., Capital One, EACL 2026 — the paper is at
[`raffles.pdf`](raffles.pdf), arXiv:2509.06822) — a training-free method that
treats fault attribution as a debate between two roles: a **Judge** that
proposes where a failed trajectory went wrong, and a panel of **Evaluators**
that grade how well the Judge argued it. The paper releases no code, so this
implementation is built from the paper alone; every choice the paper leaves
open is recorded in [`IMPLEMENTATION.md`](IMPLEMENTATION.md).

## The idea in one sentence

A single model pass over a long log tends to produce a confident but shallow
guess; RAFFLES makes the guess earn its confidence — the Judge must defend a
candidate on three separate criteria, independent verifiers score each defense,
and a candidate only wins once the panel is convinced, otherwise the critiques
go back to the Judge and it tries again.

## What it looks for: the decisive fault

The target is not just *any* mistake in the log. The paper defines a
**decisive fault** as the first mistake that actually sank the trajectory, and
splits that into three criteria the Judge must satisfy at once:

1. **Fault** — the agent really made a mistake at that step.
2. **Primacy** — it is the *first* mistake that relates to the final wrong
   outcome (earlier slips that got absorbed don't count).
3. **Decisiveness** — no later agent corrected it, or could have; had this
   step gone right, the trajectory would have succeeded.

This is a sharper target than the Who&When annotation guide's "most serious
mistaken agent", and it is what lets the method say *this* step, not just
*this neighborhood*.

## The loop

One iteration is four model calls — one Judge, three Evaluators — plus a free
rule-based check. The three Evaluator calls are independent and go out as one
round, which the API backend fans out concurrently (the parallelization the
paper uses to halve latency).

| call | asks the model to | consumes |
|---|---|---|
| Judge | name one `(agent, step)` candidate and defend it three ways: why it is a mistake (`mistake_reason`), why it is the first one (`first_mistake`), why it was never corrected (`mistake_not_corrected`). Nulls are forbidden; if nothing fits perfectly, pick the most pivotal contribution | problem + full log + all prior critiques |
| Evaluator 1 | verify the *mistake* rationale is logical and grounded in the log; score its soundness 0–100 | log + candidate + `mistake_reason` |
| Evaluator 2 | the same for the *first-mistake* rationale | log + candidate + `first_mistake` |
| Evaluator 3 | the same for the *never-corrected* rationale | log + candidate + `mistake_not_corrected` |
| rule check | *(no LLM call)* does step `t` exist, and does the agent speaking at it match the candidate? 100 or 0 | log + candidate |

The four scores sum to a confidence `C` out of 400. If `C > 350` the loop
stops and the candidate stands. Otherwise the candidate and all four verdicts
are appended to a running memory, the Judge sees that feedback on its next
attempt, and the loop repeats — up to `K` extra iterations (`max_iters`,
default 2, the paper's main setting; `K = 0` still runs one full pass). When
the budget runs out, the candidate with the highest `C` wins; ties go to the
latest, since that Judge saw the most feedback.

Evaluators grade the *argument*, not the verdict: their prompt tells them not
to be swayed by the conclusion, to reward specific reasoning a non-expert
could check against the log, and to score low whatever they cannot verify.
That asymmetry — propose globally, verify locally — is the paper's answer to
why one-shot judges and step-by-step scanners both fail on long trajectories.

Every response lands in the output file's `calls` list, and a per-iteration
`iterations` audit records each candidate, the four scores and the total, so a
finished trajectory carries the whole debate.

## What the paper reports

Step-level accuracy on Who&When without ground truth, Claude Sonnet 4, K=2:
**51.59** on Algorithmically-Generated and **22.41** on Hand-Crafted (27.59
with K=5) — the strongest published no-simulation numbers on this benchmark.
Reproduce K=5 side by side via
`EXTRA_SET="--set max_iters=5 --set method_dir=raffles.k5"`.

## GT settings

The paper evaluates Who&When **without ground truth**, so `--gt without` is
this baseline's default — the same as CORRECT, the inverse of prompting and
CHIEF. Default results therefore land in `outputs-nogt/`; `--gt with` adds the
prompting baselines' `The Answer for the problem is: ...` line to the problem
statement in every prompt (Judge and Evaluators) and writes to `outputs/`.

## Layout

- `prompts.py` — the Appendix F.3 texts (Judge, three Evaluators), the F.1
  template shell, the glue that fills them, and the JSON parsers + rule check.
- `methods.py` — `raffles_program`: the loop as a generator, so the shared
  runner supplies either columnar batching (vLLM) or per-trajectory
  concurrency (API).
- `predict.py` — one `(model, subset)` per invocation; `--max-iters`,
  `--threshold`, `--method-dir` for side-by-side K runs.
- `sweep.py` — the grid; shells out one child per combo.
- `report.py` — thin alias of `baselines.prompting.report`.
- `configs/` — `<ds>-api.yaml` (what to run) and `report_<ds>.yaml` (what to
  score); see [`configs/README.md`](configs/README.md) to add local models.

## Running

Everything runs from the repo root. The front door is `scripts/raffles/run.sh`:

```bash
DATASET=ww bash scripts/raffles/run.sh                                 # both models
DATASET=ww SUBSET=hand-crafted MODEL=gpt-4o bash scripts/raffles/run.sh
DATASET=ww MODEL=gpt-4o MAX_ITERS=5 bash scripts/raffles/run.sh        # paper's K=5
DATASET=ww MODEL=gpt-4o END_IDX=10 DRY_RUN=1 bash scripts/raffles/run.sh  # preview
```

Runs resume on file existence — rerun the same command after a crash and only
the missing trajectories execute. Then build the tables (CPU only):

```bash
python -m baselines.raffles.report --config baselines/raffles/configs/report_ww.yaml --check-only
python -m baselines.raffles.report --config baselines/raffles/configs/report_ww.yaml
```

## Cost

Up to `4 × (K+1)` model calls per trajectory (12 at the default K=2), but
every call re-reads the full log: the paper measures ~9.8k input tokens per
iteration on Algorithmically-Generated and ~41k on Hand-Crafted, or about
$0.12 and $0.31 per log at Claude Sonnet 4 prices. Early termination usually
stops well short of the worst case — the paper's own analysis shows candidate
churn collapsing by iteration 3. Budget accordingly before sweeping
`correct-error`, which is 2,226 of the repo's 2,586 trajectories.
