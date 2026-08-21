# ErrorProbe baseline

Reproduction of **ErrorProbe** ("Towards Self-Improving Error Diagnosis in
Multi-Agent Systems"; Li et al., arXiv:2604.17658 — the paper is at
[`../../vendored/ERRORPROBE/errorprobe.pdf`](../../vendored/ERRORPROBE/errorprobe.pdf)).
The original code cannot be released, so the authors ship a **simplified
reference implementation** instead, vendored under
[`vendored/ERRORPROBE/`](../../vendored/ERRORPROBE). Per this repo's rules
(GUIDE.md), that code — not the paper — is the faithfulness reference: our
prompts, parsers and control flow are parity-tested against it, and every
deliberate deviation is recorded in [`IMPLEMENTATION.md`](IMPLEMENTATION.md).

## The idea in one sentence

A failed multi-agent run hides its cause early and shows its symptom late, so
a single reading of the log tends to blame whoever spoke last; ErrorProbe
reads the story backward from the failure, names the mistake using a standard
catalog of ways agent teams fail, asks a second model to check the diagnosis,
and — in the full paper system — remembers diagnoses that survived checking so
the next case starts smarter.

## What the paper describes

The paper's pipeline has three stages plus a memory:

1. **Tag suspicious moments.** Parse the raw log into structured turns (who
   spoke, in what role, doing what) and mark local anomalies using the **MAST
   taxonomy** — a published catalog of 14 ways multi-agent systems fail,
   grouped into three families: not following the task or role as specified,
   breakdowns in how agents coordinate (ignoring each other's input, saying
   one thing and doing another), and failures of verification (checking
   nothing, or checking wrongly). These tags narrow the search before any
   deep reasoning happens.
2. **Trace backward from the symptom.** Build a dependency graph of which
   turn used which earlier turn's output, then walk from the failure back
   through only the turns the failure actually depended on, discarding
   branches that succeeded. The point is to defeat the "lost in the middle"
   problem: the decisive mistake at step 5 should not drown in the 45 steps
   that follow it.
3. **Diagnose with a three-role team.** A *Strategist* proposes hypotheses
   about the root cause; an *Investigator* must back each hypothesis with
   tool-grounded evidence (re-running code in a sandbox, probing logical
   pre/post-conditions) — it may not simply assert "the code is wrong"; an
   *Arbiter* weighs the evidence and emits the final (agent, step) verdict
   with a confidence score.
4. **Remember what was verified.** Diagnoses that pass verification with
   enough confidence enter an episodic memory ("verified-before-write"), keyed
   by an error signature and ranked by recency × frequency × impact. Future
   diagnoses retrieve similar past cases as guidance. The gate is the point:
   only evidence-backed patterns accumulate, so the memory transfers across
   domains instead of collecting hallucinations.

On its three benchmarks the paper reports that this raises average step-level
accuracy from 21.27 (one-shot LLM-as-a-judge, Claude 3.7 Sonnet) to 41.90,
and to 42.73 with memory.

## What the vendored code actually implements

The reference implementation is deliberately smaller than the paper, and two
of its gaps matter for how this baseline behaves:

| paper | vendored code |
|---|---|
| Strategist–Investigator–Arbiter team | two agents: an **Analyzer** (finds the error) and a **Verifier** (reviews the finding) |
| tool-grounded verification (code sandbox, logic probes) | the Verifier is a second LLM opinion — no tools, no execution |
| backward tracing over a parsed dependency graph | an optional turn-by-turn LLM walk from the last turn backward |
| MAST anomaly tagger as a separate stage | present in the code but never instantiated (commented out) |
| verified episodic memory guiding diagnosis | the memory machinery exists (write gate, scoring), but **retrieved patterns are never inserted into any prompt** — the baseline prompt builder accepts them and ignores them |

That last row is why this adaptation has no memory/training stage: in the
vendored code, running with or without memory provably produces identical
predictions, so building the memory would only add cost. The finding, with
line references, is in [`IMPLEMENTATION.md`](IMPLEMENTATION.md).

## The two modes

The vendored config chooses between two prediction paths; we ship both, as
separate method directories that the report scores side by side.

**`errorprobe` — truncated-history mode** (the vendored default). Two model
calls per trajectory:

| call | asks the model to | consumes |
|---|---|---|
| Analyzer | name the error span (start/end turn), the responsible agent, a MAST failure mode and a reason, with a confidence — as one JSON object | the task + the **last 15 turns** (500 chars each) + the agent list + the 14-mode taxonomy |
| Verifier | judge whether that analysis holds, estimate the impact of fixing it, and suggest a fix | the analysis + ±2 turns of context around the span (300 chars each) |

The Analyzer alone decides the prediction; the Verifier's verdict is recorded
as evidence but never overrides it (that is the vendored flow). Because the
Analyzer numbers turns inside its 15-turn window, its span is window-local;
the output document remaps it to absolute indices
(`predicted_step = span_start + max(0, len(history) − 15)`) and keeps the raw
span alongside — see IMPLEMENTATION.md for why.

**`errorprobe_bt` — backward-tracing mode** (the vendored
`backward_tracing.enabled` path). The full trace, walked from the last turn
toward the first:

| call | asks the model to | when |
|---|---|---|
| relevance | say whether this turn matters to the failure (JSON yes/no + score) | every turn examined |
| root cause | say whether this turn is the root cause, with an error type, confidence and observations | every relevant turn |
| summarize | condense the findings so far into a short context | whenever findings changed and exceed 5 observations |
| prune / merge | drop observations irrelevant to the leading hypothesis; spot duplicate hypotheses | every 15 turns examined |
| synthesize | turn the accumulated findings into the final (agent, step, type, reason) diagnosis | once, at the end |

The walk stops early once it has examined at least 10 turns and holds a
hypothesis with confidence ≥ 0.75 (or two agreeing ones ≥ 0.6), and gives up
after 100 turns. If no hypothesis ever forms, the vendored fallback answers
from the strongest observation (in practice: the final turn). The Verifier
then reviews the result, exactly as in the other mode. Steps here are
absolute — no remap needed.

## GT settings

The vendored prompts never contain the task's answer, so `--gt without` is the
vendored-faithful setting **and this baseline's default** (like correct and
raffles; the inverse of prompting and chief). Default results land in
`outputs-nogt/`. `--gt with` appends the prompting baselines' exact
`The Answer for the problem is: ...` line to the task text in every prompt
that carries the task (the Analyzer; the backward walk's root-cause and
synthesis calls) and writes under `outputs/`.

## Layout

- `prompts.py` — the Analyzer/Verifier prompt builders and JSON parsers,
  transcribed byte-for-byte from the vendored `llm_agents.py`, plus the MAST
  taxonomy table.
- `backward.py` — the backward-tracing port: trace index, working memory and
  the walk, with their prompts inline, as generators the runner can drive.
- `methods.py` — the two per-trajectory programs (`errorprobe`,
  `errorprobe_bt`) and the output-document mapping.
- `predict.py` — one `(model, subset, mode)` per invocation; `--mode`,
  `--gt`, `--method-dir`.
- `sweep.py` — the models × subsets × modes grid; shells out one child per
  combo.
- `report.py` — thin alias of `baselines.prompting.report`.
- `configs/` — `<ds>-api.yaml` (what to run) and `report_<ds>.yaml` (what to
  score); see [`configs/README.md`](configs/README.md) to add local models.

## Running

Everything runs from the repo root. The front door is
`scripts/errorprobe/run.sh`:

```bash
DATASET=ww bash scripts/errorprobe/run.sh                                # both modes, both models
DATASET=ww SUBSET=hand-crafted MODEL=gpt-4o bash scripts/errorprobe/run.sh
DATASET=ww MODEL=gpt-4o MODE=truncated bash scripts/errorprobe/run.sh    # the cheap mode only
DATASET=ww MODEL=gpt-4o END_IDX=10 DRY_RUN=1 bash scripts/errorprobe/run.sh  # preview
```

Runs resume on file existence — rerun the same command after a crash and only
the missing trajectories execute. Then build the tables (CPU only):

```bash
python -m baselines.errorprobe.report --config baselines/errorprobe/configs/report_ww.yaml --check-only
python -m baselines.errorprobe.report --config baselines/errorprobe/configs/report_ww.yaml
```

## Cost

The truncated mode is the cheapest baseline in this repo: two bounded calls
per trajectory (the Analyzer prompt caps itself at 15 × 500 characters). The
backward mode is the opposite: one relevance call per examined turn, a
root-cause call per relevant turn, plus summarize/prune/merge maintenance —
roughly 2–3 calls per examined turn, with the walk bounded at 100 turns. On
long corpora (hand-crafted, magentic) expect on the order of 100–300 calls
per trajectory; the shipped `correct-error` config therefore enables the
truncated mode only (2,226 trajectories — enable `backward` there
deliberately).
