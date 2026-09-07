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

## The three modes

The vendored config chooses between two prediction paths; we ship both, as
separate method directories that the report scores side by side. A third mode,
`errorprobe_paper`, rebuilds the paper's own pipeline, which the vendored code
does not contain; it is described in its own section below.

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

## The third mode: `errorprobe_paper`

**The idea in one sentence.** Read the trace the way the paper says to: tag
each step with the MAST failure taxonomy, keep only the steps the failure
depended on, and let a three-role team (propose, check with a probe, judge)
name the decisive error.

This mode is built from the paper (Section 4, Algorithm 1) and from the MAST
authors' own annotator, because the vendored code implements none of it. It
has no parity test; golden fixtures pin every prompt and
[`IMPLEMENTATION.md`](IMPLEMENTATION.md) records every choice the paper leaves
open. It lives in [`paper/`](paper/), plugs in as `--mode paper`, and never
touches the other two modes. Four rounds of calls per trajectory:

| round | call | asks the model to | consumes |
|---|---|---|---|
| 1 | tagger (one per chunk) | tag steps with MAST failure modes, quoting the step as evidence | the task, the 14 MAST definitions, a one-line index of earlier steps, the chunk's steps in full |
| 1 | dependency (one per chunk) | list which earlier steps each step's content uses (cites, executes, answers, verifies, follows) | the task, the one-line index, the chunk's steps |
| 2 | Strategist | propose up to 3 hypotheses (step, agent, mode, rationale, which probe to run) | the task, the agents and their roles, the tags, the **condensed trace** |
| 3 | Investigator (one per hypothesis) | run the probe and report evidence: an execution log and diff (`code_exec`) or checked pre/post-conditions (`logic_probe`); never a bare assertion | the hypothesis, its step in full, its inputs and outputs in the graph, the final step |
| 4 | Arbiter | pick the decisive error with a confidence; say whether the pattern is worth remembering; emit a signature and a guard | the surviving hypotheses with their evidence; the filtered ones as one line each |

Between rounds 1 and 2 the backward tracing runs without a model: the LLM
edges join structural ones the trace format makes explicit (code to its exit
code, an instruction to its addressee's turn), a breadth-first search walks
incoming edges from the last step, and every step it never reaches is masked
out of the condensed trace. If the search leaves too few steps, adjacent turns
are added as a floor. When the condensed trace exceeds its budget, tagged
steps, the task and the symptom stay and the rest go furthest-from-the-symptom
first: that is how the MAST tags act as priors on the trace.

Calls per trajectory: `2C + 2 + H`, where `C` is the number of chunks (about
one per ten steps) and `H ≤ 3`. A ten-step alg-gen trace costs 7 calls, a
median hand-crafted trace 13, the longest 31.

What differs from the paper, and why:

| the paper | this mode |
|---|---|
| the Investigator re-runs code in a sandbox and probes pre/post-conditions with tools | both probes are prompts: `code_exec` asks the model to trace the code as an interpreter would and compare with the recorded output; `logic_probe` asks it to check conditions against the steps that establish them. No code runs (user decision) |
| a verified episodic memory feeds the Strategist and learns from the Arbiter | absent in this pass. The hooks stay: the Strategist renders a patterns section only when given patterns, and the Arbiter emits `novel_and_robust`, a signature and a guard, recorded in `memory_candidate` with the paper's τ = 0.7 |
| MAST priors enter the backward trace (`BackwardTrace(x, MastPriors)`) | tags steer the budget trimming and the Strategist; they do not enter the dependency prompt, so both round-1 families go out together |
| a dependency graph over the whole trace | LLM edges per chunk plus structural edges; the tagger and the parser see the trace in chunks of about 12k characters with absolute step numbers |
| the Arbiter "filters out hypotheses where the evidence is empty or inconclusive" | the program filters first and shows the Arbiter what survived; when nothing survives it asks for a best guess at low confidence and records `arbiter_saw_only_inconclusive` |
| (unspecified) | an unreadable Arbiter answer falls back to the best-evidenced hypothesis, as the vendored backward mode's synthesis step does; a Strategist that proposes nothing yields a null prediction |

The output document adds what each stage produced: `structure` (the parse),
`tags`, `graph`, `erf` (what was kept and what was masked), `hypotheses`,
`evidence`, `verdict`, `memory_candidate`, `failure_stage` and
`paper_params`. `calls` records every response with its role and, for chunked
and per-hypothesis calls, which chunk or hypothesis it served.

## GT settings

The vendored prompts never contain the task's answer, so `--gt without` is the
vendored-faithful setting **and this baseline's default** (like correct and
raffles; the inverse of prompting and chief). Default results land in
`outputs-nogt/`. `--gt with` appends the prompting baselines' exact
`The Answer for the problem is: ...` line to the task text in every prompt
that carries the task (the Analyzer; the backward walk's root-cause and
synthesis calls; every paper-mode prompt) and writes under `outputs/`.

## Layout

- `prompts.py` — the Analyzer/Verifier prompt builders and JSON parsers,
  transcribed byte-for-byte from the vendored `llm_agents.py`, plus the MAST
  taxonomy table.
- `backward.py` — the backward-tracing port: trace index, working memory and
  the walk, with their prompts inline, as generators the runner can drive.
- `methods.py` — the two vendored per-trajectory programs (`errorprobe`,
  `errorprobe_bt`) and the output-document mapping.
- `paper/` — the paper mode: `structure.py` (the per-step parse and action
  classifier), `tagger.py` (the MAST step-level tagger), `graph.py`
  (dependency edges, backward search, masking), `team.py` (Strategist,
  Investigator, Arbiter) and `program.py` (the four-round program). The MAST
  definitions it reads are vendored under `vendored/MAST/`.
- `predict.py` — one `(model, subset, mode)` per invocation; `--mode`,
  `--gt`, `--method-dir`, and the paper mode's knobs.
- `sweep.py` — the models × subsets × modes grid; shells out one child per
  combo.
- `report.py` — thin alias of `baselines.prompting.report`.
- `configs/` — `<ds>.yaml` / `<ds>-api.yaml` (what to run) and
  `report_<ds>.yaml` (what to score); see [`configs/README.md`](configs/README.md).

## Running

Everything runs from the repo root. The front door is
`scripts/errorprobe/run.sh`:

```bash
# Local checkpoints (configs/<ds>.yaml: qwen3.5-9b, deepseek-8b)
DATASET=ww GPU=0 bash scripts/errorprobe/run.sh                           # both vendored modes, both local models
DATASET=ww MODEL=qwen3.5-9b MODE=paper GPU=0 bash scripts/errorprobe/run.sh  # the paper pipeline (opt-in)

# Closed-source models (configs/<ds>-api.yaml: gpt-4o, gpt-5)
export OPENAI_API_KEY=sk-...
DATASET=ww MODEL=gpt-4o bash scripts/errorprobe/run.sh                   # truncated + backward
DATASET=ww SUBSET=hand-crafted MODEL=gpt-5 MODE=truncated bash scripts/errorprobe/run.sh
DATASET=ww MODEL=gpt-5 MODE=paper bash scripts/errorprobe/run.sh
DATASET=correct-error MODEL=gpt-4o bash scripts/errorprobe/run.sh        # truncated only (config)
DATASET=ww MODEL=gpt-4o END_IDX=10 DRY_RUN=1 bash scripts/errorprobe/run.sh  # preview
```

The front door picks whichever config declares `MODEL`; without `MODEL` it
runs every model in the local config. API models take the same path as the
prompting baselines: the spec's `params` go to the API verbatim and
`_run.json` records what was sent. Both API specs decode on the same
handicap in kind as `qwen3.5-9b`: gpt-4o at 512 tokens, temperature 1.0 and
top_p 0.95; gpt-5, which rejects a temperature, at the lowest reasoning
effort with a 4096-token cap. Their paper mode runs one hypothesis on a
10k-character condensed trace, as qwen's does;
[`configs/README.md`](configs/README.md) explains the choice.

The paper mode is opt-in: the shipped configs list only the two vendored
modes under `modes:`, so `MODE=paper` (or `--set modes=[paper]`) is how it
runs. Its knobs sit under the configs' `paper:` key.

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
deliberately). The paper mode sits between the two: `2C + 2 + H` bounded
calls, where `C` grows by one per ten steps and `H` is at most 3, so 7 calls
on a ten-step trace and about 13 on a median hand-crafted one. Its tagger
prompt is the largest in this baseline (the 14 MAST definitions plus a 12k
character chunk, about 8k tokens); `include_mast_examples` adds the MAST
authors' worked examples, another 17k tokens, and does not fit a 32k window.
The shipped local configs decode `qwen3.5-9b` on a deliberate handicap in
every mode (512 generation tokens, temperature 1.0, a 16k window) and give
its paper mode one hypothesis on a 10k-character condensed trace, so its cost
sits close to the vendored truncated mode; `deepseek-8b` runs the full budget.
[`configs/README.md`](configs/README.md) explains the choice.
