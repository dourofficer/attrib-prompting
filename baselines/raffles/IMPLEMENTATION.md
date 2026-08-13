# RAFFLES implementation notes

The details that decide whether a rerun of this baseline is comparing against
the paper or against our reading of it. Everything below is a consequence of
one fact: **RAFFLES ships no code** (verified against arXiv:2509.06822 and the
authors' pages as of August 2026), so unlike every other baseline in this repo
there is no `vendored/` source and no execution-level parity test. The paper
PDF (`raffles.pdf`) is the source of truth.

## How the prompts were transcribed

The prompt texts come from Appendix F.3 (Judge, three Evaluators) and Appendix
F.1 (the template shell the paper says the Judge reuses), pages 7678–7686 of
the PDF. Transcription rules:

- **Every word and paragraph break is preserved.** A PDF wraps lines at the
  width of its figure boxes, so the original line breaks are unrecoverable;
  box-width wraps are joined into logical lines. Nothing else is normalized —
  including the stray `.**.` after "at that step" in the Judge's
  `mistake_reason` field description, which is in the paper.
- **The three Evaluator prompts are one template.** Their texts are identical
  word for word except the quoted criterion phrase each verifies (the PDF
  boxes wrap the same sentences differently, which is layout, not content).
  `prompts.py` keeps one `EVALUATOR_TEMPLATE` parameterized by criterion.
- **`{{`/`}}` in the PDF are format-string escapes.** The appendix reproduces
  the authors' Python format strings (one box even leaks a
  `binary_search_task_output = """` line), so the JSON example blocks render
  with single braces, and `{task_log}`/`{error_step}` are fill slots.
- Golden fixtures (`tests/fixtures/raffles_golden_prompts.json`, captured by
  `capture_raffles_goldens.py`) pin the resulting bytes;
  `tests/test_raffles_pipeline.py` additionally asserts the paper's key
  sentences verbatim, so regenerating the fixtures cannot silently drift from
  the paper wording.

## The glue the paper leaves open

The paper specifies the texts and Algorithm 1, but not the joins. Each choice
below is ours, made once and recorded here.

| the paper says | this implementation does |
|---|---|
| the Judge "uses the same prompt template as the Step by Step prompt template" (F.1: Task Description / Input Metadata / Task Output) | `task_description` = the F.3 Judge text; `input_metadata` = problem line + step-labelled log (+ feedback after round one); `task_output` = the F.3 format block |
| the Judge "receives relevant execution logs τ" | the log is serialized one line per turn as `Step {i} - {agent}: {content}` — the repo's step_by_step serialization, so the 0-based indices the Judge must answer with are visible in the log (the paper's own Chat-LLM example counts steps from 0) |
| Evaluator p gets "the proposed step t, τ, and the rationale r_j^p" — one rationale each (Figure 6 sends each colored block to its own Evaluator) | the `{error_step}` slot carries a JSON object with `agent_name`, `step_number` and **only** criterion p's rationale field |
| "the output of the Evaluators is appended to a memory component H … fed back to the Judge in the subsequent iteration" | a `## Feedback on your previous answers ##` block in `input_metadata`: per prior iteration, the Judge's answer plus each verifier's criterion phrase, confidence and reason |
| "an additional rule-based Evaluator p = 4 to validate whether the proposed candidate step t is consistent with the log τ" | 100 if `0 ≤ t < len(history)` and the agent speaking at `t` equals the candidate agent (case- and whitespace-insensitive), else 0 — with the mismatch spelled out in its feedback reason |
| terminate when "C is greater than a threshold of 350", else iterate up to K; then "the step t that has the highest confidence C in history H" wins | `total > threshold` (strict), `for k in range(max_iters + 1)` — K=0 still runs one full pass, matching the paper's "RAFFLES K=0" rows; best C wins, ties to the **latest** iteration, which saw strictly more feedback |
| the Judge must never output null; "t* = None signals an absence of decisive fault" when the Judge finds none *and* confidence clears the threshold | never exercised here: every trajectory in this repo's corpora is a failed one and the Judge prompt's fallback procedure forces a best guess, so no explicit no-fault path is implemented |
| all models decode greedily (temperature 0; Claude excepted in the paper) | `temperature: 0.0` in the non-reasoning API specs and the vLLM defaults |

Parsing is shared across roles: the one JSON object is read from a fenced
```` ```json ```` block, falling back to the outermost braces. An unparseable
**Evaluator** response scores confidence 0 (an unverifiable argument earns no
confidence — the instruction the prompt itself gives). An unparseable **Judge**
response costs that iteration: no Evaluator round is issued, the Judge is told
its answer was not valid JSON, and the loop moves on; if every round fails the
trajectory ends with a null prediction (counted as wrong by the shared
metrics, like every other baseline's parse failure).

## Infrastructure-level notes (no prompt bytes involved)

1. The three Evaluator prompts are yielded as **one round**, so the API
   backend's thread pool issues them concurrently — the paper's Appendix D
   parallelization — and vLLM batches them across trajectories.
2. `strip_think` runs on every response before parsing, as everywhere else in
   the repo, so reasoning backbones work unchanged.
3. `calls` stores every LLM response with `{iteration, role, criterion}`; the
   rule check is not an LLM call and lives in the `iterations` audit instead.
4. Backend failures follow GUIDE.md: the runner skips the trajectory without
   writing, and a rerun resumes it.
5. `--gt with` (non-paper extension) inserts the prompting baselines' exact
   `The Answer for the problem is: ...` line into the problem statement of
   every prompt, so the with/without pair differs by that line only.

## Tests

`tests/test_raffles_pipeline.py` (CPU-only, keyless): golden-prompt bytes +
paper-phrase pinning, criterion scoping of the Error Step, the loop's control
flow on scripted confidences (early stop, full-K run, best-C/latest-tie
selection, K=0), rule-check cases, judge-failure recovery and the all-failed
null path, GT flag reach, CLI e2e with resume/slice/`--method-dir`, sweep
dry-runs for all three datasets and a dummy e2e, and shared-report
integration.
