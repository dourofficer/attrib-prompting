# ErrorProbe implementation notes

The details that decide whether a rerun of this baseline is comparing against
the vendored code or against our reading of it. The faithfulness reference is
`vendored/ERRORPROBE/` — the authors' own simplified reproduction ("due to
institutional policy, the original source code for the paper cannot be
released"). Prompt bytes, parsers and control flow are parity-tested by
`tests/test_errorprobe_pipeline.py`, which drives the vendored classes
themselves (`AnalyzerAgent`, `VerifierAgent`, `BackwardTracer`) and asserts
identical prompts, parses and predictions.

## The vendored code vs. the paper

The vendored code is much smaller than the paper's system (README.md has the
side-by-side table). Consequences worth restating here with line references:

- **Memory never reaches a prompt.** `SimplifiedMAS.analyze_trace`
  (`simplified_mas/llm_agents.py:603`) retrieves the top-3 patterns and passes
  them to `_build_analysis_prompt(trace, similar_patterns)` — whose body never
  references the parameter (`llm_agents.py:227`, "Build simple baseline
  prompt - no Phase 1 enhancements"). The backward-tracing path never sees
  memory at all. Memory-on and memory-off therefore produce identical
  predictions, which is why this adaptation skips `train.py`, the EPM manager
  and the VBW/RFI-Δ machinery entirely (user decision, 2026-08-16).
- **`FailureModeDetector` is dead code**: constructed nowhere
  (`llm_agents.py:115`, commented out "Disabled for baseline").
- The vendored double-call of `write_from_hypothesis` (`llm_agents.py:662`
  and again at `:675`) is a memory-only side effect — moot once memory is
  skipped.

## What is transcribed verbatim

- Analyzer prompt (`_build_analysis_prompt`): the last-15-turn window, the
  500-char content cap, the agent-options block (`format_agent_options`,
  `utils/trace_utils.py:139`), the 14-mode MAST taxonomy in FC1→FC2→FC3 order
  with the `ERROR_FAMILIES` descriptions (`core/epm_schema.py:24`), the JSON
  contract. Quirks preserved: the agent list is built from **all** turns while
  the conversation shows only 15; it is an unsorted `set`, so its ordering
  depends on the process's string hashing (PYTHONHASHSEED); a turn's `role`
  is fetched and then unused in the conversation block.
- Analyzer parser (`_parse_llm_response`): fenced-JSON extraction (```` ```json ````,
  then bare ```` ``` ````, then the whole text), the
  `error_step` → `error_step_start`/`error_step_end` →
  `error_start_turn`/`error_end_turn` key cascade, the
  `error_agent`-else-`tool_or_component` agent fallback, and the failure
  sentinel (agent `"unknown"`, span `(0, 0)`, confidence 0).
- Verifier prompt (`_build_verification_prompt`): the −2/+3 context slice
  around the **raw** span, 300-char caps, `<<<ERROR TURN` markers, the raw
  `role` string as the speaker label.
- Verifier parser (`_parse_verification_response`): the evidence dict
  (verified / impact / evidence count / suggested fix), failure → unverified.
- Backward tracing (`backward_tracer.py`, `error_tracing_memory.py`,
  `trace_index.py`): all five prompt kinds (relevance, root cause — with its
  own 7-type + `other` taxonomy, which is *not* MAST — summarize, prune,
  merge, synthesis), the bare `json.loads(response.strip())` parsing with its
  per-call fallbacks (relevance → "relevant iff the turn has an
  action/tool call, score 0.5"; root cause → not-a-cause; synthesis → the top
  hypothesis's fields), the stopping rules (min 10 turns, max 100, confidence
  ≥ 0.75, or two ≥ 0.6 hypotheses on one step), the maintenance schedule
  (summarize when > 5 observations and the cache is stale; prune via LLM when
  > 10 observations at the 15-turn interval or the 50-observation cap; a merge
  probe whose response matters only when it contains "none" — otherwise the
  heuristic same-type-same-agent merge runs regardless), and the fallback
  prediction chain (strongest observation → final turn → nothing). The
  vendored action/tool-call extraction is MetaGPT-shaped (`Editor.`, `Plan.`,
  `TeamLeader.`, ...) and is preserved as-is even though this repo's corpora
  rarely match it — it only feeds the relevance parse-failure fallback.

## The glue the vendored code leaves to us

| the vendored code does | this adaptation does |
|---|---|
| reads the agent from a turn's `name`, falling back to `role` (split before `(`, generic chat roles → `Unknown`) | this repo stores the agent in `role` and its alg-gen data carries `name: "assistant"` (fields swapped vs. upstream), so the name-first read would answer "assistant" everywhere; `methods.strip_names` drops `name` so the vendored **role-fallback branch** is the one exercised. `TraceIndex` (vendored: `name` only, no fallback) gets the same rule via `prompts.agent_from_turn` |
| numbers Analyzer turns 0–14 inside `history[-15:]`, and `evaluate.py` compares that span to the **absolute** gold step — impossible to match on traces longer than 15 turns | `predicted_step = span_start + max(0, len(history) − 15)`, applied uniformly (the parse-failure sentinel included); the raw span is kept in `predicted_span_raw`, and the Verifier still sees the raw span exactly as the vendored code shows it (user decision, 2026-08-16) |
| scores a span-containment "step match" in `evaluate.py` | the shared report's exact-match `step@1` (GUIDE.md: never fork the shared metrics); the span survives in `predicted_span` for anyone who wants the vendored reading |
| returns the parse-failure agent `"unknown"` | written as-is to `predicted_agent` — wrong under the shared metrics, like every baseline's parse failure |
| passes raw JSON step values through; a string step crashes the process at the Verifier's `range(...)` | `coerce_step` (int-of-float-of-str) at the parse boundary; uncoercible values degrade to the vendored parse-failure sentinel (truncated mode) or a null `predicted_step` (backward mode) instead of crashing a resumable run |
| multiplies `analysis confidence × verification_confidence` for the hypothesis confidence | same product, `None` if either value is non-numeric |
| never puts the task answer in a prompt | `--gt without` is the default; `--gt with` appends the repo's standard `The Answer for the problem is: ...` line to the task text (`prompts.task_text`) in every task-carrying prompt — an extension, same as raffles' |
| hardcodes per-call generation params (Analyzer 2000 tokens at T=0.7; Verifier 1500 at T=0.3; tracing calls 200–500 at T=0.7) | the shared backend sends **one** param set per run (GUIDE.md); configs ship the vendored `config.yaml` model block (`temperature: 0.7, max_tokens: 4000`), recorded verbatim in `_run.json`. Infrastructure-level deviation, same class as batching and `strip_think` |
| calls the LLM inline; invoke failures fall back to heuristics | LLM calls are yielded rounds; backend failures abort the trajectory at the runner and a rerun resumes it (GUIDE.md "Resume"). Only the *unparseable-response* fallbacks are reachable, and those are verbatim |
| prints per-trace diagnostics | silent; everything lands in the output document instead |

## Infrastructure-level notes (no prompt bytes involved)

1. `strip_think` runs on every response before parsing, as everywhere else in
   the repo, so reasoning backbones work unchanged.
2. Both modes are generator programs (`methods.py`); the backward walk's
   sub-generators yield `(meta, prompts)` and `methods.py` relays them to the
   runner, logging every response into `calls` as
   `{"role": analyzer|verifier|relevance|root_cause|summarize|prune|merge|synthesize,
   "turn_idx"?, "response"}` — never the prompt.
3. One round is one prompt, so vLLM's `run_batched` still batches across
   trajectories per round, and the API backend runs trajectories concurrently.
4. `Observation`/`Hypothesis` timestamps (`datetime.now()`) are kept for
   fidelity; they surface only inside `memory_summary` in the output document.
   The parity tests compare predictions with timestamps stripped.
5. The two modes write to distinct method directories (`errorprobe`,
   `errorprobe_bt`) under the same output root, so they never collide and the
   shared report scores them as two methods.
6. `qwen3.5-9b` decodes on a handicap under this baseline only (user
   decision, 2026-09-06): 512 generation tokens, temperature 1.0 with top_p
   0.95, a 16k window with prompts clipped at 15,872 tokens, in all three
   modes, plus one hypothesis and a 10k condensed trace in the paper mode.
   The other baselines run the same checkpoint at 2048 tokens, temperature
   0.7 and a 32k window. Every ErrorProbe `qwen3.5-9b` result file records
   the handicap in its `_run.json`; compare across baselines with that in
   mind. `deepseek-8b` runs on the repo-wide settings.

## Paper mode (no vendored counterpart)

`paper/` rebuilds the pipeline the paper describes (Section 4, Algorithm 1)
because the vendored code contains none of it. There is nothing to be
byte-faithful to, so the rules differ from the rest of this file: every prompt
is ours, its provenance is named here, and
`tests/fixtures/errorprobe_paper_golden_prompts.json` pins the bytes so a
change is a visible diff.

### Where each prompt comes from

| prompt | built from |
|---|---|
| tagger (`paper/tagger.py`) | the MAST authors' LLM annotator (`llm_judge_pipeline.ipynb` in github.com/multi-agent-systems-failure-taxonomy/MAST): the same definitions file, vendored verbatim under `vendored/MAST/`, the same "only mark a failure mode if you can provide an example of it in the trace" rule, moved from one verdict per trace to one tag per (step, mode) with the step quoted as evidence, which is what the paper's Section 4.1 describes |
| dependency (`paper/graph.py`) | Section 4.2, "edges represent information flow (e.g., Agent B cites Agent A's output)"; the five edge kinds name the ways a turn can use an earlier one in these corpora |
| Strategist, Investigator, Arbiter (`paper/team.py`) | Section 4.3, role by role; the Investigator's rule "cannot simply state 'the code is wrong'; it must generate a diff or an execution log proving the discrepancy" is quoted in spirit and enforced after parsing too |
| the Arbiter's `signature` | the vendored `core/epm_schema.ErrorSignature` (`tool, api, arg_schema, context_slots, err_family`), so a later memory stage can write straight into the vendored schema |

### The choices the paper leaves open

| the paper says | this implementation does |
|---|---|
| tools: CodeExec re-runs code in a sandbox, LogicProbe checks pre/post-conditions | both are prompts (user decision, 2026-09-05). `code_exec` asks for an interpreter-style execution log, expected vs observed output and a diff; `logic_probe` for pre- and post-conditions each with the step that establishes or contradicts it. Evidence with no log, no conditions and no discrepancy is forced inconclusive (`bare_assertion`) |
| MAST priors enter the backward trace | tags steer only the budget trimming (tagged steps are pinned) and the Strategist prompt. The dependency prompt never sees them, so both round-1 prompt families go out in one round |
| a dependency graph over the whole trace | LLM edges per chunk of about 12k rendered characters (`chunk_chars`), each step clipped to 1200 characters head-and-tail, plus structural edges: code or a tool call to the next execution output, an instruction to the next turn of its addressee, and the last two turns |
| BFS on incoming edges from the symptom | plain adjacency is not an edge, or the search would reach every turn and mask nothing. If the search leaves fewer than 6 steps, adjacent turns are added from the symptom backward until it does (`erf.floor_applied`); `sequential_edges: always` restores full adjacency for an ablation. The task step is always kept |
| "mask unconnected branches" | masked runs render as one line each: `[steps 7-11 masked: 5 turns (WebSurfer x3, Orchestrator x2), no dependency path to the failure]`. Over the 20k-character budget (`condensed_chars`) the per-step clip drops to 600, then steps go by priority: symptom, task and tagged steps stay; the rest leave furthest-from-the-symptom first, ties nearest the start first |
| the trace is "parsed into a structured representation S_x = {(agent_t, role_t, action_t)}" | `paper/structure.py`: agent from the role string; role text from alg-gen's `system_prompt` dict or the hand-crafted team block in the first Orchestrator thought; action from first-match regex rules over the shapes this repo's corpora contain (tool call, code fence, exit code or traceback, WebSurfer observation, ledger JSON, addressed instruction, speaker selection, final answer, termination) |
| agent identity | the repo rule, role split before any parenthesis, with one difference from the vendored extraction the other two modes inherit: only lower-case `human`, `user`, `system`, `assistant` are generic. A capitalised `Assistant` is a real Magentic-One agent and a gold answer in four hand-crafted traces |
| the Strategist "formulates a set of hypotheses" | up to `max_hypotheses` (3), most likely first; an agent not in the run is replaced by the speaker at that step (`agent_fixed`); an unknown probe name becomes the default for the step's action class; duplicates on (step, mode) are dropped |
| the Arbiter "filters out hypotheses where E_h is empty or inconclusive" | the program filters first (`conclusive and supports_hypothesis`); survivors are shown with evidence, the rest as one line each. With no survivor every hypothesis is shown under a heading saying so and the prompt asks for a low-confidence best guess (`arbiter_saw_only_inconclusive`) |
| (unspecified) JSON reading | every answer is read strictly first, then leniently: literal newlines inside strings are allowed and a backslash JSON forbids is doubled. Math-heavy traces make models write LaTeX (`\sqrt`, `\omega`) inside JSON strings; on the first full run the strict reader threw away 608 of 47,298 answers for that reason, including 44 whole diagnoses on math500 |
| (unspecified) parse failures | an unreadable tagger or dependency answer contributes nothing (`parsed: false`); no hypotheses ends the trajectory with a null prediction (`failure_stage: "strategist"`, `raw` = the Strategist's answer); an unreadable Arbiter answer lets the best-evidenced hypothesis stand (`failure_stage: "arbiter"`), mirroring the vendored backward mode's synthesis fallback |
| Table 3: k = 5, τ = 0.7, α = 0.6, d = 1536, T = 0.7 | k, α and d belong to the absent memory stage. τ is applied to nothing but is recorded in `memory_candidate.eligible` (verified ∧ c ≥ 0.7 ∧ novel). T = 0.7 is already this baseline's sampling default |
| the failure symptom is an input | every prompt states "The run ended with an incorrect final answer"; `--gt with` appends the repo's answer line to the task text in all five prompt kinds |
| MAST numbers 3.2 and 3.3 | `definitions.txt` calls 3.2 "Weak Verification" and 3.3 "No or Incorrect Verification"; the annotator notebook numbers them the other way. Modes are joined by name: Weak → `incomplete_verification`, No or Incorrect → `incorrect_verification`. The tagger accepts an id, a number or a name and normalises all three |

The MAST worked examples (`examples.txt`, 67 KB) are vendored but off by
default (`include_mast_examples`): they add about 17k tokens to every tagger
prompt and do not fit the local models' 32k window.

## Tests

`tests/test_errorprobe_paper_pipeline.py` (CPU-only, keyless): golden-prompt
parity for all nine prompt kinds plus phrase pins on the sentences that carry
each contract; the action classifier on one string per real corpus pattern;
addressee, role and task-step extraction; structural edges, edge parsing,
BFS with a masked side branch, the floor, and budget trimming by priority;
every parser's failure path; the program's four rounds, the no-hypothesis,
all-inconclusive and Arbiter-fallback paths, tiny and 60-step traces, the GT
flag reaching every prompt, and the memory hook staying off; CLI e2e with
resume, knobs and `--method-dir`; sweep dry-runs for all six shipped configs
(and the check that none runs the paper mode by default); and the shared
report scoring all three method dirs.

`tests/test_errorprobe_pipeline.py` (CPU-only, keyless): Analyzer/Verifier
prompt and parser parity against the vendored classes (litellm stubbed) over
alg-gen-shaped, hand-crafted-shaped, short and generic-only traces; MAST
taxonomy equality with `core.get_all_mast_families()`; backward-tracing parity
driving the real vendored `BackwardTracer` on two scripted scenarios (a rich
walk exercising summarize/prune/merge and the early stop, and an
all-unparseable walk ending in the fallback prediction); the step remap on
long and short traces; the raw span reaching the Verifier; the parse-failure
sentinel; `strip_names`; GT-flag reach per prompt kind; CLI e2e with
resume/slice/`--mode`/`--method-dir`; sweep dry-runs for all three shipped
configs and a dummy e2e; and shared-report integration over both method dirs.
