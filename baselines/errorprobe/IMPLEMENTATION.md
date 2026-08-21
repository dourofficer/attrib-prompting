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

## Tests

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
