# CHIEF — implementation notes

Reference: `vendored/CHIEF/CHIEF.py` (the authors' code) and
`vendored/CHIEF/chief.pdf` (the paper). **The code is the reference**, not the
paper — see "Where the code differs from the paper" below.

## The one prompt deviation: the 0-index sentence

Every stage prompt gains one sentence, appended to the existing count sentence:

```
There are total 29 steps, each entry provides an agent's output. Steps are indexed from 0, so the first entry is step 0.
```

`--step-hint off` (or `step_hint: false` in a config) removes it and restores
the vendored bytes; `tests/test_chief_parity.py` asserts that deleting this
sentence from the default prompt reproduces the vendored bytes character for
character, in all six stages.

**Why.** The vendored prompts state only how *many* steps there are, never where
the numbering starts. Stages 5 and 6 — the ones that produce the prediction —
say merely `step_id: <integer id in conversation>` and `Step Number: (a single
integer step id)`. The convention leaks only through two incidental format
examples in stages 1–2 (`like 0-2`, `[0, 1, 2]`), which govern subtask ranges,
not the answer. The vendored code got away with this because the original
Who&When hand-crafted files carry a per-turn `step` field — `'step0'`,
`'step1'`, … — and the prompt dumps the history with `str()`, so the model reads
the convention off the data. This repo's corpus has no such field, so nothing
tells the model where counting starts, and it counts from 1.

**Evidence.** On ww/hand-crafted with gpt-4o, trajectory 1 (gold `WebSurfer` @
12): the vendored prompt yielded `WebSurfer` @ **13**, and index 13 is
`Orchestrator` while index 12 is `WebSurfer` — the model meant the 13th turn.
With the sentence added it answers **12**, agent and step both correct.

**Why it names no index.** A first draft ended "…and the last is step {n-1}".
On an 86-turn trajectory the model then answered exactly 85 — the number the
sentence had just supplied. A stated index acts as an anchor, not merely a
definition, so the shipped wording gives only the starting point.

This is the only place where the prompts depart from the vendored bytes. It is
a deliberate, flagged, tested exception to the repo's faithfulness rule, taken
because the alternative is measuring a systematic off-by-one instead of the
method.

## Faithfulness notes (the details that bite)

- **All six prompts are byte-identical to the vendored ones** with
  `--step-hint off`; with the default `on` they differ by exactly the sentence
  above and nothing else. `tests/test_chief_parity.py` loads
  `vendored/CHIEF/CHIEF.py`, monkeypatches its `call_model` to capture what each
  `stepN_*` builds, and asserts equality against `stages.build_stepN` — plus
  deep-equality of every parsed structure. Stage 1 also deviates if retrieval is
  switched off (`rag.kb: []`), which drops the retrieved-example section; the
  shipped configs never do that.
- **Retrieval keeps the vendored off-by-one.** `rag_search.py:65` returns
  `combined_sorted[1:top_k]` — it *discards the best hit*. At the vendored
  `top_k=2` that means exactly one injected exemplar, the runner-up. Almost
  certainly a bug upstream; reproduced, not fixed. It doubles as a crude
  self-contamination guard: most Who&When questions appear verbatim in the GAIA
  knowledge base, so the dropped rank-0 hit is often the trajectory's own task.
  The paper claims explicit id-based exclusion of the evaluated task
  (Appendix A) — the code has none, and neither do we.
- **Stage 3 consumes stage 1, not stage 2.** The subtask edges built in stage 2
  never reach stages 3 or 4; they first matter when the graph is assembled for
  stage 5. The dependency order is 1 → {2, 3}, 3 → 4, {1..4} → 5 → 6.
- **Stage parsers are lenient by design — never "fix" them.** They silently drop
  malformed blocks, default missing floats to `0.0`, and truncate the subtask
  list to the shortest parsed field list (`min(len(names), …)`), so one missing
  `Evidence:` line drops every later subtask. Stage 5 searches each field
  un-anchored across the whole step block, so `explanation:` can pick up a
  neighbouring object's value. Tightening any of this changes the method.
- **The `split("(")` agent cleanup is load-bearing.** Stage 6 truncates the
  predicted agent at the first `(`, turning `Orchestrator (thought)` into
  `Orchestrator` — which is what Who&When hand-crafted ground truth stores.
- **`Step Number:` must be a bare integer.** The vendored regex is
  `r"Step Number:\s*([0-9]+)"`; `step 12` or `12 (WebSurfer)` yields no
  prediction, which the report counts as wrong.
- **Indexing is 0-based** — the step index is the position in `history`. No ±1
  shifting anywhere.
- **Sampling.** The vendored `call_model` pins `temperature=0.0` and sets no
  token cap. Local runs are greedy (`temperature 0.0 / top_p 1.0`); API specs
  send exactly their declared `params`.

## Deliberate deviations (all infrastructure-level; prompt bytes unchanged)

1. **Retrieval is precomputed, not inline.** The vendored code searches FAISS
   inside stage 1 of every trajectory. Here `ragprep.py` does it once per subset
   into `artifacts/<ds>/<subset>/rag/<embed_model>.json`, and `predict.py` reads
   the rendered string. Same bytes in the prompt; the detection host needs
   neither faiss, sentence-transformers, nor torch, and reruns are reproducible.
2. **Both GT settings** (GUIDE.md "GT settings"). Every vendored stage prompt
   carries `The correct answer for the problem is: {ground_truth}`, so
   `--gt with` is the vendored default. `--gt without` removes that sentence and
   keeps its trailing blank line, so the problem statement and the conversation
   stay separate paragraphs — the same elision the RAG-off branch performs.
   Note the vendored code reaches the without-GT state *by accident* on Who&When
   hand-crafted: it reads `data.get("ground_truth")`, but that subset stores the
   key as `groundtruth`, so the sentence is silently empty upstream.
3. **`strip_think` before every parse.** Reasoning backbones emit `<think>`
   blocks the vendored parsers never saw; the hardened `methods.strip_think`
   removes them so the verbatim regexes see only the answer.
4. **Per-trajectory output files with the full transcript.** One
   `<id>.json` per trajectory (resume comes free) carrying
   `calls=[{"stage": 1..6, "response": ...}]`. The vendored code wrote one
   append-mode JSONL per run and kept stages 1-5 while dropping stage 6's text.
5. **Retries, concurrency and batching from the shared runner.** The vendored
   code has no retry logic — one API error loses the whole six-stage chain for
   that sample, and the failed sample still counts in the denominator. Here a
   trajectory whose calls exhaust their retries is skipped *without writing*, so
   a rerun resumes it. `run_batched` gives vLLM the columnar batching (stage N
   across all trajectories in one call); `run_streaming` drives whole
   trajectories concurrently on an API backend.
6. **A malformed stage output ends the trajectory with a null prediction** and
   the transcript so far, which the report counts as wrong — matching the
   vendored accounting, where a crashed sample stays in the denominator.
7. **Evaluation via the shared report** (agent@1 + step@1 on per-seed splits)
   instead of the vendored whole-corpus accuracy printout.
8. **Deterministic, sorted input order.** `load_records` sorts; the vendored
   `os.listdir` order is arbitrary, which mattered only for its `--limit`-style
   truncation.

Not reproduced (upstream bugs with no behavioural upside): the Windows
backslash paths in `rag_search.py`, the module-import-time `RAGRetriever()`
construction, `DEBUG_MODE = True` with `DEBUG_SAMPLE_LIMIT = 1` hardcoded at
`CHIEF.py:21-22`, and the unsanitized model name in output filenames.

## Where the code differs from the paper

The vendored implementation is a *reduced* pipeline compared to the paper's
appendix. Per user decision the code is the reference, so nothing below is
implemented — it is recorded so nobody mistakes one for the other.

| paper | vendored code / here |
|---|---|
| Fig 5 — RAG task decomposition | stage 1, and the code is a **superset**: it also asks for `Loop Info` and `Evidence` |
| Fig 7 — OTAR parsing | stage 3, plus a `Data_Flow` section the paper does not show |
| Fig 8 — subtask **and** agent edges in one prompt, carrying `Counterfactual_Patterns: Bias → Anomaly` | **split** into stages 2 and 4, and the counterfactual-pattern block is replaced by `Data_Transfer` + `Failure Modes` |
| Fig 9 — step-edge construction | **absent**; folded into stage 3's `Data_Flow` and stage 2's `Data_Transfer` |
| Fig 10 — oracle synthesis as its own stage (`Goal / Precondition / Key Evidence / Acceptance Criteria`, sequential with global consistency check) | **absent**; stage 1 asks only for a one-line `The Oracle:` |
| Fig 11 — hierarchical backtracking (reverse-topological, binary discrepancy per level) | stage 5 is a **different prompt**: three localization rules (loop reasonableness, data provenance, first-irrecoverable) and rich per-step candidate nodes |
| Fig 13 — counterfactual attribution in four named stages (Local / Planning-Control / Data-Flow / Final Screening) | stage 6 restates the same three rules instead; only the closing answer-format block is shared verbatim |

The paper also reports DeepSeek-V3.2 (thinking) as the base model and averages
three runs; this repo runs GPT-4o and GPT-5 once, greedily.

## Tests

- `tests/test_chief_parity.py` — drives the vendored module itself: byte-identical
  prompts and deep-equal parses for all six stages, the system prompt, the RAG
  block format, DAG assembly, and the without-GT elision.
- `tests/test_chief_pipeline.py` — end-to-end on the dummy backend: the six-call
  transcript, resume, `--gt` threading, RAG injection into stage 1 only, the
  ragprep artifact (skipped without the `[rag]` extra), sweep dry-runs against
  the shipped configs, and the shared report reading chief's output.
