# CHIEF sweep — {gpt-4o, gpt-5} × {ww, traceelephant} + gpt-4o × correct-error, both GT settings

**gpt-5 on correct-error is deliberately excluded** (too costly) and already
dropped from `configs/report_correct-error.yaml`, so `--check-only` won't report
it missing. Everything resumes on file existence — rerun after a crash and only
the missing trajectories execute.

Detection is API-only, **no GPU anywhere**; stage 1 (retrieval) is done and
committed, so nothing needs faiss or torch unless you change the encoder.

Volume: 10 (model, dataset, GT) combos over 2,586 trajectories = **5,892
trajectory-runs → ~35,400 LLM calls** (six per trajectory). Run `ww` and
`traceelephant` first — correct-error is 75% of the total.

| dataset | trajectories | models | runs |
|---|---|---|---|
| ww (126 alg-gen + 58 hand-crafted) | 184 | gpt-4o, gpt-5 | 736 |
| traceelephant (91 magentic + 85 captain; `swe` excluded by config) | 176 | gpt-4o, gpt-5 | 704 |
| correct-error (7 subsets) | 2,226 | **gpt-4o only** | 4,452 |

## Status (2026-08-12)

| stage | state |
|---|---|
| 1. ragprep | ✅ all 11 subsets in `artifacts/<ds>/<subset>/rag/all-MiniLM-L6-v2.json`, committed |
| 2. predict | 0 / 5,892 — the 2 smoke-test files in `outputs/ww/hand-crafted/gpt-4o/chief/` predate the step-hint prompt; **delete that directory** so they re-run |
| report | not started |

---

- [ ] **1. Setup** — API machine: `pip install -e ".[api]"` and
  `export OPENAI_API_KEY=sk-...`

- [x] **2. Smoke test** — ww/hand-crafted × gpt-4o, 2 trajectories; all six
  stages returned parseable output, and with the step hint trajectory 1 predicts
  `WebSurfer @ 12` against gold `WebSurfer @ 12`. Recipe for a fresh subset:

  ```bash
  DATASET=ww SUBSET=hand-crafted MODEL=gpt-4o END_IDX=2 DRY_RUN=1 bash scripts/chief/run.sh
  DATASET=ww SUBSET=hand-crafted MODEL=gpt-4o END_IDX=2 bash scripts/chief/run.sh
  jq '{gt_in_prompt, rag_in_prompt, predicted_agent, predicted_step,
       stages: [.calls[].stage]}' outputs/ww/hand-crafted/gpt-4o/chief/1.json
  ```

  `stages` must be `[1,2,3,4,5,6]` and `predicted_step` non-null — a null step
  means stage 6 answered in a form the vendored bare-integer regex rejects.

- [x] **3. Retrieval** — complete. Only rerun to change the encoder:
  `DATASET=$ds STAGES=ragprep bash scripts/chief/run.sh`

- [ ] **4. Detection — ww + traceelephant**, both models, both GT settings
  (1,440 runs ≈ 8,600 calls). `MODEL` omitted ⇒ every model in the config;
  `GT=with` writes `outputs/`, `GT=without` writes `outputs-nogt/`:

  ```bash
  for gt in with without; do
    for ds in ww traceelephant; do
      DATASET=$ds GT=$gt STAGES=predict bash scripts/chief/run.sh
    done
  done
  ```

- [ ] **5. Detection — correct-error, gpt-4o only** (4,452 runs ≈ 26,700 calls,
  three quarters of the sweep). **`MODEL=gpt-4o` is required** — omitting it
  runs gpt-5 too:

  ```bash
  for gt in with without; do
    DATASET=correct-error MODEL=gpt-4o GT=$gt STAGES=predict bash scripts/chief/run.sh
  done
  ```

- [ ] **6. Check completion, then evaluate.** Every row must read `DONE`; tables
  land in `outputs[-nogt]/<ds>/reports/chief/`:

  ```bash
  for gt in with without; do
    for ds in ww traceelephant correct-error; do
      python -m baselines.chief.report --config baselines/chief/configs/report_${ds}.yaml --gt $gt --check-only
      python -m baselines.chief.report --config baselines/chief/configs/report_${ds}.yaml --gt $gt
    done
  done
  ```

- [ ] **7. Commit** — results are part of the repo:

  ```bash
  git add outputs/ outputs-nogt/ && git commit -m "CHIEF: GPT-4o/GPT-5 results, both GT settings"
  ```

## Watch-outs

- **Context on the long tail.** Nothing truncates (nor does the vendored code).
  Stage-1 prompts are fine — median 12k tokens on ww/hand-crafted, 5 of 58 above
  40k — but stages 5–6 add the causal graph on top of the full history, so the
  one 80k-token hand-crafted trajectory may exceed gpt-4o's 128k window there.
  Symptom: an id that stays missing in `--check-only` across reruns.
- **Failed vs malformed.** A trajectory whose calls exhaust their retries is
  skipped *without writing*, so a rerun resumes it. One whose stage output is
  unparseable is written with a null prediction and `raw` starting
  `[chief-error]` — that scores as wrong and will not re-run; delete the file to
  retry it.
- **Params are per-model and recorded.** gpt-4o sends
  `{max_tokens: 8192, temperature: 0.0}` — the vendored `call_model` value, so
  runs are greedy and reproducible. gpt-5 sends `{max_completion_tokens: 16384}`
  and nothing else: reasoning models reject a non-default temperature, and
  reasoning effort is left at the server default. Each `_run.json` logs exactly
  what was sent.
