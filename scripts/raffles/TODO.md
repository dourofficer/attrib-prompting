# RAFFLES sweep — {gpt-4o, gpt-5} × {ww, traceelephant} + gpt-4o × correct-error, both GT settings

**gpt-5 on correct-error is deliberately excluded** (too costly) and already
dropped from `configs/report_correct-error.yaml`, so `--check-only` won't report
it missing. Everything resumes on file existence — rerun after a crash and only
the missing trajectories execute.

Everything is API-only, **no GPU anywhere**, and RAFFLES has no offline stage —
nothing in `artifacts/`, nothing needs faiss or torch.

Volume: 10 (model, dataset, GT) combos over 2,586 trajectories = **5,892
trajectory-runs → ~24k–71k LLM calls** (4 per iteration — one Judge, three
concurrent Evaluators — and 1 to 3 iterations at K=2; early termination decides
where in that range a run lands). Run `ww` and `traceelephant` first —
correct-error is 75% of the total.

| dataset | trajectories | models | runs |
|---|---|---|---|
| ww (126 alg-gen + 58 hand-crafted) | 184 | gpt-4o, gpt-5 | 736 |
| traceelephant (91 magentic + 85 captain; `swe` excluded by config) | 176 | gpt-4o, gpt-5 | 704 |
| correct-error (7 subsets) | 2,226 | **gpt-4o only** | 4,452 |

Reminder: this baseline's default GT setting is **`without`** (the paper
evaluates Who&When without ground truth), so `GT=without` writes the
paper-faithful tree `outputs-nogt/` and `GT=with` writes the extension tree
`outputs/`. The loops below set `GT` explicitly both ways, so the default never
matters mid-sweep.

## Status (2026-08-13)

| stage | state |
|---|---|
| predict | 0 / 5,892 — nothing run yet (dummy-backend e2e only) |
| report | not started |

---

- [ ] **1. Setup** — API machine: `pip install -e ".[api]"` and
  `export OPENAI_API_KEY=sk-...`

- [ ] **2. Smoke test** — ww/hand-crafted × gpt-4o, 2 trajectories, default
  (without-GT) setting, preview first:

  ```bash
  DATASET=ww SUBSET=hand-crafted MODEL=gpt-4o END_IDX=2 DRY_RUN=1 bash scripts/raffles/run.sh
  DATASET=ww SUBSET=hand-crafted MODEL=gpt-4o END_IDX=2 bash scripts/raffles/run.sh
  jq '{gt_in_prompt, predicted_agent, predicted_step, confidence, n_iterations,
       calls: [.calls[] | {iteration, role}]}' \
    outputs-nogt/ww/hand-crafted/gpt-4o/raffles/1.json
  ```

  Expect `gt_in_prompt: false`, a non-null `predicted_step`, and per iteration
  one `judge` call followed by three `evaluator` calls. A null prediction with
  `confidence: null` means every Judge round failed to parse — inspect `calls`
  before scaling up. Gold for trajectory 1 is `WebSurfer @ 12`.

- [ ] **3. Detection — ww + traceelephant**, both models, both GT settings
  (1,440 runs). `MODEL` omitted ⇒ every model in the config; `GT=without`
  writes `outputs-nogt/`, `GT=with` writes `outputs/`:

  ```bash
  for gt in without with; do
    for ds in ww traceelephant; do
      DATASET=$ds GT=$gt bash scripts/raffles/run.sh
    done
  done
  ```

- [ ] **4. Detection — correct-error, gpt-4o only** (4,452 runs, three quarters
  of the sweep). **`MODEL=gpt-4o` is required** — omitting it runs gpt-5 too:

  ```bash
  for gt in without with; do
    DATASET=correct-error MODEL=gpt-4o GT=$gt bash scripts/raffles/run.sh
  done
  ```

- [ ] **5. Check completion, then evaluate.** Every row must read `DONE`; tables
  land in `outputs[-nogt]/<ds>/reports/raffles/`:

  ```bash
  for gt in without with; do
    for ds in ww traceelephant correct-error; do
      python -m baselines.raffles.report --config baselines/raffles/configs/report_${ds}.yaml --gt $gt --check-only
      python -m baselines.raffles.report --config baselines/raffles/configs/report_${ds}.yaml --gt $gt
    done
  done
  ```

- [ ] **6. Commit** — results are part of the repo:

  ```bash
  git add outputs/ outputs-nogt/ && git commit -m "RAFFLES: GPT-4o/GPT-5 results, both GT settings"
  ```

## Watch-outs

- **Context on the long tail.** Nothing truncates. Every call re-reads the full
  log — the paper measures ~9.8k input tokens per iteration on alg-gen and ~41k
  on hand-crafted — and the Judge prompt grows each iteration as prior
  candidates and critiques are appended, so the one 80k-token hand-crafted
  trajectory may exceed gpt-4o's 128k window by iteration 3. Symptom: an id
  that stays missing in `--check-only` across reruns.
- **Failed vs malformed.** A trajectory whose calls exhaust their retries is
  skipped *without writing*, so a rerun resumes it. One whose Judge rounds all
  return unparseable output is written with a null prediction and
  `confidence: null` — that scores as wrong and will not re-run; delete the
  file to retry it. A single bad round is cheaper: the loop skips that
  iteration's Evaluators and the next Judge round usually recovers.
- **Params are per-model and recorded.** gpt-4o sends
  `{max_tokens: 8192, temperature: 0.6}` — note the paper decodes greedily
  (temperature 0.0); 0.6 is this repo's chosen setting. gpt-5 sends
  `{max_completion_tokens: 16384, reasoning_effort: low}` and no
  temperature: reasoning models reject a non-default one, and reasoning tokens
  come out of the cap, so an exhausted cap surfaces as a zero-confidence
  evaluator or a skipped judge iteration rather than an error. Each `_run.json`
  logs exactly what was sent.
- **K=5 is not part of this sweep.** The paper's Who&When SOTA row uses K=5;
  reproduce it later, side by side, without touching these outputs:
  `DATASET=ww MODEL=gpt-4o EXTRA_SET="--set max_iters=5 --set method_dir=raffles.k5" bash scripts/raffles/run.sh`
  (then point a report config's `methods:` at `raffles.k5`).
