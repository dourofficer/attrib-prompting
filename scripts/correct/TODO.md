# CORRECT sweep — {with-GT, without-GT} × {gpt-4o, gpt-5} × {ww, traceelephant, correct-error}

Method: `correct` only (schema-guided). Everything resumes on file existence —
rerun any command after a crash and only the missing trajectories execute.

Volume: 2,586 trajectories in scope (ww 184, traceelephant 176, correct-error
2,226) → **2,586 schemagen calls once**, then **10,344 detection calls**
(2 models × 2 GT settings). Run `ww` and `traceelephant` first; `correct-error`
is 86% of the cost.

## Which machine runs what

| stage | needs | GPU? | API key? |
|---|---|---|---|
| 1. schemagen | `data/` | no — API models only | yes |
| 2. similarity | `data/` + the BGE-M3 checkpoint | yes (CPU works, slower) | no |
| 3. predict | `data/` + **both** artifacts | no | yes |
| report | `data/` + predictions | no | no |

Stages 1–2 write to **`artifacts/`** (never `outputs/`, which is predictions
only): `artifacts/<ds>/<subset>/schemagen/<schema_model>/` and
`artifacts/<ds>/<subset>/similarities/<embed_model>.json`.

`torch` is imported only by `similarity.py`, and lazily — stages 1 and 3 never
load it. Stages 1 and 2 read nothing but `data/`, so they are independent and
can run **at the same time** on the two machines; only stage 3 needs both.
Splitting the run that way is fully supported — see step 3.

- [ ] **1. Setup** — on the API machine:

  ```bash
  pip install -e ".[api]"
  export OPENAI_API_KEY=sk-...
  ```

  On the GPU machine (this one), only stage 2's dependencies:

  ```bash
  pip install -e . && pip install torch transformers
  ```

- [x] **2. Smoke test** — done, and then some: ww/algorithm-generated ran to
  completion for gpt-4o in both GT settings (126 + 126 predictions), on top of
  its 126 schemata. Kept here as the recipe for a fresh subset:

  ```bash
  DATASET=ww SUBSET=hand-crafted MODEL=gpt-4o END_IDX=10 DRY_RUN=1 bash scripts/correct/run.sh
  DATASET=ww SUBSET=hand-crafted MODEL=gpt-4o END_IDX=10 bash scripts/correct/run.sh
  DATASET=ww SUBSET=hand-crafted MODEL=gpt-4o END_IDX=10 GT=with bash scripts/correct/run.sh
  ```

  Check one schema and one prediction:

  ```bash
  jq -r '.schema | .[:400]' artifacts/ww/hand-crafted/schemagen/gpt-4o/1.json
  jq '{schema_cases, num_schemata, gt_in_prompt, predicted_agent, predicted_step}' \
     outputs-nogt/ww/hand-crafted/gpt-4o/correct/1.json
  ```

  `schema_cases` must be non-empty (k=10 here) and `gt_in_prompt` `false`;
  the `outputs/…` twin must show `true`. On a subset whose schemata are only
  partly generated, retrieval may come up short — that self-corrects once
  step 3a finishes the subset.

- [x] **3b. Similarity — complete for all 11 subsets.** The command that ran:
  ```bash
  for ds in ww traceelephant correct-error; do
    DATASET=$ds STAGES=similarity bash scripts/correct/run.sh
  done
  ```

- [ ] **3a. Schemagen — 2,460 of 2,586 remaining** (ww/algorithm-generated is
  done). API machine, no GPU:

  ```bash
  for ds in ww traceelephant correct-error; do
    DATASET=$ds STAGES=schemagen bash scripts/correct/run.sh
  done
  # writes artifacts/<ds>/<subset>/schemagen/gpt-4o/<id>.json
  ```

- [ ] **4. Detection — the full grid.** `MODEL` omitted ⇒ every model in the
  config (`gpt-4o`, `gpt-5`):

  ```bash
  for gt in without with; do
    for ds in ww traceelephant correct-error; do
      DATASET=$ds GT=$gt STAGES=predict bash scripts/correct/run.sh
    done
  done
  ```

  A subset whose schemata are incomplete will still run, silently retrieving
  fewer than k. Finish 3a for a subset before detecting on it.

- [ ] **5. Check completion, then evaluate** — both settings, per dataset:

  ```bash
  for gt in without with; do
    for ds in ww traceelephant correct-error; do
      python -m baselines.correct.report --config baselines/correct/configs/report_${ds}.yaml --gt $gt --check-only
      python -m baselines.correct.report --config baselines/correct/configs/report_${ds}.yaml --gt $gt
    done
  done
  ```

  Every row must read `DONE`. For unparsed rows (`fmt_fail`) try
  `python -m baselines.prompting.reparse` before rerunning anything. Tables land
  in `outputs/<ds>/reports/correct/` and `outputs-nogt/<ds>/reports/correct/`.

- [ ] **6. Commit** — artifacts and results are part of the repo:

  ```bash
  git add artifacts/ outputs/ outputs-nogt/ && git commit -m "CORRECT: GPT-4o/GPT-5 results, both GT settings"
  ```

  `artifacts/` carries the schemata and the similarity JSONs, so
  nobody has to install torch or re-pay for stage 1 again.
