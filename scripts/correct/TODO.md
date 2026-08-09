# CORRECT sweep — {with-GT, without-GT} × {gpt-4o, gpt-5} × {ww, traceelephant, correct-error}

Method: `correct` only (schema-guided). Everything resumes on file existence —
rerun any command after a crash and only the missing trajectories execute.

Volume: 2,586 trajectories in scope (ww 184, traceelephant 176, correct-error
2,226) → **2,586 schemagen calls once**, then **10,344 detection calls**
(2 models × 2 GT settings). Run `ww` and `traceelephant` first; `correct-error`
is 86% of the cost.

- [ ] **1. Setup**

  ```bash
  pip install -e ".[api]"
  pip install torch transformers      # stage 2 only (BGE-M3 encoder)
  export OPENAI_API_KEY=sk-...
  ```

- [ ] **2. Smoke test** — 10 trajectories, preview then run, both settings:

  ```bash
  DATASET=ww SUBSET=hand-crafted MODEL=gpt-4o END_IDX=10 DRY_RUN=1 bash scripts/correct/run.sh
  DATASET=ww SUBSET=hand-crafted MODEL=gpt-4o END_IDX=10 bash scripts/correct/run.sh
  DATASET=ww SUBSET=hand-crafted MODEL=gpt-4o END_IDX=10 GT=with bash scripts/correct/run.sh
  ```

  Check one schema and one prediction:

  ```bash
  jq -r '.schema | .[:400]' outputs/ww/hand-crafted/gpt-4o/schemagen/1.json
  jq '{schema_cases, num_schemata, gt_in_prompt, predicted_agent, predicted_step}' \
     outputs-nogt/ww/hand-crafted/gpt-4o/correct/1.json
  ```

  `schema_cases` must be non-empty (k=10 here) and `gt_in_prompt` `false`;
  the `outputs/…` twin must show `true`. Note that with only 10 schemata
  generated so far, retrieval may come up short — that self-corrects after
  step 3.

- [ ] **3. Offline stages, once per dataset** (GT-independent; both settings
  and both detectors share these artifacts):

  ```bash
  for ds in ww traceelephant correct-error; do
    DATASET=$ds STAGES=schemagen,similarity bash scripts/correct/run.sh
  done
  ```

  Schemata come from `schema_model: gpt-4o` in every config — one generator,
  both detectors, as the paper intends. (To use GPT-5 schemata instead, set
  `schema_model: gpt-5` and rerun this step; results then land in a separate
  `<subset>/gpt-5/schemagen/` tree.)

- [ ] **4. Detection — the full grid.** `MODEL` omitted ⇒ every model in the
  config (`gpt-4o`, `gpt-5`):

  ```bash
  for gt in without with; do
    for ds in ww traceelephant correct-error; do
      DATASET=$ds GT=$gt STAGES=predict bash scripts/correct/run.sh
    done
  done
  ```

  `without` is the paper setting and writes to `outputs-nogt/`; `with` adds the
  answer line and writes to `outputs/`.

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
  git add outputs/ outputs-nogt/ && git commit -m "CORRECT: GPT-4o/GPT-5 results, both GT settings"
  ```

  This includes the `schemagen/` schemata and the `_similarities/*.json`, so
  nobody has to install torch or re-pay for stage 1 again.
