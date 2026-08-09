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

- [ ] **2. Smoke test** — 10 trajectories, preview then run, both settings.
  These run all three stages, so on a split setup either do this on the GPU
  machine's clone for the similarity part, or simply run it after step 3:

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
  the `outputs/…` twin must show `true`. Note that with only 10 schemata
  generated so far, retrieval may come up short — that self-corrects after
  step 3.

- [ ] **3. Offline stages, once per dataset** (GT-independent; both settings
  and both detectors share these artifacts).

  Single machine — both stages in order:

  ```bash
  for ds in ww traceelephant correct-error; do
    DATASET=$ds STAGES=schemagen,similarity bash scripts/correct/run.sh
  done
  ```

  Two machines — run these two blocks **in parallel**, they don't depend on
  each other:

  ```bash
  # (3a) API machine — no GPU needed
  for ds in ww traceelephant correct-error; do
    DATASET=$ds STAGES=schemagen bash scripts/correct/run.sh
  done
  # writes artifacts/<ds>/<subset>/schemagen/gpt-4o/<id>.json

  # (3b) GPU machine — no API key needed
  for ds in ww traceelephant correct-error; do
    DATASET=$ds STAGES=similarity GPU=0 bash scripts/correct/run.sh
  done
  # writes artifacts/<ds>/<subset>/similarities/bge-m3.json (+ .meta.json)
  ```

  Then move the similarity artifacts to the API machine — a few MB for
  correct-error, kilobytes elsewhere. Via git if both clones share a remote,
  otherwise directly:

  ```bash
  # on the GPU machine
  git add 'artifacts/*/*/similarities/*' && git commit -m "CORRECT: BGE-M3 similarities" && git push
  # or: rsync -a --include='*/' --include='similarities/**' --exclude='*' artifacts/ user@api-host:/path/to/attrib-prompting/artifacts/
  ```

  Schemata come from `schema_model: gpt-4o` in every config — one generator,
  both detectors, as the paper intends. (To use GPT-5 schemata instead, set
  `schema_model: gpt-5` and rerun 3a; results then land in a separate
  `<subset>/schemagen/gpt-5/` tree.)

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
  answer line and writes to `outputs/`. On the API machine, both stage-1 and
  stage-2 artifacts must be present — if either is missing, `predict` exits
  with the exact command that produces it rather than running a degraded
  prompt.

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
