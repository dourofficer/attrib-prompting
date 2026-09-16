# TODO — ErrorProbe paper mode on GPT-4o and GPT-5, both GT settings

Only the paper mode (`errorprobe_paper`) runs on the API models. The two
vendored modes (`truncated`, `backward`) stay local-only: their gpt rows will
read `MISSING` in the reports, which is expected. The local models
(`qwen3.5-9b`, `deepseek-8b`) are complete for every mode the configs enable.
The API specs already sit in `baselines/errorprobe/configs/<ds>-api.yaml`
with their handicap and one-hypothesis paper block, and the report configs
already list both models, so nothing below edits a config.

- [x] **1. Credits.** Confirm working API with one short trace before launching anything:

  ```bash
  export PATH=/root/dataDisk/home/thanhdo/attrib-prompting/.venv/bin:$PATH   # the repo's interpreter
  export OPENAI_API_KEY=sk-...
  DATASET=ww SUBSET=algorithm-generated MODEL=gpt-4o MODE=paper END_IDX=1 bash scripts/errorprobe/run.sh
  ```

  Expect `1/1 files` and a doc at
  `outputs-nogt/ww/algorithm-generated/gpt-4o/errorprobe_paper/1.json` whose
  `calls` runs tagger, dependency, strategist, one investigator, arbiter. A
  `429` in the log means the balance is still empty; the run wrote nothing,
  rerun after fixing it.

- [x] **2. Smoke test, both models, both GT settings** (2 short traces each;
  algorithm-generated traces run 5–15 turns, about 7 calls per trace):

  ```bash
  for M in gpt-4o gpt-5; do
    DATASET=ww SUBSET=algorithm-generated MODEL=$M MODE=paper END_IDX=2 bash scripts/errorprobe/run.sh
    DATASET=ww SUBSET=algorithm-generated MODEL=$M MODE=paper END_IDX=2 GT=with bash scripts/errorprobe/run.sh
  done
  ```

  Check one doc per model: `predicted_step` is an integer (not null),
  `gt_in_prompt` matches the tree, `_run.json` shows `max_hypotheses: 1`
  and `condensed_chars: 10000`, and for gpt-5 `raw` is not empty — an empty
  `raw` means the 4096-token cap was spent on reasoning even at `minimal`
  effort; raise `max_completion_tokens` in the spec before the full run if
  it recurs.

- [x] **3. GPT-4o, paper mode everywhere.** Runs resume, so rerun after any
  crash:

  ```bash
  for gt in without with; do
    for ds in ww traceelephant; do
      DATASET=$ds MODEL=gpt-4o MODE=paper GT=$gt bash scripts/errorprobe/run.sh
    done
  done
  DATASET=correct-error MODEL=gpt-4o MODE=paper bash scripts/errorprobe/run.sh   # no GT only
  ```

  Cost, per trajectory: `2C + 3` calls with one hypothesis, where `C` is one
  chunk per ~10 steps — about 5 on a ten-step trace and 11 on a median
  hand-crafted one. Run `ww` first, then `traceelephant`; `correct-error`
  is 2,226 short trajectories.

- [x] **4. GPT-5, paper mode.** The same loop with `MODEL=gpt-5`, minus
  `correct-error` (excluded for gpt-5 on every baseline; see
  `report_correct-error.yaml`). Its spec sends `reasoning_effort: minimal`
  with a 4096-token completion cap; the `_run.json` records both.

- [x] **5. Check completion and evaluate**, once per dataset per setting:

  ```bash
  for gt in "" "--gt with"; do
    for ds in ww traceelephant correct-error; do
      python -m baselines.errorprobe.report --config baselines/errorprobe/configs/report_${ds}.yaml $gt --check-only
      python -m baselines.errorprobe.report --config baselines/errorprobe/configs/report_${ds}.yaml $gt
    done
  done
  ```

  Every gpt `errorprobe_paper` row that was run must read `DONE`; their
  `errorprobe` and `errorprobe_bt` rows read `MISSING` by design. A few
  `fmt_fail` are normal (the Arbiter occasionally returns no JSON). Tables
  land in `outputs*/<ds>/reports/errorprobe/`.

- [x] **6. Commit the results:**

  ```bash
  git add outputs/ outputs-nogt/ && git commit -m "ErrorProbe paper mode: GPT-4o/GPT-5 results, both GT settings"
  ```

---

## Remaining: ErrorProbe on CORRECT-Error with ground truth (GPT-4o)

- [x] **7. Open models, correct-error with GT (paper mode).** Done 2026-09-14;
  both cells (Qwen3.5-9B 59.57, DeepSeek-8B 40.39) are in the manuscript.

  ```bash
  export PATH=/root/dataDisk/home/thanhdo/attrib-prompting/.venv/bin:$PATH
  for M in qwen3.5-9b deepseek-8b; do
    DATASET=correct-error MODEL=$M MODE=paper GT=with GPU=<g> \
      bash scripts/errorprobe/run.sh 2>&1 | tee logs/open/errorprobe-correct-error-${M}-gt.log
  done
  ```

  2,226 trajectories × 2 models. The qwen3.5-9b config's handicap applies
  (512 new tokens, temperature 1.0, top-p 0.95, 16k window). deepseek-8b
  runs at the config's default (8192 new tokens). Runs resume on file
  existence — rerun after a crash.

- [ ] **8. GPT-4o, correct-error with GT (paper mode).** API only, no GPU:

  ```bash
  export OPENAI_API_KEY=sk-...
  DATASET=correct-error MODEL=gpt-4o MODE=paper GT=with bash scripts/errorprobe/run.sh
  ```

  2,226 trajectories, one hypothesis, handicapped decoding (512 new tokens,
  temperature 1.0, top-p 0.95). This fills the only `--` in the GPT-4o
  block of `tab:gt-full`.

  Do NOT set `OVERWRITE=1`: `outputs/correct-error/*/gpt-4o/` already holds the
  other five baselines' with-GT predictions.

- [ ] **9. Evaluate correct-error with GT:**

  ```bash
  python -m baselines.errorprobe.report --config baselines/errorprobe/configs/report_correct-error.yaml --gt with --check-only
  python -m baselines.errorprobe.report --config baselines/errorprobe/configs/report_correct-error.yaml --gt with
  ```

  Every `errorprobe_paper` row that was run must read `DONE`; `errorprobe`
  and `errorprobe_bt` rows read `MISSING` (the vendored truncated mode was
  not run with GT). Tables land in `outputs/correct-error/reports/errorprobe/`.
---
