# TODO — outstanding Who&When runs (WW-AG, WW-HC), both GT settings, open and closed judges

Written 2026-09-13 for the cost table of the SOAP manuscript (`app:compute`,
`tab:cost`). Scope is the two Who&When subsets, the open judge `qwen3.5-9b`
and the closed judges `gpt-4o` / `gpt-5`. Every cell the manuscript REPORTS
already exists and is complete (126 / 58 files per cell; verified 2026-09-13):

| judge | methods present in `outputs/` and `outputs-nogt/` |
|---|---|
| gpt-4o, gpt-5 | all_at_once, step_by_step, binary_search, correct, chief, raffles, errorprobe_paper |
| qwen3.5-9b, deepseek-8b | the same seven plus errorprobe, errorprobe_bt |

Two facts shape the rest:

* **The "weak" ErrorProbe judge has no directory.** Since 2026-09-06 the handicap
  (512 new tokens, temperature 1.0, top-p 0.95, 16k window clipped from the front)
  is applied in place to the `qwen3.5-9b` spec of
  `baselines/errorprobe/configs/ww.yaml`, so `outputs*/ww/*/qwen3.5-9b/errorprobe_paper/`
  IS the run the manuscript reports (39.68 / 21.84 on WW without GT). The full-budget
  run it replaced was overwritten and survives only in the manuscript's `.tex`
  comments. `soap/scripts/prompting/evaluate.py` no longer lists a `qwen3.5-9b-weak`
  judge.
* **Six with-GT open-judge prompting cells carry no per-call log.** They were imported
  from a legacy JSONL (`_run.json` says `imported_from: .../attribscope/...`): every
  `calls` list is empty in
  `outputs/ww/{algorithm-generated,hand-crafted}/qwen3.5-9b/{all_at_once,step_by_step,binary_search}/`
  (and the `deepseek-8b` twins). Their accuracies stand; they cannot be costed offline.

Token counts and call counts need NO model: `scripts/cost_report.py` replays each
method's generator program with the stored responses, so the prompts are rebuilt
byte-exactly and tokenized offline (`reports/cost_ww.tsv`). What is still missing
is wall-clock for the open judge on a few cells, and the six legacy cells above.

Runs write to a SEPARATE root so the reported predictions are never touched:
`outputs-cost-gt/ww` for with-GT and its sibling `outputs-cost-nogt/ww` for
without-GT (`baselines.shared.common.nogt_root` maps `outputs-<family>-gt` to
`outputs-<family>-nogt`; any other name raises). Both roots are gitignored.

- [ ] **0. Wait for an idle GPU.** The prompting config asks vLLM for
  `gpu_memory_utilization: 0.90` on one H200, and a timing number taken on a shared
  GPU is meaningless. Check `nvidia-smi`; all eight were busy on 2026-09-13.

  ```bash
  export PATH=/root/dataDisk/home/thanhdo/attrib-prompting/.venv/bin:$PATH   # the repo's interpreter
  cd /root/dataDisk/home/thanhdo/attrib-prompting
  mkdir -p logs/cost
  ```

- [ ] **1. The six legacy with-GT cells, re-run with per-call logs** (open judge,
  Who&When methods, with the answer). `step_by_step` batches every step's prompt on
  vLLM, as the reported without-GT runs did; keep it that way so the cells are
  comparable.

  ```bash
  for m in all_at_once step_by_step binary_search; do
    MODEL=qwen3.5-9b DATASET=ww GT=with GPU=<g> \
      EXTRA_SET="--set outputs_root=outputs-cost-gt/ww" \
      bash scripts/prompting/${m}.sh 2>&1 | tee logs/cost/qwen3.5-9b-ww-${m}-gt.log
  done
  ```

  Each log must end with two `wrote outputs-cost-gt/ww/<subset>/qwen3.5-9b/<m>  (N/N files, Xs)`
  lines (126 and 58). Decoding is temperature 0.6, so the accuracies will drift from
  the imported cells; report the drift, do not replace the imported cells.

- [ ] **2. Wall-clock for the cells whose original log is missing** (open judge).
  `logs/open/*.log` already holds complete `wrote ... (N/N files, Xs)` lines for
  qwen3.5-9b without GT on all_at_once, step_by_step, binary_search, correct, chief,
  errorprobe, errorprobe_bt, raffles, and with GT on correct, chief, errorprobe,
  errorprobe_bt, raffles. Missing: **`errorprobe_paper` in both settings**, and the
  three with-GT prompting methods (covered by step 1).

  ```bash
  for gt in without with; do
    DATASET=ww MODEL=qwen3.5-9b MODE=paper GT=$gt GPU=<g> \
      EXTRA_SET="--set outputs_root=outputs-cost-gt/ww" \
      bash scripts/errorprobe/run.sh 2>&1 | tee logs/cost/qwen3.5-9b-ww-errorprobe-paper-${gt}.log
  done
  ```

  The paper mode's `_run.json` must show `max_tokens: 512, temperature: 1.0,
  truncate_prompt_tokens: 15872` (the handicap of the reported run). If CORRECT or
  CHIEF are ever re-timed the same way, add `--set artifacts_root=artifacts/ww` to
  `EXTRA_SET`: their default artifacts root is derived from `outputs_root`.

- [ ] **3. Harvest.** The SOAP side reads the `wrote` lines from `logs/open/` and
  `logs/cost/` and takes the LAST complete run per cell:

  ```bash
  cd /root/dataDisk/home/thanhdo/attrib-prompting && .venv/bin/python scripts/cost_report.py   # reports/cost_ww.tsv
  cd /root/dataDisk/home/thanhdo/soap && ./.venv/bin/python scripts/ablations/c1_cost.py --print-table   # results-ablations/c1_cost.tsv
  ```

  Then paste the printed rows into `tab:cost` in `manuscript/sections/appendix.tex`.
  `cost_report.py` reports `n_mismatch = 0` for every cell it replays exactly; a
  non-zero count means the reconstruction is approximate for that cell.

- [ ] **4. SOAP's own timing (in the SOAP repo).** The appendix's 10 min / 2 h 15 min
  are file-mtime reconstructions of the July run. On an idle GPU:

  ```bash
  cd /root/dataDisk/home/thanhdo/soap
  for s in algorithm-generated hand-crafted; do
    { echo "START $(date +%s)";
      CUDA_VISIBLE_DEVICES=<g> ./.venv/bin/python -m main extract --config configs-main/ww.yaml \
        --set results_base=results-scratch/timing --model qwen3.5-9b --subset $s;
      echo "END $(date +%s)"; } 2>&1 | tee logs/c1_timing_$s.log
  done
  rm -rf results-scratch/timing
  ```

  `c1_cost.py` picks these logs up automatically (`time_source` becomes `timed:...`).

- [ ] **5. Not planned, listed so nobody re-discovers them.**
  * `errorprobe` (truncated) and `errorprobe_bt` on GPT-4o / GPT-5: deliberately
    local-only (`scripts/errorprobe/TODO.md`); the manuscript reports the paper mode.
  * A full-budget Qwen3.5-9B ErrorProbe row (42.33 / 12.64 in the `.tex` comments).
    If it is ever wanted, add a `qwen3.5-9b-full` key to `model_specs` in
    `baselines/errorprobe/configs/ww.yaml` (same `model_path`, default decoding) and run
    `MODE=paper` in both settings; do not overwrite `qwen3.5-9b/`.
  * GPT-4o / GPT-5 re-runs for timing: rejected (API spend); their time cells stay `--`.
  * The `deepseek-8b` twins of step 1: out of the manuscript's cost-table scope.
  * Stray `metrics_by_seed.tsv` files sit at the judge level under
    `outputs/ww/*/gpt-4o|gpt-5/`; harmless, delete when convenient.
