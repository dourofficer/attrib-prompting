# Full prompting sweep (both GT settings)

- Date: 2026-08-09 (Asia/Shanghai)
- Models: `gpt-4o`, `gpt-5`
- Methods: `all_at_once`, `binary_search`, `step_by_step`
- Datasets: Who&When, TraceElephant, CORRECT-Error (11 subsets total)
- Settings: with ground truth (`outputs/`) and without ground truth (`outputs-nogt/`)
- API concurrency: 64
- Completed experiment cells: **132 / 132**
- Target result JSON files: **31,032 / 31,032**
  - each model x setting combination: 7,758 files
- Invalid JSON files: 0
- Model metadata mismatches: 0
- Empty API call logs: 0

Every GPT-4o/GPT-5 row in the six target completion reports is `DONE`.
`fmt_fail` means that the API returned a complete raw response which did not
match the attribution parser; it is retained and scored as incorrect. There
are 24 such outputs across the full matrix. `no_pred` for `step_by_step` means
that no step was judged erroneous before the trajectory ended; there are 701
such trajectories. Neither category is a missing API result.

Some results inherited from the earlier half-sweep predate the per-record
`gt_in_prompt` field. Their setting remains unambiguous from the output root
and each method directory's `_run.json`; all newly generated records carry the
field, and no stored field has the wrong value.

## Usage and cost for this completion run

The values below are the difference between the account snapshot immediately
before the compatibility smoke tests and a stable snapshot after all missing
cells finished. Smoke-test calls are included. A second snapshot 30 seconds
later was identical.

- Requests: 53,687
- Prompt tokens: 183,857,750
- Completion tokens: 21,847,787
- Cached tokens: 19,525,632
- Quota delta: 174,241,355
- Conversion: quota / 500,000 USD
- Total incremental cost: **US$348.482710**

The earlier half-sweep and its cost are documented in Git history; this value
is only the incremental cost of completing the new both-GT TODO.

## Evaluation reports

Each report directory contains `completion_status.tsv`, per-model/per-subset
`comparison_by_seed.tsv` files, and `summary_mean_over_seeds.tsv`.

- `outputs/ww/reports/`
- `outputs/traceelephant/reports/`
- `outputs/correct-error/reports/`
- `outputs-nogt/ww/reports/`
- `outputs-nogt/traceelephant/reports/`
- `outputs-nogt/correct-error/reports/`

## Server artifacts

- `/mnt/DeepResearch/junlinfang/attrib-prompting-results/both-gt/outputs_both_gt_final.tar.gz`
- `/mnt/DeepResearch/junlinfang/attrib-prompting-results/both-gt/run_ledger.tsv`
- `/mnt/DeepResearch/junlinfang/attrib-prompting-results/both-gt/full_run.log`
- `/mnt/DeepResearch/junlinfang/attrib-prompting-results/both-gt/usage_before.json`
- `/mnt/DeepResearch/junlinfang/attrib-prompting-results/both-gt/usage_after_stable.json`
