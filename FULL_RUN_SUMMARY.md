# Full prompting sweep (Apodex)

- Date: 2026-08-09 (Asia/Shanghai)
- Models: `gpt-4o`, `gpt-5`
- Methods: `all_at_once`, `binary_search`, `step_by_step`
- Dataset subsets: 11
- API concurrency: 64
- Completed cells: 66 / 66
- Output JSON files: 15,516 / 15,516 (`gpt-4o`: 7,758; `gpt-5`: 7,758)
- Invalid JSON files: 0
- Empty or unparsed outputs after repair: 0
- Explicit `None` / `N/A` predictions: 8 (`gpt-5`, `all_at_once`; valid responses scored as incorrect)
- Step-by-step trajectories reaching the end without flagging an error: 480

## Usage and cost

The values below are computed from the account usage snapshot immediately before
the first successful compatibility probe and a stable snapshot after all inference
completed. The probe cost is therefore included.

- Requests: 54,076
- Prompt tokens: 185,898,119
- Completion tokens: 24,172,852
- Cached tokens: 18,014,592
- Quota delta: 187,436,426
- Conversion: quota / 500,000 USD
- Total cost: **US$374.872852**

The targeted repair reran 25 empty GPT-5 responses. It added 25 requests and
cost **US$1.554154**. Two additional parenthesized responses were recovered by
re-parsing and required no API calls.

## Server artifacts

- `/mnt/DeepResearch/junlinfang/attrib-prompting-results/full/all_outputs_final.tar.gz`
- `/mnt/DeepResearch/junlinfang/attrib-prompting-results/full/reports.tar.gz`
- `/mnt/DeepResearch/junlinfang/attrib-prompting-results/full/run_ledger.tsv`
- `/mnt/DeepResearch/junlinfang/attrib-prompting-results/full/full_run.log`
- `/mnt/DeepResearch/junlinfang/attrib-prompting-results/full/usage_before_full.json`
- `/mnt/DeepResearch/junlinfang/attrib-prompting-results/full/usage_after_full.json`
