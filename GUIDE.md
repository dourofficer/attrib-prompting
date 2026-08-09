# Baseline adaptation guide

Conventions every baseline in this repo (prompting — done; chief, correct —
upcoming) must follow. The contract covers **I/O, configs, resume, and
evaluation**; the method's internal logic stays free to follow its vendored
implementation faithfully.

## The two rules

1. **Faithful core.** Prompts, parsers and decision rules come verbatim from
   the method's `vendored/` codebase. Every deliberate deviation (batching,
   `strip_think`, retries, ...) is infrastructure-only, documented in the
   baseline's README, and — where feasible — guarded by a parity test that
   drives the vendored code itself (see `tests/test_vendored_parity.py`).
2. **Uniform shell.** Everything around that core — how trajectories are read,
   where outputs land, how runs resume, how results are evaluated — follows the
   conventions below, so all baselines are run and compared the same way.

## Input

Read trajectories with `baselines.prompting.predict.load_records(data_dir)`
(don't reimplement): it yields records with
`id` (filename stem), `filename`, `question_id`, `history`, `question`,
`ground_truth`, `gold_agent`, `gold_step`, skipping empty histories. The agent
identity is `history[t]["role"]`; steps are 0-based; never regenerate `data/`.

## Output — one JSON file per trajectory

```
outputs/<dataset>/<subset>/<model>/<method>/<id>.json    # method = all_at_once, chief, correct, ...
outputs/<dataset>/<subset>/<model>/<method>/_run.json    # run snapshot
```

Required keys per file (extra method-specific keys are fine):

```json
{
  "id": "1", "filename": "1.json", "question_id": "...",
  "method": "chief", "model": "gpt-4o", "backend": "openai",
  "predicted_agent": "WebSurfer",      // str | null
  "predicted_step": 4,                 // int | null  (0-based, no ±1 shifts)
  "gold_agent": "...", "gold_step": "...",   // copied from the record, uncoerced
  "raw": "...",                        // the decisive model response (str | null)
  "calls": [ {"...": "...", "response": "..."} ]   // full response log, issue order
}
```

`calls` entries carry method-specific metadata (prompting: step/round info;
chief: `{"stage": n}`) but never the prompt text — prompts are deterministic
reconstructions from `data/`. Write files **atomically** and only when the
trajectory is complete: use `baselines.shared.runner.OutputWriter`
(`write()` = tmp + rename; `write_run_config()` for `_run.json` with the exact
`request_params` sent, counts, and timestamps).

## Resume — file existence is the ledger

- Before running, drop records whose `<id>.json` already exists
  (`OutputWriter.done_ids()`); print `skip (complete)` when nothing remains.
- `--overwrite` clears the method dir; there is no other state file.
- Granularity is the trajectory: multi-call methods lose in-flight calls on a
  crash and re-run that trajectory — accepted. A trajectory whose API calls
  exhaust retries is logged and **skipped without writing**, so a rerun picks
  it up.

## Execution

Get models through `baselines.shared.backends.get_backend()` — never import
vLLM or an SDK directly. The whole coupling is
`generate(list[message_lists]) -> list[str]`; dispatch on
`backend.prefers_streaming` (False → batch across trajectories, True → complete
trajectories independently so each writes as it finishes).

Preferred shape: express the method as a per-trajectory generator (yield one
round's prompts, receive responses, return the output doc) and reuse
`baselines.shared.runner.run_batched` / `run_streaming` — prompting shows the
pattern in `baselines/prompting/methods.py`. A method may keep its own loop
(e.g. chief's stage-columnar pipeline) as long as the I/O and resume
conventions above hold.

## Configs

Per baseline, mirroring `baselines/prompting/configs/`:

- `configs/<ds>.yaml` — local vLLM models; `configs/<ds>-api.yaml` —
  closed-source APIs. Never mix the two.
- Shared keys: `models`, `subsets`, `data_dir`, `outputs_root`
  (= `outputs/<ds>`), `start_idx`/`end_idx`, `overwrite`, and `model_specs` —
  one spec per model declaring `backend` plus backend-specific fields
  (`model_path`/`tokenizer`/sampling overrides for vllm; `model`/`base_url`/
  `api_key_env`/`params` for openai, with `params` sent to the API verbatim).
- A `sweep.py` iterates the grid and shells out one `predict` per combo,
  supporting `--set k.sub=v` dot-path overrides and `--dry-run`.
- Method-specific knobs (RAG toggles, schema paths, ...) are plain config keys
  — keep them out of `model_specs`.

## Evaluation — don't write your own

The shared report reads any method dir whose files carry the required keys:
add the method to `methods:` in `configs/report_<ds>.yaml` (or reuse
`baselines.prompting.report` like chief/correct already do). Splits come from
`data/` stems + `baselines.shared.common.split_data` with explicit seeds
(ww/traceelephant 1–20, correct-error 1–3); metrics are the shared
`_agent_hit`/`_step_hit` (missing prediction = wrong). Never fork these rules.

## Checklist for a new adaptation

- [ ] Core prompts/parsers verbatim from `vendored/`; deviations documented
- [ ] Reads records via `load_records`; writes per-trajectory files via `OutputWriter`
- [ ] Resumes from existing files; `--overwrite` supported
- [ ] Runs on all backends through `get_backend()`; `_run.json` records what was sent
- [ ] `<ds>.yaml` + `<ds>-api.yaml` configs with `model_specs`; sweep with `--dry-run`
- [ ] Outputs readable by the shared report (`--check-only` shows DONE)
- [ ] Parity + resume tests, CPU-only and keyless; `python -m pytest tests/ -q` green
