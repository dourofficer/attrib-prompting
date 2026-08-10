# CORRECT sweep — {with-GT, without-GT} × {gpt-4o, gpt-5} × {ww, traceelephant, correct-error}

Method `correct` only. Everything resumes on file existence: rerun any command
and only missing trajectories execute. 2,586 trajectories (ww 184,
traceelephant 176, correct-error 2,226) → 2,586 schemagen calls once, then
10,344 detection calls. Do correct-error last; it is 86% of the cost.

Stage 1 and 3 need an API key, no GPU. Stage 2 needs a GPU (CPU works), no key.
Stages 1–2 read only `data/`, so they can run concurrently on two machines;
stage 3 needs both their artifacts. Artifacts live in `artifacts/`
(`<ds>/<subset>/schemagen/<schema_model>/` and `.../similarities/<embedder>.json`),
never in `outputs/`.

## Status (2026-08-10)

| stage | done | left |
|---|---|---|
| 1 schemagen (gpt-4o) | ww/algorithm-generated (126) | 2,460 — the other 10 subsets |
| 2 similarity | **all 11 subsets** ✅ | — |
| 3 detection | ww/algorithm-generated × gpt-4o × both GT | everything else |

- [ ] **1. Setup.** API machine: `pip install -e ".[api]"`, `export OPENAI_API_KEY=sk-...`.
  GPU machine: `pip install -e . && pip install torch transformers`.

- [x] **2. Smoke test** — superseded by the completed ww/algorithm-generated run.
  For a fresh subset: add `SUBSET=<name> MODEL=gpt-4o END_IDX=10` (and `DRY_RUN=1`
  to preview) to the step-4 command, then check a doc has non-empty
  `schema_cases` and the expected `gt_in_prompt`.

- [x] **3b. Similarity — complete.** No GPU work left; reruns are a no-op.
  **Commit it — 10 of the 11 are untracked:**

  ```bash
  git add artifacts/ && git commit -m "CORRECT: BGE-M3 trajectory similarities, all subsets"
  ```

  A separate API-machine clone needs these files (stage 3 fails without them):
  `git pull`, or
  `rsync -a --include='*/' --include='similarities/**' --exclude='*' artifacts/ user@api-host:<repo>/artifacts/`.

  Re-verify anytime:

  ```bash
  python - <<'EOF'
  import json, pathlib
  for f in sorted(pathlib.Path("artifacts").glob("*/*/similarities/bge-m3.json")):
      ds, sub = f.parts[1], f.parts[2]
      ids = {p.stem for p in pathlib.Path(f"data/{ds}/{sub}").glob("*.json") if p.stem.isdigit()}
      m = json.loads(f.read_text())
      bad = [k for k, v in m.items() if len(v) != len(ids) - 1 or int(k) in v]
      print(f"{ds}/{sub}: {'OK' if set(m) == ids and not bad else 'INCOMPLETE'} ({len(ids)})")
  EOF
  ```

- [ ] **3a. Schemagen — 2,460 left.** API machine; finished subsets are skipped:

  ```bash
  for ds in ww traceelephant correct-error; do
    DATASET=$ds STAGES=schemagen bash scripts/correct/run.sh
  done
  ```

  **Decide first:** `schema_model` is `gpt-4o`, but the paper used **GPT-5**
  (Appendix A.3, "we first generate all the error schemata using GPT-5 model").
  To match it, set `schema_model: gpt-5` in the three `-api.yaml` before running
  — schemata land in a separate `<subset>/schemagen/gpt-5/` tree, and the 126
  existing gpt-4o ones become a side experiment. Either way one generator serves
  both detectors, as the paper intends.

- [ ] **4. Detection.** `MODEL` omitted ⇒ every model in the config:

  ```bash
  for gt in without with; do
    for ds in ww traceelephant correct-error; do
      DATASET=$ds GT=$gt STAGES=predict bash scripts/correct/run.sh
    done
  done
  ```

  `without` (paper setting) writes to `outputs-nogt/`, `with` to `outputs/`.
  Finish 3a for a subset first: incomplete schemata still run, silently
  retrieving fewer than k.

- [ ] **5. Evaluate** — `--check-only` first; every row must read `DONE`:

  ```bash
  for gt in without with; do
    for ds in ww traceelephant correct-error; do
      python -m baselines.correct.report --config baselines/correct/configs/report_${ds}.yaml --gt $gt --check-only
      python -m baselines.correct.report --config baselines/correct/configs/report_${ds}.yaml --gt $gt
    done
  done
  ```

  For `fmt_fail` rows try `python -m baselines.prompting.reparse` before
  rerunning anything. Tables land in `<gt-root>/<ds>/reports/correct/`.

- [ ] **6. Commit** — `git add artifacts/ outputs/ outputs-nogt/`; the artifacts
  spare everyone else torch and a second stage-1 bill.
