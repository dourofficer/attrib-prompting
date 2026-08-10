# Instruction — SOAP vs prompting-baseline comparison table (build in `../attribscope`)

Hand-off spec for adding a comparison table of **all_at_once, step_by_step,
binary_search, SOAP** to `attribscope`, on the *same* seed selection SOAP uses.
The prompting predictions come from this repo (`attrib-prompting`); everything
else already exists upstream in `attribscope/src/`.

Residual cells (CHIEF, CORRECT, RAFFLES, …) stay blank — fine for now.

---

## 0. What already exists — do not rebuild

| piece | where | what it gives you |
|---|---|---|
| seed windows + per-window selection | `src/reports/triples.py` | `tables/325/triples_selection.tsv` (one row per model × subset × window × method, **already including** `All-at-once` / `Step-by-step` / `Binary search` rows) and `triples_summary.tsv` (SOAP-family columns, plus `subset="average"` rows when `average_subsets: true`) |
| the window-choice rule | `src/reports/manuscript.py::pick_window` | per (model, subset): among windows prefer those where the strategy strictly beats `SVD (proj)`, then max step-acc, tiebreak larger diff, then earliest window |
| baseline scoring | `src/reports/baselines.py::baseline_cell` | mean (step, agent) test accuracy over a seed list, `_acc` divisor `len(test_ids)` (missing prediction = wrong) |
| split reproduction | `baselines.py::val_test_ids` + `list_rep_files(reps_root/qwen3.5-9b/<subset>)` | verified id-identical to this repo's `data/<ds>/<subset>/*.json` stems — do **not** change the id source |

So this is mostly **plumbing**, not new evaluation logic: export predictions into
the layout `baseline_cell` expects, give `triples.py` a judge-model axis, and add
one report module that joins baselines to SOAP's chosen window.

---

## 1. Export the prompting predictions into attribscope's layout

`baseline_cell` reads exactly:

```
<outputs-base>/<ds>/baselines/prompting/<model>/<subset>/predictions_method-<method>.jsonl
```

one JSON object per line; only these keys are read:
`id`, `predicted_agent`, `predicted_step`, `gold_agent`, `gold_step`
(`load_predictions` keys by `str(row["id"])`; extra keys are harmless).

This repo stores one JSON **per trajectory** instead, so write a converter
(either side; simplest as a script here that writes into attribscope):

```
attrib-prompting  outputs[-nogt]/<ds>/<subset>/<model>/<method>/<id>.json
attribscope       <outputs-base>/<ds>/baselines/prompting/<model>/<subset>/predictions_method-<method>.jsonl
```

**⚠ The GT tree names are inverted between the two repos.** Get this right or
Table 1 and Table 2 silently swap:

| setting | manuscript table | attrib-prompting | attribscope `<outputs-base>` |
|---|---|---|---|
| without-GT | Table 1 (`tab:main`, the main one) | `outputs-nogt/` | `outputs/` |
| with-GT | Table 2 (`tab:main-gt`) | `outputs/` | `outputs-gt/` |

`outputs-gt/<ds>/baselines/` does not exist yet — create it.

Requirements:
- **Complete subsets only.** `baseline_cell` returns `None` when
  `len(preds) < len(reps_files)`, so a partial export silently blanks the cell.
- Note in `baseline_cell`'s docstring — *"their GT-ness is a corpus property, not
  a knob of the run"* — is now **false**: this repo has an explicit `--gt` flag,
  so the same corpus yields two prediction sets. The tree you export into is what
  now carries GT-ness. Update that docstring.

---

## 2. Give `triples.py` a judge-model axis

`run()` currently loops `for _, mk in MODEL_DISPLAY:` (qwen3.5-9b, deepseek-8b)
and scores baselines with that same `mk`. The prompting baselines now run on
**GPT-4o / GPT-5**, which is independent of SOAP's proxy backbone.

Minimal change that keeps the existing schema:

- Add a protocol-config key, e.g. `baseline_models: [gpt-4o, gpt-5]`
  (default to the `MODEL_DISPLAY` keys so nothing else changes).
- Split the baseline scoring out of the SOAP model loop: for each
  `judge ∈ baseline_models`, each subset, each window, call
  `baseline_cell(cfg, judge, sk, root, method, win, reps_files)` and append a
  `selection` row with **`model = judge`**, `row = <label>`.
- `reps_files` stays `list_rep_files(reps_root / SPLIT_MODEL / sk)` — the split
  id-source is always qwen, independent of who was scored.

Result: `triples_selection.tsv` gains rows like
`gpt-4o | hand-crafted | 13,14,15 | All-at-once | … | step_acc_test | agent_acc_test`
next to the existing `qwen3.5-9b | … | backprop | …` rows. No schema change.

---

## 3. New module `src/reports/comparison.py`

Shape it on `manuscript.py` (same `COLUMNS`, `MODELS`, `TABLES`, `TAG`).

For each GT setting, for each **SOAP backbone** `mk`, for each column
`(label, ds, subset)`:

1. `win = pick_window(triples_summary, mk, subset, step_col, agent_col)` —
   **import it from `manuscript.py`, don't reimplement.** Use
   `--strategy backprop` as the default: the paper's `\soap` row is `backprop`,
   while `manuscript.py` defaults to `succ-near`.
   For the CE column `subset == "average"` (those rows exist because
   `configs/protocol/correct-error.yaml` sets `average_subsets: true`).
2. SOAP cell = `win["step"]`, `win["agent"]`.
3. Baseline cells, **at the same `win["seeds"]` string** — this is the whole
   point of the exercise:
   ```
   sel[(sel.model == judge) & (sel.row == label) &
       (sel.subset == subset) & (sel.seeds == win["seeds"])]
   ```
4. **CE macro-average** (your choice, per the decision): `triples_selection.tsv`
   has no `average` subset for baselines, so average the **7 CE subsets'** cells
   at that same window — unweighted mean over subsets, *not* pooled over
   trajectories. Emit blank if any subset is missing, and say which.

Output one TSV per GT setting, e.g.
`outputs/manuscript-tables/table{1,2}_comparison_<strategy>.tsv`:

```
Backbone      Judge    Method          Metric        WW-AG  WW-HC  CE  TE-Cap  TE-Mag
Qwen3.5-9B    gpt-4o   All-at-once     Step-level    …
Qwen3.5-9B    gpt-4o   Step-by-step    Step-level    …
Qwen3.5-9B    gpt-4o   Binary search   Step-level    …
Qwen3.5-9B    —        SOAP (backprop) Step-level    …
…                                      Agent-level   …
```

Also write a `*_selection.tsv` companion recording, per cell, the window's seeds
and which rows were found/missing — same bookkeeping habit as `manuscript.py`.

---

## 4. The one judgement call you must make first

**SOAP's windows are chosen per (backbone, subset), and the two backbones
disagree.** E.g. WW-AG: qwen picks seeds `2,3,4`, deepseek picks `6,7,8`. The
prompting baselines sit in their own closed-source band with no SOAP counterpart,
so "the same seed selection" is ambiguous. Pick one and state it in the caption:

- **(a) Per-band** — score each baseline once per backbone band, at that band's
  window. Literally "same seeds as the SOAP cell it is compared against", but the
  same GPT-4o baseline then shows two different numbers per column.
- **(b) One fixed window per column** for every row including SOAP. Loses the
  per-cell optimum but is the only variant that answers "were the seeds chosen
  after seeing the results?" — windows are currently argmaxed on **test**, and
  `pick_window` explicitly prefers windows where SOAP beats its own base scorer,
  so scoring baselines on SOAP-optimal seeds is structurally favourable to SOAP.
- **(c) All 20 seeds ± std** for every row — cheapest to defend, but then it is
  not the manuscript's selection.

The table layout above assumes (a) (hence the `Judge` column). For (b)/(c), drop
the per-cell `pick_window` call and feed a fixed seed list to the same join.

---

## 5. Run order

```bash
# 1. export predictions (both GT trees) from attrib-prompting
# 2. regenerate selection tables so baseline rows carry the judge models
python -m src.reports.triples --config configs/protocol/ww.yaml
python -m src.reports.triples --config configs/protocol/traceelephant.yaml
python -m src.reports.triples --config configs/protocol/correct-error.yaml
# 3. the new table
python -m src.reports.comparison --strategy backprop
```

`triples.py` is two-pass by design (before and after `src.rescore.run`); you only
need the post-rescore pass here since the sweeps already exist.

---

## 6. Verification

- **Export fidelity:** for one (ds, subset, model, method), re-score the exported
  JSONL with `baseline_cell` at a fixed seed and compare against this repo's
  `report.py --set seeds=[<seed>]` `{method}_step@1_test` cell — must match to
  the printed precision. This repo's split universe was already verified
  id-identical to `outputs/<ds>/activations/qwen3.5-9b/<subset>/*.safetensors`
  (58 / 304 / 85 stems on ww-HC / CE-arc / TE-Cap).
- **No regression:** qwen/deepseek baseline rows already in
  `triples_selection.tsv` must be byte-identical after the judge-axis change.
- **Window agreement:** the SOAP column of the new table must equal
  `table1_soap_backprop.tsv` cell-for-cell (same `pick_window`, same strategy).
- **CE:** confirm the macro-average is over 7 subsets and that its window came
  from the `subset == "average"` rows.

---

## 7. Gotchas

- **SOAP is `backprop` on disk.** `succ-strong` / `succ-near` are unreported
  variants; `manuscript.py`'s default (`succ-near`) is *not* the paper row.
- **`outputs-gt/correct-error/` has only `activations/` + `attention/`** — no
  scores/rescore/tables, so the CE cell of Table 2 has no SOAP window to join to
  and will stay blank until that extraction lands. (The corpus now *has* answers,
  so the paper's "CORRECT-Error has no gold answers" justification for dashing CE
  in Table 2 is stale.)
- **`outputs-nogt/correct-error/` in this repo is not a true `--gt without` run.**
  Those files are the older empty-GT-corpus runs (`gt_in_prompt` absent in their
  `_run.json`); their prompts contain `The Answer for the problem is: ` with an
  empty value, whereas `--gt without` omits the line entirely. ww and
  traceelephant are clean (`gt_in_prompt: false`). Re-run CE with
  `GT=without OVERWRITE=1` if you want the CE column consistent with the others.
- **Percentages:** the manuscript prints `48.15`, the TSVs store `0.4815`.
- **Numbers are transcribed by hand** into `manuscript/sections/experiments.tex`
  — there is no `\input` of a generated table. Keep the provenance comment there
  pointing at the new TSV.
