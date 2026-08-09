"""Re-parse stored `all_at_once` predictions from their saved raw text — no GPU.

Every per-trajectory output file keeps the model's raw output, so the
agent/step can be re-derived post-hoc. This applies the current
(markdown-tolerant) `parse_all_at_once` to every `all_at_once` output,
recovering predictions that a stricter parse missed (chiefly DeepSeek-R1's
bolded labels) without re-running inference.

Only `all_at_once` is re-parsed:
  * `step_by_step` is kept exactly as the vendored code (its raw is only the
    flagged step, and the yes/no logic is intentionally left literal), and
  * `binary_search` never parses raw (its step comes from the recursion), so
    there is nothing to re-derive.

`raw`, `calls`, `gold_*`, and all other fields are preserved, so the operation
is idempotent. Rerun `report.py` afterward (it reads these files).

    python -m baselines.prompting.reparse [--pred-root DIR ...] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .methods import parse_all_at_once

DEFAULT_PRED_ROOTS = [
    "outputs/ww",
    "outputs/correct-error",
    "outputs/traceelephant",
    # without-GT mirrors (missing roots are simply skipped)
    "outputs-nogt/ww",
    "outputs-nogt/correct-error",
    "outputs-nogt/traceelephant",
]


def reparse_dir(method_dir: Path, dry_run: bool) -> tuple[int, int, int, int]:
    """Return (n, changed, recovered, still_null) for one all_at_once dir."""
    n = changed = recovered = still_null = 0
    for path in sorted(method_dir.glob("*.json"), key=lambda p: (len(p.stem), p.stem)):
        if not path.stem.isdigit():
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        n += 1
        old = (doc.get("predicted_agent"), doc.get("predicted_step"))
        new_agent, new_step = parse_all_at_once(doc.get("raw") or "")
        if (new_agent, new_step) != old:
            changed += 1
            if old[1] is None and new_step is not None:
                recovered += 1
            doc["predicted_agent"] = new_agent
            doc["predicted_step"] = new_step
            if not dry_run:
                path.write_text(json.dumps(doc, indent=2, ensure_ascii=False),
                                encoding="utf-8")
        if new_step is None:
            still_null += 1
    return n, changed, recovered, still_null


def main() -> None:
    p = argparse.ArgumentParser(prog="baselines.prompting.reparse", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred-root", dest="pred_roots", action="append", default=None,
                   metavar="DIR", help="dataset output root(s); default: all three datasets")
    p.add_argument("--dry-run", action="store_true", help="report counts without writing")
    args = p.parse_args()

    pred_roots = args.pred_roots or DEFAULT_PRED_ROOTS
    # outputs/<ds>/<subset>/<model>/all_at_once/
    dirs = []
    for root in pred_roots:
        dirs += sorted(d for d in Path(root).glob("*/*/all_at_once") if d.is_dir())

    if not dirs:
        print(f"No all_at_once output dirs under: {pred_roots}")
        return

    tag = " (dry-run)" if args.dry_run else ""
    print(f"Re-parsing {len(dirs)} all_at_once dir(s){tag}\n")
    print(f"{'dataset:subset/model':44} {'n':>5} {'changed':>8} {'recovered':>10} {'still_null':>11}")
    tot = [0, 0, 0, 0]
    for d in dirs:
        n, changed, recovered, still_null = reparse_dir(d, args.dry_run)
        label = f"{d.parents[2].name}:{d.parents[1].name}/{d.parents[0].name}"
        print(f"{label:44} {n:>5} {changed:>8} {recovered:>10} {still_null:>11}")
        for i, v in enumerate((n, changed, recovered, still_null)):
            tot[i] += v
    print(f"\n{'TOTAL':44} {tot[0]:>5} {tot[1]:>8} {tot[2]:>10} {tot[3]:>11}")
    if args.dry_run:
        print("\n(dry-run — no files written)")
    else:
        print("\nDone. Rerun `python -m baselines.prompting.report` to pick up "
              "the corrected numbers.")


if __name__ == "__main__":
    main()
