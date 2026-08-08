"""Re-parse stored `all_at_once` predictions from their saved raw text — no GPU.

The `predictions_method-all_at_once.jsonl` files keep the model's raw output in
each row, so the agent/step can be re-derived post-hoc. This applies the current
(markdown-tolerant) `parse_all_at_once` to every row, recovering predictions that
the original stricter parse missed (chiefly DeepSeek-R1's bolded labels) without
re-running inference.

Only `all_at_once` is re-parsed:
  * `step_by_step` is kept exactly as the vendored code (its raw is only the
    flagged step, and the yes/no logic is intentionally left literal), and
  * `binary_search` never parses raw (its step comes from the recursion), so
    there is nothing to re-derive.

`raw`, `gold_*`, and all other fields are preserved, so the operation is
idempotent and reproducible. Rebuild the tables afterward (they read these files).

    python -m baselines.prompting.reparse [--pred-root DIR ...] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .methods import parse_all_at_once

DEFAULT_PRED_ROOTS = [
    "outputs-ww/prompting",
    "outputs-correct-error/prompting",
    "outputs-traceelephant/prompting",
]


def reparse_file(path: Path, dry_run: bool) -> tuple[int, int, int, int]:
    """Return (n, changed, recovered, still_null) for one all_at_once jsonl."""
    rows = []
    n = changed = recovered = still_null = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            n += 1
            old_agent, old_step = row.get("predicted_agent"), row.get("predicted_step")
            new_agent, new_step = parse_all_at_once(row.get("raw") or "")
            if (new_agent, new_step) != (old_agent, old_step):
                changed += 1
                if old_step is None and new_step is not None:
                    recovered += 1
            row["predicted_agent"] = new_agent
            row["predicted_step"] = new_step
            if new_step is None:
                still_null += 1
            rows.append(row)

    if not dry_run and changed:
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return n, changed, recovered, still_null


def main() -> None:
    p = argparse.ArgumentParser(prog="baselines.prompting.reparse", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred-root", dest="pred_roots", action="append", default=None,
                   metavar="DIR", help="prediction root(s); default: all three datasets")
    p.add_argument("--dry-run", action="store_true", help="report counts without writing")
    args = p.parse_args()

    pred_roots = args.pred_roots or DEFAULT_PRED_ROOTS
    files = []
    for root in pred_roots:
        files += sorted(Path(root).glob("*/*/predictions_method-all_at_once.jsonl"))

    if not files:
        print(f"No all_at_once prediction files under: {pred_roots}")
        return

    tag = " (dry-run)" if args.dry_run else ""
    print(f"Re-parsing {len(files)} all_at_once file(s){tag}\n")
    print(f"{'model/subset':40} {'n':>5} {'changed':>8} {'recovered':>10} {'still_null':>11}")
    tot = [0, 0, 0, 0]
    for f in files:
        n, changed, recovered, still_null = reparse_file(f, args.dry_run)
        label = f"{f.parent.parent.name}/{f.parent.name}"
        # include dataset root for disambiguation
        ds = f.parents[3].name.replace("outputs-", "")
        print(f"{ds+':'+label:40} {n:>5} {changed:>8} {recovered:>10} {still_null:>11}")
        for i, v in enumerate((n, changed, recovered, still_null)):
            tot[i] += v
    print(f"\n{'TOTAL':40} {tot[0]:>5} {tot[1]:>8} {tot[2]:>10} {tot[3]:>11}")
    if args.dry_run:
        print("\n(dry-run — no files written)")
    else:
        print("\nDone. Rebuild tables to pick up the corrected numbers "
              "(e.g. bash scripts/build_tables.sh).")


if __name__ == "__main__":
    main()
