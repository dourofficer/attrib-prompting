"""Score StepFinder with the repo's shared protocol, plus one extra table.

The vendored-trained family needs nothing special: it predicts every
trajectory, so ``baselines.prompting.report`` reads it exactly as it reads a
prompting baseline, over the same seeded splits. Routing through that report
rather than the paper's own evaluation is what makes the comparison mean
anything.

The in-corpus family needs one thing more. ``stepfinder.e7`` was trained on
seed 7's training partition, so it may only be scored on seed 7's val and test
ids — but the shared report scores every listed method on every listed seed,
counting a missing prediction as wrong (``report.py:89-99, 205-229``). Nineteen
of each method's twenty rows are therefore meaningless, and the mean over them
is worse than meaningless. Rather than fork the shared rules, this module runs
them untouched and then writes ``diagonal.tsv``, which keeps only the cell
where the method's seed and the split's seed agree.

Usage::

    python -m stepfinder.report --config baselines-rp/stepfinder/configs/report_ww.yaml
    python -m stepfinder.report --config .../report_ww_incorpus.yaml --diagonal
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from baselines.prompting.report import main as shared_main

DIAGONAL_METHOD = re.compile(r"\.e(\d+)$")


def write_diagonal(out_root: Path) -> list[Path]:
    """Reduce each ``comparison_by_seed.tsv`` to its meaningful cells.

    For every ``<method>.e<s>`` column, keep the row where ``seed == s``. The
    result has one row per seed and the columns a reader actually wants, plus a
    mean over the diagonal.
    """
    import pandas as pd

    written = []
    for path in sorted(out_root.rglob("comparison_by_seed.tsv")):
        df = pd.read_csv(path, sep="\t")
        if "seed" not in df.columns:
            continue
        methods = {}
        for col in df.columns:
            m = DIAGONAL_METHOD.search(col.split("_")[0])
            if m:
                methods.setdefault(int(m.group(1)), col.split("_")[0])
        if not methods:
            continue

        rows = []
        for seed, method in sorted(methods.items()):
            match = df[df["seed"] == seed]
            if match.empty:
                continue
            row = {"seed": seed, "method": method, "n_val": int(match["n_val"].iloc[0]),
                   "n_test": int(match["n_test"].iloc[0])}
            for metric in ("step@1_val", "step@1_test", "agent@1_val", "agent@1_test"):
                col = f"{method}_{metric}"
                if col in df.columns:
                    row[metric] = float(match[col].iloc[0])
            rows.append(row)
        if not rows:
            continue

        out = pd.DataFrame(rows)
        means = out.drop(columns=["method", "seed"]).mean(numeric_only=True).to_dict()
        out = pd.concat(
            [out, pd.DataFrame([{"seed": "mean", "method": "diagonal", **means}])],
            ignore_index=True,
        )

        dest = path.parent / "diagonal.tsv"
        out.to_csv(dest, sep="\t", index=False)
        print(f"wrote {dest}")
        written.append(dest)
    return written


def main() -> None:
    p = argparse.ArgumentParser(prog="stepfinder.report", add_help=False)
    p.add_argument("--diagonal", action="store_true",
                   help="After the shared report, reduce `.e<seed>` methods to their "
                        "own seed's row (the in-corpus family).")
    args, rest = p.parse_known_args()

    sys.argv = [sys.argv[0], *rest]
    shared_main()

    if args.diagonal and "--check-only" not in rest:
        import yaml
        cfg_path = None
        for i, token in enumerate(rest):
            if token == "--config" and i + 1 < len(rest):
                cfg_path = rest[i + 1]
        if cfg_path is None:
            raise SystemExit("--diagonal needs --config to find the report's out_root")
        cfg = yaml.safe_load(Path(cfg_path).read_text())
        out_root = Path(cfg["out_root"])
        gt = cfg.get("gt", "with")
        if "--gt" in rest:
            gt = rest[rest.index("--gt") + 1]
        if gt == "without":
            from baselines.shared.common import nogt_root
            out_root = Path(nogt_root(str(out_root)))
        write_diagonal(out_root)


if __name__ == "__main__":
    main()
