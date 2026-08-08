"""Import legacy prediction JSONLs into the per-trajectory output layout.

The pre-standalone pipeline stored one ``predictions_method-<method>.jsonl``
per (model, subset). This converts such a tree into this repo's layout —
one ``outputs/<ds>/<subset>/<model>/<method>/<id>.json`` per trajectory — so
existing (already-billed) results can be inspected and evaluated without
re-running inference.

Imported files carry the legacy row fields plus method/model/backend labels;
they have no ``calls`` log (the legacy format only kept the decisive ``raw``).
Provenance is recorded in each method dir's ``_run.json``.

Usage:
    python scripts/import_legacy_jsonl.py \
        --src ../attribscope/outputs/ww/baselines/prompting \
        --dst outputs/ww [--backend vllm] [--dry-run]

``--src`` must contain ``<model>/<subset>/predictions_method-<method>.jsonl``.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baselines.prompting.runner import OutputWriter  # noqa: E402


def _parse_rows(pf: Path) -> list[dict]:
    """Parse a legacy JSONL, healing rows torn by a literal newline in `raw`."""
    rows, pending = [], ""
    for lineno, line in enumerate(pf.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() and not pending:
            continue
        candidate = f"{pending}\\n{line}" if pending else line
        try:
            rows.append(json.loads(candidate))
            pending = ""
        except json.JSONDecodeError:
            pending = candidate
    if pending:
        raise SystemExit(f"{pf}: unparseable trailing content near line {lineno}")
    return rows


def import_file(pf: Path, dst_root: Path, backend: str, dry_run: bool) -> int:
    model, subset = pf.parent.parent.name, pf.parent.name
    method = pf.stem.removeprefix("predictions_method-")
    method_dir = dst_root / subset / model / method
    rows = _parse_rows(pf)
    if dry_run:
        print(f"  would import {len(rows):>5} rows -> {method_dir}")
        return len(rows)

    writer = OutputWriter(method_dir)
    for row in rows:
        doc = {
            "id": row["id"],
            "filename": row.get("filename"),
            "question_id": row.get("question_id"),
            "method": method,
            "model": model,
            "backend": backend,
            "predicted_agent": row.get("predicted_agent"),
            "predicted_step": row.get("predicted_step"),
            "gold_agent": row.get("gold_agent"),
            "gold_step": row.get("gold_step"),
            "raw": row.get("raw"),
        }
        writer.write(str(row["id"]), doc)
    writer.write_run_config({
        "model": model,
        "method": method,
        "subset": subset,
        "backend": backend,
        "imported_from": str(pf),
        "imported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_total": len(rows),
        "note": "imported from a legacy predictions JSONL; per-call logs unavailable",
    })
    print(f"  imported {len(rows):>5} rows -> {method_dir}")
    return len(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, required=True,
                   help="legacy root: <model>/<subset>/predictions_method-*.jsonl")
    p.add_argument("--dst", type=Path, required=True, help="dataset output root, e.g. outputs/ww")
    p.add_argument("--backend", default="vllm",
                   help="backend label recorded on imported docs (default: vllm)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    files = sorted(args.src.glob("*/*/predictions_method-*.jsonl"))
    if not files:
        raise SystemExit(f"no predictions_method-*.jsonl under {args.src}")
    total = sum(import_file(pf, args.dst, args.backend, args.dry_run) for pf in files)
    print(f"{'would import' if args.dry_run else 'imported'} {total} trajectories "
          f"from {len(files)} files")


if __name__ == "__main__":
    main()
