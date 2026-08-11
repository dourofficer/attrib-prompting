"""Completion check + per-seed comparison tables for the CHIEF baseline.

CHIEF writes predictions in the exact schema every other baseline uses, and the
shared report is method-list-driven (it reads ``methods`` from the config and
looks under ``<pred_root>/<subset>/<model>/<method>/``). So we reuse it
wholesale: point a ``report_*.yaml`` with ``methods: [chief]`` and
``pred_root: outputs/<ds>`` at it, and every table places CHIEF next to the
other baselines on the identical per-seed val/test splits.

Usage
-----
python -m baselines.chief.report --config baselines/chief/configs/report_ww.yaml [--check-only]
"""
from baselines.prompting.report import main

if __name__ == "__main__":
    main()
