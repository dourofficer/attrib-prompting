"""Report for the CORRECT baseline — reuses the prompting report wholesale.

The shared report is method-list-driven over per-trajectory output files, so a
config with ``methods: [correct, correct_baseline]`` pointed at the same
``pred_root`` gives completion checking and the per-seed val/test tables with
the exact attribscope-mirroring metrics. The GT axis works as everywhere else:
``--gt without`` (or ``gt: without`` in the config — the default here, the
paper setting) evaluates the ``outputs-nogt/`` mirror and reports
``gt_in_prompt = False``.

    python -m baselines.correct.report --config baselines/correct/configs/report_ww.yaml
    python -m baselines.correct.report --config ... --gt with   # with-GT tree
"""
from baselines.prompting.report import main

if __name__ == "__main__":
    main()
