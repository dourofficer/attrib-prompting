"""Report for the CORRECT baseline — reuses the prompting report wholesale.

The prompting report is method-list-driven (it reads ``methods`` from its config
and looks for ``predictions_method-{m}.jsonl``), so pointing a config with
``methods: [correct]`` at ``outputs-<ds>/correct`` gives us completion checking
and the per-seed val/test comparison tables next to SVD/CRR for free.

    python -m baselines.correct.report --config baselines/correct/configs/report_ww.yaml
"""
from baselines.prompting.report import main

if __name__ == "__main__":
    main()
