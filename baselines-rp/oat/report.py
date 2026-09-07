"""Score OAT's predictions with the repo's shared report.

Step@1 and agent@1 over the seeded val/test splits, identical in every respect
to how the prompting baselines are scored — that is the point of routing
through ``baselines.prompting.report`` rather than the vendored ``evaluate.py``:
a representation-based method and a prompting one have to be measured the same
way before they can be compared.

The paper's own view of the same predictions — precision, recall, F1 and hit
rate over the top-k and conformal *sets*, plus step AUROC and AUPRC — is
``python -m rb_shared.rb_metrics``, which reads the scores stored in the very
same files.
"""

from baselines.prompting.report import main

if __name__ == "__main__":
    main()
