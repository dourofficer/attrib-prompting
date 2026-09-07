"""Infrastructure shared by the representation-based baselines.

These baselines do not prompt a model, so they cannot reuse the prompting
shell's chat backend or its per-call bookkeeping. What they do share lives
here: an atomic cache for the tensors an extraction stage produces, and the
set-based metrics their papers report on top of the repo's step@1 / agent@1.
"""
