"""CHIEF baseline — hierarchical causal-graph failure attribution.

Two stages: ``ragprep`` (offline retrieval of decomposition exemplars) and
``predict`` (six sequential LLM calls per trajectory: subtasks, subtask edges,
agents, agent edges, candidate error set, counterfactual attribution). See
``README.md``.
"""
