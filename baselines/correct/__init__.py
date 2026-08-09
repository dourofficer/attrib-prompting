"""CORRECT baseline — schema-guided failure attribution (cloud-paper variant).

Three stages: ``schemagen`` (offline error-schema distillation per trajectory),
``similarity`` (BGE-M3 neighbour ranking), ``predict`` (all-at-once detection
with top-k retrieved neighbour schemata). See ``README.md``.
"""
