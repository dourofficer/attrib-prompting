"""RAFFLES baseline: iterative Judge-Evaluator fault attribution (Zhu et al., 2026).

A Judge proposes a candidate (agent, step) with one rationale per decisive-fault
criterion; three LLM Evaluators and one rule-based check score those rationales;
the loop repeats with the critiques fed back until confidence clears a threshold
or the iteration budget runs out. See README.md and IMPLEMENTATION.md here.
"""
