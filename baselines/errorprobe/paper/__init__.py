"""ErrorProbe as the paper describes it, not as the vendored code simplifies it.

This subpackage builds the pipeline of "Towards Self-Improving Error Diagnosis
in Multi-Agent Systems" (Li et al., arXiv:2604.17658, Section 4 and
Algorithm 1) from the paper alone. The authors' vendored reproduction ports
none of it, so nothing here has a vendored counterpart or a parity test;
golden fixtures pin the prompt bytes instead, and every open choice is
recorded in ``baselines/errorprobe/IMPLEMENTATION.md``.

The modules follow the paper's stages in order:

- ``structure`` parses a trajectory into the paper's S_x: one
  (agent, role, action) record per step, with a deterministic action
  classifier written for this repo's corpora.
- ``tagger`` runs the MAST failure taxonomy over the steps as a step-level
  variant of the MAST LLM annotator, producing anomaly tags.
- ``graph`` asks the model for information-flow edges, adds structural edges
  the trace format makes explicit, walks backward from the failure symptom,
  and masks every turn the failure never depended on.
- ``team`` holds the Strategist, Investigator and Arbiter prompts and parsers.
- ``program`` drives the four rounds as one generator the shared runner runs.

Memory (the paper's Phase 3) is deliberately absent. The hooks it would use
stay in place: the Strategist accepts retrieved patterns and renders nothing
when there are none, and the Arbiter emits the signature a memory would key on.
"""
