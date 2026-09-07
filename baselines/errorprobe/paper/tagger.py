"""MAST-guided anomaly tagging: the paper's Phase 1 as a step-level annotator.

The paper (Section 4.1) applies "a taxonomy-conditioned prompt to tag
potential local anomalies" so that "these tags identify where specific
step-level deviations occurred (e.g., a 'Tool output ignored' warning at
Step 5)". The taxonomy is MAST (Cemri et al., arXiv:2503.13657), whose
authors ship an LLM annotator (``llm_judge_pipeline.ipynb``): one user prompt
holding the trace, the mode definitions and worked examples, asking for a
yes/no per failure mode and a one-sentence summary for the whole trace.

This module keeps that annotator's ingredients and rules and moves the unit
from the trace to the step: the same 14 definitions (vendored verbatim under
``vendored/MAST/``), the same "only mark a failure mode if you can provide an
example of it in the trace" discipline, and a JSON answer that names the step
each anomaly appears at. Long traces are tagged in contiguous chunks with
absolute step numbers; every chunk also sees a one-line index of the earlier
steps, so modes that need history (repetition, reset, lost context) stay
detectable.

Mode ids are the vendored ``MAST_FAMILIES`` keys, so a tag's ``mode`` and the
final ``error_family`` share one vocabulary across all three ErrorProbe modes.
One naming conflict is resolved by name, never by number: the MAST
``definitions.txt`` calls 3.2 "Weak Verification" and 3.3 "No or Incorrect
Verification", while the annotator notebook numbers them the other way round.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from ..prompts import MAST_FAMILIES, messages, task_text
from .structure import Step, extract_json_object, render_index_line, render_step

REPO_ROOT = Path(__file__).resolve().parents[3]
MAST_DIR = REPO_ROOT / "vendored" / "MAST" / "taxonomy_definitions_examples"
DEFINITIONS_PATH = MAST_DIR / "definitions.txt"
EXAMPLES_PATH = MAST_DIR / "examples.txt"

SYMPTOM = "The run ended with an incorrect final answer."

# (mode id, MAST number, MAST name) in taxonomy order. Numbers follow
# definitions.txt; names are the join key into it.
MAST_MODE_IDS: list[tuple[str, str, str]] = [
    ("fail_task_spec", "1.1", "Disobey Task Specification"),
    ("fail_role_spec", "1.2", "Disobey Role Specification"),
    ("step_repetition", "1.3", "Step Repetition"),
    ("context_loss", "1.4", "Loss of Conversation History"),
    ("unaware_stopping", "1.5", "Unaware of Termination Conditions"),
    ("conversation_reset", "2.1", "Conversation Reset"),
    ("fail_clarification", "2.2", "Fail to Ask for Clarification"),
    ("task_derailment", "2.3", "Task Derailment"),
    ("info_withholding", "2.4", "Information Withholding"),
    ("ignored_input", "2.5", "Ignored Other Agent's Input"),
    ("reasoning_action_mismatch", "2.6", "Action-Reasoning Mismatch"),
    ("premature_termination", "3.1", "Premature Termination"),
    ("incomplete_verification", "3.2", "Weak Verification"),
    ("incorrect_verification", "3.3", "No or Incorrect Verification"),
]

_BY_NAME = {name.lower(): mode for mode, _, name in MAST_MODE_IDS}
_BY_NUMBER = {num: mode for mode, num, _ in MAST_MODE_IDS}
_HEADER_RE = re.compile(r"^(\d\.\d) ([^:\n]+):\s*(.*)$")


def normalize_mode(value) -> str | None:
    """Accept a mode id, a MAST number (``2.5``, ``FM-2.5``) or a MAST name."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    key = text.lower().replace("-", "_").replace(" ", "_")
    if key in MAST_FAMILIES:
        return key
    num = re.sub(r"^(?:fm|fc)[\s_\-]*", "", text.lower()).strip()
    if num in _BY_NUMBER:
        return _BY_NUMBER[num]
    name = " ".join(text.lower().replace("_", " ").replace("-", " ").split())
    for mast_name, mode in _BY_NAME.items():
        if name == " ".join(mast_name.replace("-", " ").split()):
            return mode
    return None


@lru_cache(maxsize=1)
def load_definitions() -> dict[str, str]:
    """The MAST definitions, one paragraph per mode, keyed by mode id."""
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in DEFINITIONS_PATH.read_text().splitlines():
        m = _HEADER_RE.match(line)
        if m and m.group(2).strip().lower() in _BY_NAME:
            current = _BY_NAME[m.group(2).strip().lower()]
            sections[current] = [m.group(3).strip()] if m.group(3).strip() else []
            continue
        if current is not None:
            sections[current].append(line.rstrip())
    return {mode: "\n".join(lines).strip() for mode, lines in sections.items()}


@lru_cache(maxsize=1)
def load_examples() -> str:
    return EXAMPLES_PATH.read_text().strip()


def render_taxonomy(include_examples: bool = False) -> str:
    defs = load_definitions()
    blocks = []
    for mode, num, name in MAST_MODE_IDS:
        blocks.append(f"- {mode} (MAST {num}, {name}): {MAST_FAMILIES[mode]}\n"
                      f"  Definition: {defs.get(mode, '')}")
    text = "\n".join(blocks)
    if include_examples:
        text += "\n\nWORKED EXAMPLES FROM THE MAST AUTHORS:\n" + load_examples()
    return text


# ── chunking ─────────────────────────────────────────────────────────────────

def chunk_steps(steps: list[Step], chunk_chars: int, step_chars: int) -> list[tuple[int, int]]:
    """Contiguous inclusive ``(lo, hi)`` ranges whose rendered size fits the budget."""
    if not steps:
        return []
    chunks: list[tuple[int, int]] = []
    lo, size = steps[0].index, 0
    for step in steps:
        cost = min(len(step.content), step_chars) + 80
        if size and size + cost > chunk_chars:
            chunks.append((lo, step.index - 1))
            lo, size = step.index, 0
        size += cost
    chunks.append((lo, steps[-1].index))
    return chunks


# ── prompt ───────────────────────────────────────────────────────────────────

def build_tagger_prompt(record: dict, steps: list[Step], chunk: tuple[int, int], *,
                        include_gt: bool = False, include_examples: bool = False,
                        step_chars: int = 1200, context_steps: int = 2) -> str:
    lo, hi = chunk
    n = len(steps)
    earlier = [render_index_line(s) for s in steps if s.index < lo - context_steps]
    context = [render_step(s, step_chars, mark="context only")
               for s in steps if lo - context_steps <= s.index < lo]
    target = [render_step(s, step_chars) for s in steps if lo <= s.index <= hi]

    earlier_text = "\n".join(earlier) if earlier else "(none)"
    context_text = "\n---\n".join(context) if context else "(none)"
    target_text = "\n---\n".join(target)

    return f"""You annotate one failed multi-agent run with the MAST failure taxonomy, step by step.

TASK GIVEN TO THE AGENTS:
{task_text(record, include_gt)}

FAILURE SYMPTOM:
{SYMPTOM}

The trace has {n} steps, numbered from 0. Step numbers below are absolute. You are tagging steps {lo} to {hi}.

FAILURE TAXONOMY (tag with the id before the parenthesis):
{render_taxonomy(include_examples)}

EARLIER STEPS (one line each, for reference only):
{earlier_text}

CONTEXT (the steps just before the ones you tag; do not tag these):
{context_text}

STEPS TO TAG:
{target_text}

RULES:
- Only mark a failure mode at a step if you can quote that step as an example of it. Put the quote in "evidence".
- Tag only steps {lo} to {hi}. One entry per (step, mode) pair. A step may carry several modes; most steps carry none.
- "severity" is your confidence, from 0.0 to 1.0, that the step shows the mode.
- Leave "tags" empty when nothing in these steps shows a failure mode.

RESPOND IN JSON FORMAT:
{{
  "tags": [
    {{"step": <absolute step number>, "mode": "<mode id>", "evidence": "<one sentence quoting the step>", "severity": <0.0 to 1.0>}}
  ],
  "summary": "<one sentence on what these steps show>"
}}

Only output valid JSON, no other text."""


def tagger_messages(*args, **kwargs) -> list[dict]:
    return messages(build_tagger_prompt(*args, **kwargs))


# ── parse ────────────────────────────────────────────────────────────────────

def _clamp(value, default: float = 0.5) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, x))


def parse_tags(raw: str | None, lo: int, hi: int) -> list[dict] | None:
    """Tags inside ``[lo, hi]`` with known modes; ``None`` when the JSON is unreadable."""
    data = extract_json_object(raw)
    if data is None:
        return None
    items = data.get("tags")
    if not isinstance(items, list):
        return []
    seen = set()
    tags = []
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            step = int(float(str(item.get("step")).strip()))
        except (TypeError, ValueError):
            continue
        if step < lo or step > hi:
            continue
        mode = normalize_mode(item.get("mode"))
        if mode is None or (step, mode) in seen:
            continue
        seen.add((step, mode))
        tags.append({"step": step, "mode": mode,
                     "evidence": str(item.get("evidence", "") or "")[:500],
                     "severity": _clamp(item.get("severity"))})
    return tags
