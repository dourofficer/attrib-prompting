"""Turn one trajectory record into the three arrays the network eats.

The vendored featurizer (``feature_construction.py:66-101``) walks a log's
history and, per step, embeds the content, embeds the agent, and copies the
step's mistake flag. This module does the same, with three adaptations forced
by this repo's data.

**Which field holds the agent.** The vendored rule is ``step["name"] or
step["role"]`` (``feature_construction.py:91``), and applying it here would
quietly ruin the one subset the paper's Algorithm-Generated model targets.
``data/ww/algorithm-generated`` stores the two fields the other way round from
the upstream Who&When copy it came from: across all 1,099 of its steps,
``name`` is only ever ``user`` or ``assistant`` while ``role`` carries
``Excel_Expert``, ``Computer_terminal`` and 189 other agents. The vendored rule
would hand the model the string ``"assistant"`` as the agent identity for every
model turn, collapsing 191 agents into two. So the field is named explicitly
per source rather than guessed: ``name`` for the vendored training corpus,
which follows the upstream convention, and ``role`` for every corpus in
``data/``, which is this repo's documented agent identity.

**What the agent string looks like.** The agent embedding is a text embedding
of the agent's *name*, so the attention bias ``cos(r_i, r_j)`` only carries
information when the training and test vocabularies overlap. They do not:
the Hand-Crafted training set says ``Orchestrator``, while
``data/ww/hand-crafted`` says ``Orchestrator (thought)`` and
``Orchestrator (-> WebSurfer)``. Normalizing collapses those variants back onto
the string the model was trained against, and is the default for that reason;
``raw`` keeps the vendored behaviour.

**Where the answer goes under ``--gt with``.** StepFinder has no question row —
the model sees only ``history`` — so the task's answer is appended to the first
step's content. In ``ww/hand-crafted`` and ``correct-error`` that step is the
user turn carrying the question; in ``traceelephant`` it is an orchestrator
turn or a tool call, so the placement is a convention rather than a natural
home. It is consistent, and it is the only placement that leaves every other
step's vector untouched — which is what lets a with-GT run reuse the
without-GT cache and re-encode a single row.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from baselines.shared.common import standardize_role
from stepfinder.encode import AGENT_DIM, CONTENT_DIM, Encoder

# The line every prompting baseline in this repo interpolates.
GT_LINE = "The Answer for the problem is: {ground_truth}"

# Which key holds the agent, per source. Not a fallback chain: see the module
# docstring for what a fallback would do to ww/algorithm-generated.
AGENT_FIELD_REPO = "role"        # every corpus under data/
AGENT_FIELD_VENDORED = "name"    # vendored/StepFinder/data/*/train


def agent_text(step: dict, field: str = AGENT_FIELD_REPO, normalize: str = "standardize") -> str:
    """The agent identity string for one step."""
    agent = str(step.get(field) or "").strip()
    if normalize == "standardize":
        return standardize_role(agent)
    if normalize == "raw":
        return agent
    raise ValueError(f"normalize must be 'raw' or 'standardize', got {normalize!r}")


def content_text(step: dict, index: int, include_gt: bool, ground_truth: str) -> str:
    """One step's text, with the answer appended to the first step under ``--gt with``."""
    content = str(step.get("content", "") or "")
    if include_gt and index == 0 and str(ground_truth or "").strip():
        return content + "\n" + GT_LINE.format(ground_truth=ground_truth)
    return content


def label_vector(gold_step, num_steps: int) -> Optional[np.ndarray]:
    """One-hot over steps, or ``None`` when the record cannot supervise.

    ``None`` is not the same as an all-zero vector, and the difference matters:
    ``compute_loss`` takes ``targets.argmax(dim=1)`` (``model.py:336``), which
    on an all-zero row silently returns 0 and teaches the model that step 0 was
    the culprit. Two ``traceelephant/magentic`` records have a ``mistake_step``
    past the end of their history, so callers must drop a ``None`` from any
    training set — while still scoring the trajectory, which needs no label.
    """
    try:
        index = int(gold_step)
    except (TypeError, ValueError):
        return None
    if not 0 <= index < num_steps:
        return None
    labels = np.zeros(num_steps, dtype=np.int64)
    labels[index] = 1
    return labels


def build_features(
    record: dict,
    encoder: Encoder,
    *,
    agent_field: str = AGENT_FIELD_REPO,
    agent_normalize: str = "standardize",
    include_gt: bool = False,
    content_dim: int | None = CONTENT_DIM,
    agent_dim: int | None = AGENT_DIM,
) -> dict:
    """The vendored three arrays, plus the provenance needed to audit them.

    The extra keys are ignored by ``collate.sequence_collate_fn``, which reads
    only the three the vendored code produced — so a payload built here can be
    fed straight to the vendored trainer in a parity test.
    """
    history = record.get("history") or []
    if not history:
        width = encoder.width() if (content_dim is None or agent_dim is None) else 0
        return {
            "content_features": np.zeros((0, content_dim if content_dim is not None else width), dtype=np.float32),
            "agent_features": np.zeros((0, agent_dim if agent_dim is not None else width), dtype=np.float32),
            "mistake_labels": np.zeros(0, dtype=np.int64),
            "id": record.get("id"),
            "num_steps": 0,
            "gold_step": record.get("gold_step"),
            "step_agents": [],
            "n_empty_content": 0,
            "n_truncated_steps": 0,
        }

    ground_truth = record.get("ground_truth", "")
    before = encoder.n_truncated
    content_features, agent_features, agents, n_empty = [], [], [], 0

    for i, step in enumerate(history):
        content = content_text(step, i, include_gt, ground_truth)
        agent = agent_text(step, agent_field, agent_normalize)
        if not content.strip():
            n_empty += 1
        content_features.append(encoder.embed(content, content_dim))
        agent_features.append(encoder.embed(agent, agent_dim))
        agents.append(agent)

    labels = label_vector(record.get("gold_step"), len(history))
    return {
        "content_features": np.stack(content_features, axis=0),
        "agent_features": np.stack(agent_features, axis=0),
        # The vendored key. All-zero when the gold step is unusable; callers
        # read `trainable` rather than inferring it from the array.
        "mistake_labels": labels if labels is not None else np.zeros(len(history), dtype=np.int64),
        "trainable": labels is not None,
        "id": record.get("id"),
        "num_steps": len(history),
        "gold_step": record.get("gold_step"),
        "question": record.get("question", ""),
        "step_agents": agents,
        "n_empty_content": n_empty,
        "n_truncated_steps": encoder.n_truncated - before,
        "gt_in_prompt": bool(include_gt),
        "agent_normalize": agent_normalize,
    }


def derive_gt_features(payload: dict, record: dict, encoder: Encoder,
                       content_dim: int | None = CONTENT_DIM) -> dict:
    """A with-GT payload from a without-GT one, re-encoding a single row.

    Only the first step's content changes between the two settings, so the
    other T-1 content vectors and every agent vector are copied. Across the
    three corpora that is 2,630 forward passes instead of 36,363.
    """
    history = record.get("history") or []
    if not history:
        return dict(payload, gt_in_prompt=True)
    text = content_text(history[0], 0, True, record.get("ground_truth", ""))
    content = payload["content_features"].copy()
    content[0] = encoder.embed(text, content_dim)
    return dict(payload, content_features=content, gt_in_prompt=True)
