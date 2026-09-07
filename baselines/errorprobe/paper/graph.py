"""Symptom-driven backward tracing: the paper's Phase 2 (Section 4.2).

The paper builds a dependency graph whose edges are information flow ("Agent
B cites Agent A's output"), runs breadth-first search on incoming edges from
the failure symptom to find the Effective Receptive Field of the error, and
masks every branch the failure never depended on. The condensed trace x' is
what the diagnosis team reads.

This module does that in three parts:

1. **LLM edges.** One prompt per chunk of steps asks which earlier steps each
   step's content uses. The model sees every earlier step as a one-line
   index and the chunk's steps in full.
2. **Structural edges**, always on, from what the trace format makes
   explicit: code or a tool call to the execution output that follows it, an
   instruction to the turn of the agent it addresses, and the last two turns
   (so the symptom is never isolated). Plain adjacency is *not* an edge: with
   it the search reaches every turn and nothing is ever masked. It serves as
   a floor instead, added only when the graph leaves the search with too few
   steps.
3. **Search and masking.** BFS from the last step over incoming edges; steps
   outside the reachable set render as one masked line per gap. When x' still
   exceeds its budget, steps are dropped by priority: the symptom, the task
   statement and every tagged step stay; the rest go furthest-from-the-symptom
   first. That is how the MAST tags act as priors on the trace
   (Algorithm 1's ``BackwardTrace(x, MastPriors)``).
"""
from __future__ import annotations

from collections import Counter, deque

from ..prompts import messages, task_text
from .structure import Step, clip, extract_json_object, render_index_line, render_step, step_label

EDGE_KINDS = ("cites", "executes", "responds_to", "verifies", "instructs")


# ── LLM edges ────────────────────────────────────────────────────────────────

def build_dependency_prompt(record: dict, steps: list[Step], chunk: tuple[int, int], *,
                            include_gt: bool = False, step_chars: int = 1200) -> str:
    lo, hi = chunk
    n = len(steps)
    earlier = [render_index_line(s) for s in steps if s.index < lo]
    target = [render_step(s, step_chars) for s in steps if lo <= s.index <= hi]
    earlier_text = "\n".join(earlier) if earlier else "(none: this chunk starts at step 0)"
    target_text = "\n---\n".join(target)

    return f"""You map information flow in one failed multi-agent run.

TASK GIVEN TO THE AGENTS:
{task_text(record, include_gt)}

The trace has {n} steps, numbered from 0. Step numbers are absolute. You are mapping steps {lo} to {hi}.

EARLIER STEPS (one line each):
{earlier_text}

STEPS TO MAP:
{target_text}

For each step from {lo} to {hi}, list the earlier steps whose content it uses: a step it quotes or builds on ("cites"), code or a call it runs ("executes"), a message it answers ("responds_to"), a result it checks ("verifies"), or an instruction it carries out ("instructs"). An edge goes from the earlier step to the later one. Never list a step merely because it comes before; list it only when the later step actually uses what it produced.

RESPOND IN JSON FORMAT:
{{
  "edges": [
    {{"from": <earlier step>, "to": <later step, {lo} to {hi}>, "kind": "cites|executes|responds_to|verifies|instructs", "why": "<short reason>"}}
  ]
}}

Only output valid JSON, no other text."""


def dependency_messages(*args, **kwargs) -> list[dict]:
    return messages(build_dependency_prompt(*args, **kwargs))


def parse_edges(raw: str | None, lo: int, hi: int, n: int) -> list[dict] | None:
    """Well-formed edges into ``[lo, hi]``; ``None`` when the JSON is unreadable."""
    data = extract_json_object(raw)
    if data is None:
        return None
    items = data.get("edges")
    if not isinstance(items, list):
        return []
    seen = set()
    edges = []
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            src = int(float(str(item.get("from")).strip()))
            dst = int(float(str(item.get("to")).strip()))
        except (TypeError, ValueError):
            continue
        if not (0 <= src < dst < n) or not (lo <= dst <= hi) or (src, dst) in seen:
            continue
        seen.add((src, dst))
        kind = str(item.get("kind", "") or "").strip().lower()
        edges.append({"from": src, "to": dst,
                      "kind": kind if kind in EDGE_KINDS else "cites",
                      "source": "llm", "why": str(item.get("why", "") or "")[:200]})
    return edges


# ── structural edges ─────────────────────────────────────────────────────────

def structural_edges(steps: list[Step]) -> list[dict]:
    edges: list[dict] = []
    n = len(steps)
    for step in steps:
        i = step.index
        if step.action in ("code", "tool_call"):
            for later in steps[i + 1:]:
                if later.action in ("code", "tool_call"):
                    break
                if later.action == "execution_output":
                    edges.append({"from": i, "to": later.index, "kind": "executes",
                                  "source": "structural", "why": "code or call followed by its output"})
                    break
        if step.addressee and step.action in ("instruction", "speaker_selection"):
            for later in steps[i + 1:]:
                if later.agent == step.addressee:
                    edges.append({"from": i, "to": later.index, "kind": "instructs",
                                  "source": "structural", "why": f"addressed to {step.addressee}"})
                    break
    if n >= 2:
        edges.append({"from": n - 2, "to": n - 1, "kind": "responds_to",
                      "source": "structural", "why": "the symptom answers the turn before it"})
    return edges


def sequential_edges(n: int) -> list[dict]:
    return [{"from": i - 1, "to": i, "kind": "responds_to", "source": "structural",
             "why": "adjacent turns"} for i in range(1, n)]


def merge_edges(*groups: list[dict]) -> list[dict]:
    seen: dict[tuple[int, int], dict] = {}
    for group in groups:
        for e in group:
            key = (e["from"], e["to"])
            if key not in seen:
                seen[key] = e
    return sorted(seen.values(), key=lambda e: (e["to"], e["from"]))


# ── search ───────────────────────────────────────────────────────────────────

def _bfs(edges: list[dict], start: int) -> dict[int, int]:
    incoming: dict[int, list[int]] = {}
    for e in edges:
        incoming.setdefault(e["to"], []).append(e["from"])
    depth = {start: 0}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for src in incoming.get(node, ()):
            if src not in depth:
                depth[src] = depth[node] + 1
                queue.append(src)
    return depth


def effective_receptive_field(edges: list[dict], n: int, *, task_step: int = 0,
                              min_erf_steps: int = 6) -> tuple[list[int], dict[int, int], bool]:
    """BFS from the symptom; returns (steps, depth by step, whether the floor kicked in)."""
    if n == 0:
        return [], {}, False
    symptom = n - 1
    depth = _bfs(edges, symptom)
    floor_applied = False
    floor = min(min_erf_steps, n)
    if len(depth) < floor:
        floor_applied = True
        extra = list(edges)
        cursor = symptom
        while len(depth) < floor and cursor > 0:
            extra.append({"from": cursor - 1, "to": cursor, "kind": "responds_to",
                          "source": "structural", "why": "floor"})
            cursor -= 1
            depth = _bfs(extra, symptom)
    if task_step not in depth:
        depth[task_step] = max(depth.values()) + 1 if depth else 0
    return sorted(depth), depth, floor_applied


# ── condensation ─────────────────────────────────────────────────────────────

def _gap_line(steps: list[Step], a: int, b: int, reason: str) -> str:
    who = Counter(s.agent for s in steps[a:b + 1])
    names = ", ".join(f"{agent} x{k}" for agent, k in who.most_common())
    count = b - a + 1
    span = f"step {a}" if a == b else f"steps {a}-{b}"
    what = ("no dependency path to the failure" if reason == "erf"
            else "dropped to fit the budget")
    return f"[{span} masked: {count} turn{'s' if count > 1 else ''} ({names}), {what}]"


def _render(steps: list[Step], keep: set[int], reasons: dict[int, str], step_chars: int) -> str:
    lines = []
    i = 0
    n = len(steps)
    while i < n:
        if i in keep:
            lines.append(render_step(steps[i], step_chars))
            i += 1
            continue
        j = i
        while j + 1 < n and (j + 1) not in keep and reasons.get(j + 1) == reasons.get(i):
            j += 1
        lines.append(_gap_line(steps, i, j, reasons.get(i, "erf")))
        i = j + 1
    return "\n---\n".join(lines)


def condense(steps: list[Step], erf: list[int], depth: dict[int, int], tags: list[dict], *,
             task_step: int = 0, condensed_chars: int = 20000,
             step_chars: int = 1500, min_step_chars: int = 600) -> dict:
    """The condensed trace x': ERF steps in order, everything else masked."""
    n = len(steps)
    if n == 0:
        return {"text": "", "kept": [], "masked": [], "chars": 0, "step_chars": step_chars}
    symptom = n - 1
    keep = set(erf)
    reasons = {i: "erf" for i in range(n) if i not in keep}
    severity = {}
    for t in tags:
        severity[t["step"]] = max(severity.get(t["step"], 0.0), t["severity"])

    chars = step_chars
    text = _render(steps, keep, reasons, chars)
    if len(text) > condensed_chars:
        chars = min_step_chars
        text = _render(steps, keep, reasons, chars)
    if len(text) > condensed_chars:
        pinned = {symptom, task_step} | (set(severity) & keep)
        droppable = sorted(
            (i for i in keep if i not in pinned),
            key=lambda i: (-depth.get(i, 0), i))  # furthest from the symptom first
        # Then the tagged steps, lowest severity first, if still over budget.
        droppable += sorted((i for i in pinned - {symptom, task_step}),
                            key=lambda i: (severity[i], -depth.get(i, 0), i))
        for i in droppable:
            if len(text) <= condensed_chars:
                break
            keep.discard(i)
            reasons[i] = "budget"
            text = _render(steps, keep, reasons, chars)

    masked = []
    i = 0
    while i < n:
        if i in keep:
            i += 1
            continue
        j = i
        while j + 1 < n and (j + 1) not in keep and reasons.get(j + 1) == reasons.get(i):
            j += 1
        masked.append([i, j, reasons.get(i, "erf")])
        i = j + 1
    return {"text": text, "kept": sorted(keep), "masked": masked, "chars": len(text),
            "step_chars": chars}


def incoming_neighbours(edges: list[dict], step: int, limit: int = 4) -> list[int]:
    srcs = sorted({e["from"] for e in edges if e["to"] == step}, reverse=True)
    return srcs[:limit]


def outgoing_outputs(edges: list[dict], step: int) -> list[int]:
    return sorted({e["to"] for e in edges if e["from"] == step and e["kind"] == "executes"})


__all__ = [
    "EDGE_KINDS", "build_dependency_prompt", "dependency_messages", "parse_edges",
    "structural_edges", "sequential_edges", "merge_edges", "effective_receptive_field",
    "condense", "incoming_neighbours", "outgoing_outputs", "clip", "step_label",
]
