"""The paper's structural parse S_x: one (agent, role, action) record per step.

The paper (Section 4.1) parses a raw trace into ``{(agent_t, role_t,
action_t)}`` before any model reads it, where action is the semantic kind of
the turn: a natural-language message, a tool call, or an execution output.
This module does that parse with deterministic rules written for the shapes
this repo's corpora actually contain (Who&When, TraceElephant, CORRECT-Error):

- tool calls look like ``[WebSurfer] tool_call web_search(...)``;
- code comes as fenced ``python``/``sh`` blocks;
- execution outputs start with ``exitcode: N`` or ``Code output:``, carry a
  Python traceback, or are a WebSurfer observation ("I typed ... Here is a
  screenshot");
- orchestrator ledgers are JSON whose first key is ``is_request_satisfied``;
- instructions name their addressee in the role (``Orchestrator (-> X)``),
  in ledger JSON (``instruction_or_question``) or as ``Next Agent: X``;
- speaker selections are a bare agent name or ``Next speaker X``.

The fine label is kept everywhere (prompts and the output document) because
the Investigator's tool choice keys on it; the paper's three-way mapping is
``ACTION_KIND``.

Also here: the head-and-tail ``clip`` every prompt uses (so exit codes and
final lines survive truncation), the shared step renderers, and a JSON reader
that never raises.

Agent identity is the repo rule (``history[t]["role"]``, split before any
parenthesis) on a ``strip_names`` view. One difference from the vendored
extraction the other two modes inherit: only the lower-case chat roles
``human``, ``user``, ``system`` and ``assistant`` are generic (the task giver,
not an agent). A capitalised ``Assistant`` is a real team member in the
hand-crafted Who&When traces and a gold answer in four of them, so it keeps
its name.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from baselines.prompting.methods import strip_think

from ..methods import strip_names

GENERIC_ROLES = ("human", "user", "system", "assistant")  # lower case only

# ── action classes ───────────────────────────────────────────────────────────

# The paper's three action kinds, keyed by this module's finer labels.
ACTION_KIND = {
    "instruction": "message",
    "ledger": "message",
    "speaker_selection": "message",
    "message": "message",
    "final_answer": "message",
    "termination": "message",
    "tool_call": "tool_call",
    "code": "tool_call",
    "execution_output": "execution_output",
}

_TOOL_CALL_RE = re.compile(r"^\[[^\]\n]+\]\s*tool_call\s+\w+\(", re.M)
_CODE_FENCE_RE = re.compile(r"```(?:python|py|sh|bash)\b", re.I)
_EXEC_OUTPUT_RE = re.compile(
    r"^exitcode:\s*-?\d+|^Code output:|Traceback \(most recent call last\)", re.M)
_OBSERVATION_RE = re.compile(
    r"^I (?:typed|clicked|scrolled|opened|visited|navigated|pressed|hovered)\b"
    r"|Here is a screenshot|The web browser is open", re.M)
_LEDGER_RE = re.compile(r"^\s*Updated Ledger|^\s*\{\s*\"is_request_satisfied\"", re.S)
_ADDRESSED_ROLE_RE = re.compile(r"\(\s*->\s*([^)]+?)\s*\)")
_INSTRUCTION_JSON_RE = re.compile(r"^\s*\{\s*\"instruction_or_question\"", re.S)
_NEXT_AGENT_RE = re.compile(r"^\s*Next Agent:\s*([A-Za-z_][\w\-]*)", re.M | re.I)
_NEXT_SPEAKER_RE = re.compile(r"^\s*(?:Next speaker:?\s+)?([A-Za-z_][\w\-]*)\s*$")
_NEXT_SPEAKER_JSON_RE = re.compile(r"\"next_speaker\"\s*:\s*(?:\{[^}]*\"answer\"\s*:\s*)?\"([^\"]+)\"")
_FINAL_ANSWER_RE = re.compile(r"^##\s*ANSWER|FINAL ANSWER", re.M | re.I)
_TERMINATE_RE = re.compile(r"TERMINATE\s*$")


def classify_action(role: str, content: str) -> str:
    """The fine action label of one turn; first matching rule wins."""
    stripped = content.strip()
    if "(termination condition)" in role or (
            _TERMINATE_RE.search(stripped) and len(stripped) < 40):
        return "termination"
    if _TOOL_CALL_RE.search(content):
        return "tool_call"
    has_fence = bool(_CODE_FENCE_RE.search(content))
    if _EXEC_OUTPUT_RE.search(content) and not has_fence:
        return "execution_output"
    if _OBSERVATION_RE.search(content):
        return "execution_output"
    if has_fence:
        return "code"
    if _LEDGER_RE.match(content):
        return "ledger"
    if _ADDRESSED_ROLE_RE.search(role) or _INSTRUCTION_JSON_RE.match(content) \
            or _NEXT_AGENT_RE.search(content):
        return "instruction"
    if _FINAL_ANSWER_RE.search(content):
        return "final_answer"
    return "message"


def addressee(role: str, content: str, agents: set[str]) -> str | None:
    """The agent a turn hands off to, when the trace format says so."""
    m = _ADDRESSED_ROLE_RE.search(role)
    if m:
        return m.group(1).strip()
    m = _NEXT_AGENT_RE.search(content)
    if m and m.group(1) in agents:
        return m.group(1)
    m = _NEXT_SPEAKER_JSON_RE.search(content)
    if m and m.group(1) in agents:
        return m.group(1)
    m = _NEXT_SPEAKER_RE.match(content.strip())
    if m and m.group(1) in agents:
        return m.group(1)
    return None


# ── agent identity ───────────────────────────────────────────────────────────

def agent_of(role: str | None) -> str:
    """The agent behind a role string; generic lower-case chat roles are ``Unknown``."""
    name = (role or "").split("(")[0].strip()
    if not name or name in GENERIC_ROLES:
        return "Unknown"
    return name


def agent_list(steps: list["Step"]) -> list[str]:
    """Every agent in order of first appearance, generic roles excluded."""
    seen: list[str] = []
    for step in steps:
        if step.agent != "Unknown" and step.agent not in seen:
            seen.append(step.agent)
    return seen or ["Unknown"]


# ── steps ────────────────────────────────────────────────────────────────────

@dataclass
class Step:
    index: int
    agent: str
    role_raw: str
    action: str
    addressee: str | None
    content: str

    @property
    def has_code(self) -> bool:
        return bool(_CODE_FENCE_RE.search(self.content))

    @property
    def has_tool_call(self) -> bool:
        return bool(_TOOL_CALL_RE.search(self.content))

    def to_doc(self) -> dict:
        return {"step": self.index, "agent": self.agent, "action": self.action,
                "addressee": self.addressee}


def build_steps(record: dict) -> list[Step]:
    """Parse ``record["history"]`` into steps (two passes: classify, then addressees)."""
    view = strip_names(record)
    steps = []
    for idx, turn in enumerate(view["history"]):
        role = turn.get("role", "") or ""
        content = turn.get("content", "") or ""
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        steps.append(Step(index=idx, agent=agent_of(role), role_raw=role,
                          action=classify_action(role, content), addressee=None,
                          content=content))
    agents = set(agent_list(steps))
    for step in steps:
        who = addressee(step.role_raw, step.content, agents)
        if step.action == "message" and who is not None \
                and _NEXT_SPEAKER_RE.match(step.content.strip()):
            step.action = "speaker_selection"
        step.addressee = who
    return steps


def task_step_index(steps: list[Step], question: str) -> int:
    """The step that states the task: a generic-role opener, else the first quoting it."""
    if not steps:
        return 0
    if steps[0].agent == "Unknown":
        return 0
    probe = (question or "").strip()[:60]
    if probe:
        for step in steps:
            if probe in step.content:
                return step.index
    return 0


# ── roles and system descriptions ────────────────────────────────────────────

_TEAM_MARKER = "assembled the following team"
_TEAM_LINE_RE = re.compile(r"^([A-Za-z_][\w\- ]*?):\s+(.+)$")


def agent_roles(record: dict, steps: list[Step], max_chars: int = 300) -> dict[str, str]:
    """Per-agent role text: alg-gen ``system_prompt``, else a hand-crafted team block."""
    sp = record.get("system_prompt")
    if isinstance(sp, dict) and sp:
        return {str(k): str(v)[:max_chars] for k, v in sp.items()}
    for step in steps:
        if _TEAM_MARKER not in step.content:
            continue
        roles: dict[str, str] = {}
        after = step.content.split(_TEAM_MARKER, 1)[1].splitlines()
        started = False
        for line in after:
            m = _TEAM_LINE_RE.match(line.strip())
            if m:
                roles[m.group(1).strip()] = m.group(2).strip()[:max_chars]
                started = True
            elif started and line.strip() == "":
                if roles:
                    break
            elif started:
                break
        if roles:
            return roles
    return {}


def system_description(record: dict, max_chars: int = 1500) -> str | None:
    """TraceElephant's prose description of the agent system, clipped."""
    text = record.get("system")
    if isinstance(text, str) and text.strip():
        return clip(text.strip(), max_chars)
    return None


# ── rendering ────────────────────────────────────────────────────────────────

def clip(text: str, n: int) -> str:
    """Keep the head and the tail so exit codes and final lines survive."""
    if n <= 0 or len(text) <= n:
        return text
    head = int(n * 0.7)
    tail = n - head
    omitted = len(text) - n
    return f"{text[:head]}\n... [{omitted} chars omitted] ...\n{text[-tail:] if tail else ''}"


def _one_line(text: str, n: int) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= n else flat[:n - 3] + "..."


def step_label(step: Step) -> str:
    who = f"{step.agent} | {step.action}"
    if step.addressee:
        who += f" -> {step.addressee}"
    return f"Step {step.index} [{who}]"


def render_step(step: Step, n: int, mark: str | None = None) -> str:
    prefix = f"[{mark}] " if mark else ""
    return f"{prefix}{step_label(step)}: {clip(step.content, n)}"


def render_index_line(step: Step, n: int = 100) -> str:
    return f"{step_label(step)}: {_one_line(step.content, n)}"


# ── JSON that never raises ───────────────────────────────────────────────────

def extract_json_object(raw: str | None) -> dict | None:
    """The first JSON object in a response: fenced block, then outermost braces.

    Each candidate is read strictly first, then leniently: literal newlines
    inside strings are allowed, and a backslash that JSON forbids (``\\sqrt``,
    ``\\omega``, a Windows path) is doubled. Math-heavy traces make models
    write LaTeX inside JSON strings, and a strict reader threw those answers
    away.
    """
    if not raw:
        return None
    text = strip_think(raw)
    candidates = []
    if "```json" in text:
        candidates.append(text.split("```json", 1)[1].split("```", 1)[0])
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            candidates.append(parts[1])
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])
    candidates.append(text)
    for cand in candidates:
        cand = cand.strip()
        if cand.startswith("json"):
            cand = cand[4:].strip()
        for attempt in (cand, _lenient(cand)):
            try:
                data = json.loads(attempt, strict=False)
            except Exception:
                continue
            if isinstance(data, dict):
                return data
            break
    return None


_BAD_ESCAPE_RE = re.compile(r'\\(?!["\\/bfnrtu])')
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def _lenient(text: str) -> str:
    """Escape backslashes JSON forbids (so LaTeX like ``\\sqrt`` survives) and drop trailing commas."""
    return _TRAILING_COMMA_RE.sub(r"\1", _BAD_ESCAPE_RE.sub(r"\\\\", text))
