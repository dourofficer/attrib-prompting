"""The diagnosis team: Strategist, Investigator and Arbiter (paper Section 4.3).

The paper splits diagnosis across three roles that mimic a human debugging
workflow. The Strategist reads the condensed trace and the anomaly tags and
proposes hypotheses, each a step and a suspected MAST failure mode. The
Investigator must back every hypothesis with evidence from a tool: it "cannot
simply state 'the code is wrong'; it must generate a diff or an execution log
proving the discrepancy". The Arbiter weighs the hypotheses with their
evidence, drops the ones whose evidence is empty or inconclusive, and gives
the final (agent, step, mode) with a confidence.

Both tools are language-model probes here (a user decision recorded in
IMPLEMENTATION.md): ``code_exec`` asks the model to trace the code as an
interpreter would and compare with the recorded output; ``logic_probe`` asks
it to check the pre- and post-conditions a step relies on against the steps
that establish them. No code runs. The no-bare-assertion rule is enforced
after parsing as well: evidence with no log, no conditions and no discrepancy
is never conclusive, whatever the model claims.

Memory hooks: the Strategist accepts ``retrieved_patterns`` and renders no
section when the list is empty; the Arbiter emits ``novel_and_robust``, a
signature shaped like the vendored ``ErrorSignature``, and a one-sentence
guard. Nothing reads them yet.
"""
from __future__ import annotations

from ..prompts import MAST_FAMILIES, coerce_step, messages, task_text
from .graph import incoming_neighbours, outgoing_outputs
from .structure import Step, clip, extract_json_object, render_step, step_label
from .tagger import normalize_mode

TOOLS = ("code_exec", "logic_probe")

DECISIVE_STEP = ("the earliest step that, had it been done right, would have "
                 "prevented the wrong final answer")
NO_BARE_ASSERTION = ("You may not simply assert that the step is wrong. Produce a log or "
                     "a diff that shows the discrepancy, or report that you could not.")
MEMORY_HEADER = "VERIFIED FAILURE PATTERNS FROM PAST DIAGNOSES"


def default_tool(step: Step) -> str:
    return "code_exec" if step.action in ("code", "tool_call", "execution_output") else "logic_probe"


def _clamp(value, default: float = 0.5) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, x))


def _mode_lines() -> str:
    return "\n".join(f"- {mode}: {desc}" for mode, desc in MAST_FAMILIES.items())


def _speaker(steps: list[Step], index: int) -> str:
    return steps[index].agent if 0 <= index < len(steps) else "Unknown"


# ── Strategist ───────────────────────────────────────────────────────────────

def render_tags(tags: list[dict], steps: list[Step]) -> str:
    if not tags:
        return "(none)"
    lines = []
    for t in sorted(tags, key=lambda t: (t["step"], t["mode"])):
        lines.append(f"- Step {t['step']} [{_speaker(steps, t['step'])}] {t['mode']} "
                     f"({t['severity']:.2f}): {t['evidence']}")
    return "\n".join(lines)


def render_patterns(patterns) -> str:
    if not patterns:
        return ""
    lines = [f"{MEMORY_HEADER}:"]
    for p in patterns:
        mode = p.get("mode") or p.get("err_family") or "unknown"
        lines.append(f"- [{mode}] {p.get('guard') or p.get('recipe') or ''}".rstrip())
    return "\n".join(lines) + "\n\n"


def build_strategist_prompt(record: dict, steps: list[Step], condensed_text: str,
                            tags: list[dict], agents: list[str], *,
                            retrieved_patterns=(), include_gt: bool = False,
                            max_hypotheses: int = 3, system_text: str | None = None,
                            roles: dict[str, str] | None = None) -> str:
    n = len(steps)
    system_block = ""
    if system_text:
        system_block = f"THE AGENT SYSTEM:\n{system_text}\n\n"
    elif roles:
        lines = "\n".join(f"- {name}: {text}" for name, text in roles.items())
        system_block = f"THE AGENT SYSTEM (roles as the run defined them):\n{lines}\n\n"
    agent_list = "\n".join(f"  {i}. {a}" for i, a in enumerate(agents, 1))

    return f"""You lead the diagnosis of one failed multi-agent run. Your job is to propose hypotheses about the decisive error for an investigator to check.

TASK GIVEN TO THE AGENTS:
{task_text(record, include_gt)}

FAILURE SYMPTOM:
The run ended with an incorrect final answer.

{system_block}AGENTS IN THIS RUN:
{agent_list}

WHAT YOU ARE LOOKING FOR:
The decisive error is {DECISIVE_STEP}. Later steps that merely inherit the mistake are not the decisive error. Steps are numbered from 0 and the numbers below are absolute; the trace has {n} steps.

FAILURE MODES (MAST taxonomy):
{_mode_lines()}

ANOMALY TAGS (heuristic priors from a step-level scan; they may be wrong or incomplete):
{render_tags(tags, steps)}

{render_patterns(retrieved_patterns)}CONDENSED TRACE (steps on the dependency path to the failure; the rest are masked):
{condensed_text}

Propose up to {max_hypotheses} hypotheses, most likely first. Prefer steps shown in the condensed trace. For each, name the step, the agent who spoke it, the failure mode, why it is the decisive error, and the probe the investigator should run: "code_exec" when the step holds code, a tool call or an execution output, "logic_probe" otherwise. "check" states concretely what to verify and which steps to compare; "related_steps" lists the steps the investigator needs to see.

RESPOND IN JSON FORMAT:
{{
  "hypotheses": [
    {{"step": <absolute step number>, "agent": "<agent name from the list>", "mode": "<failure mode id>",
      "rationale": "<why this is the decisive error>", "tool": "code_exec|logic_probe",
      "check": "<what to verify>", "related_steps": [<step numbers>]}}
  ]
}}

Only output valid JSON, no other text."""


def parse_hypotheses(raw: str | None, steps: list[Step], agents: list[str],
                     max_hypotheses: int) -> list[dict] | None:
    """Well-formed hypotheses in rank order; ``None`` when the JSON is unreadable."""
    data = extract_json_object(raw)
    if data is None:
        return None
    items = data.get("hypotheses")
    if not isinstance(items, list):
        return []
    n = len(steps)
    known = {a.lower(): a for a in agents}
    seen = set()
    out = []
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            step = coerce_step(item.get("step"))
        except (TypeError, ValueError):
            continue
        if not (0 <= step < n):
            continue
        mode = normalize_mode(item.get("mode")) or "unknown"
        if (step, mode) in seen:
            continue
        seen.add((step, mode))
        agent = str(item.get("agent", "") or "").strip()
        agent_fixed = False
        if agent.lower() not in known:
            agent, agent_fixed = _speaker(steps, step), True
        else:
            agent = known[agent.lower()]
        tool = str(item.get("tool", "") or "").strip().lower()
        if tool not in TOOLS:
            tool = default_tool(steps[step])
        related = []
        for r in item.get("related_steps") or []:
            try:
                ri = coerce_step(r)
            except (TypeError, ValueError):
                continue
            if 0 <= ri < n and ri != step and ri not in related:
                related.append(ri)
        out.append({"step": step, "agent": agent, "agent_fixed": agent_fixed, "mode": mode,
                    "rationale": str(item.get("rationale", "") or "")[:1000],
                    "tool": tool, "check": str(item.get("check", "") or "")[:600],
                    "related_steps": sorted(related)[:6]})
        if len(out) >= max_hypotheses:
            break
    return out


# ── Investigator ─────────────────────────────────────────────────────────────

def evidence_window(steps: list[Step], edges: list[dict], hypothesis: dict) -> list[int]:
    """The steps the Investigator sees: the step, its inputs, its outputs, the symptom."""
    n = len(steps)
    focus = hypothesis["step"]
    window = {focus, n - 1}
    window.update(hypothesis.get("related_steps", []))
    window.update(incoming_neighbours(edges, focus))
    if hypothesis["tool"] == "code_exec":
        window.update(outgoing_outputs(edges, focus))
    return sorted(i for i in window if 0 <= i < n)


def build_investigator_prompt(record: dict, steps: list[Step], edges: list[dict],
                              hypothesis: dict, *, include_gt: bool = False,
                              step_chars: int = 1200, focus_chars: int = 4000) -> str:
    focus = hypothesis["step"]
    window = evidence_window(steps, edges, hypothesis)
    rendered = []
    for i in window:
        if i == focus:
            rendered.append(render_step(steps[i], focus_chars, mark="HYPOTHESIS STEP"))
        elif i == len(steps) - 1:
            rendered.append(render_step(steps[i], step_chars, mark="FINAL STEP (symptom)"))
        else:
            rendered.append(render_step(steps[i], step_chars))
    window_text = "\n---\n".join(rendered)
    tool = hypothesis["tool"]

    head = f"""You are the investigator for one hypothesis about a failed multi-agent run. You check it with the "{tool}" probe and report evidence.

TASK GIVEN TO THE AGENTS:
{task_text(record, include_gt)}

HYPOTHESIS:
- Step {focus}, spoken by {hypothesis['agent']}
- Suspected failure mode: {hypothesis['mode']}
- Rationale: {hypothesis['rationale']}
- What to check: {hypothesis['check'] or '(not specified)'}

EVIDENCE WINDOW (absolute step numbers; the trace has {len(steps)} steps):
{window_text}

RULE:
{NO_BARE_ASSERTION}
"""
    if tool == "code_exec":
        return head + f"""
PROBE: code_exec. Trace the code or the tool call at step {focus} as an interpreter would, using the inputs visible in the window. Write the execution log step by step. State what output the code should have produced, quote what the trace recorded, and give the discrepancy as a diff. If the window holds no runnable code, say so and mark the evidence inconclusive.

RESPOND IN JSON FORMAT:
{{
  "tool": "code_exec",
  "execution_log": "<your trace of the execution>",
  "expected_output": "<what a correct run produces>",
  "observed_output": "<what the trace recorded, quoted>",
  "discrepancy": "<the diff between expected and observed, or empty>",
  "supports_hypothesis": <true or false>,
  "conclusive": <true or false>,
  "confidence": <0.0 to 1.0>,
  "notes": "<anything the arbiter should know>"
}}

Only output valid JSON, no other text."""
    return head + f"""
PROBE: logic_probe. State the preconditions step {focus} relies on (facts or results it takes from earlier steps) and the postconditions it claims (what it asserts to have established). For each, quote the step that establishes or contradicts it and say whether it holds. Then state the discrepancy, if any, between what the step assumed or claimed and what the trace shows.

RESPOND IN JSON FORMAT:
{{
  "tool": "logic_probe",
  "preconditions": [
    {{"condition": "<what the step relies on>", "source_steps": [<step numbers>], "holds": <true or false>, "evidence": "<quote>"}}
  ],
  "postconditions": [
    {{"condition": "<what the step claims>", "used_by_steps": [<step numbers>], "holds": <true or false>, "evidence": "<quote>"}}
  ],
  "discrepancy": "<what was assumed or claimed versus what the trace shows, or empty>",
  "supports_hypothesis": <true or false>,
  "conclusive": <true or false>,
  "confidence": <0.0 to 1.0>
}}

Only output valid JSON, no other text."""


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "yes", "1")


def parse_evidence(raw: str | None, tool: str) -> dict:
    """Structured evidence; a parse failure or a bare assertion is never conclusive."""
    data = extract_json_object(raw)
    if data is None:
        return {"tool": tool, "conclusive": False, "supports_hypothesis": False,
                "confidence": 0.0, "discrepancy": "", "parse_error": True}
    discrepancy = str(data.get("discrepancy", "") or "").strip()
    ev = {"tool": tool,
          "supports_hypothesis": _as_bool(data.get("supports_hypothesis", False)),
          "conclusive": _as_bool(data.get("conclusive", False)),
          "confidence": _clamp(data.get("confidence"), default=0.0),
          "discrepancy": discrepancy[:1500], "parse_error": False}
    if tool == "code_exec":
        for key in ("execution_log", "expected_output", "observed_output", "notes"):
            ev[key] = str(data.get(key, "") or "")[:2000]
        substance = bool(ev["execution_log"].strip() or discrepancy)
    else:
        for key in ("preconditions", "postconditions"):
            items = data.get(key)
            ev[key] = [i for i in items if isinstance(i, dict)][:8] if isinstance(items, list) else []
        substance = bool(ev["preconditions"] or ev["postconditions"] or discrepancy)
    if not substance:
        ev["conclusive"] = False
        ev["bare_assertion"] = True
    return ev


# ── Arbiter ──────────────────────────────────────────────────────────────────

def split_hypotheses(hypotheses: list[dict], evidence: list[dict]) -> tuple[list[int], list[int]]:
    """Indices of hypotheses whose evidence is conclusive and supportive, and the rest."""
    survivors, filtered = [], []
    for j, ev in enumerate(evidence):
        if ev.get("conclusive") and ev.get("supports_hypothesis"):
            survivors.append(j)
        else:
            filtered.append(j)
    return survivors, filtered


def _render_evidence(ev: dict) -> str:
    lines = [f"  tool: {ev['tool']}, conclusive: {ev['conclusive']}, "
             f"supports: {ev['supports_hypothesis']}, confidence: {ev['confidence']:.2f}"]
    if ev.get("parse_error"):
        lines.append("  (the investigator's answer could not be read)")
    if ev["tool"] == "code_exec":
        for key in ("execution_log", "expected_output", "observed_output"):
            if ev.get(key):
                lines.append(f"  {key}: {clip(ev[key], 800)}")
    else:
        for key in ("preconditions", "postconditions"):
            for c in ev.get(key, []):
                lines.append(f"  {key[:-1]}: {str(c.get('condition', ''))[:200]} "
                             f"-> holds: {c.get('holds')} (steps {c.get('source_steps') or c.get('used_by_steps') or []}); "
                             f"{str(c.get('evidence', ''))[:200]}")
    if ev.get("discrepancy"):
        lines.append(f"  discrepancy: {clip(ev['discrepancy'], 800)}")
    if ev.get("notes"):
        lines.append(f"  notes: {clip(ev['notes'], 300)}")
    return "\n".join(lines)


def _render_hypothesis(j: int, h: dict, ev: dict | None, full: bool) -> str:
    head = (f"Hypothesis {j}: step {h['step']} by {h['agent']}, mode {h['mode']}\n"
            f"  rationale: {clip(h['rationale'], 600 if full else 200)}")
    if not full or ev is None:
        return head
    return head + "\n" + _render_evidence(ev)


def build_arbiter_prompt(record: dict, steps: list[Step], agents: list[str],
                         hypotheses: list[dict], evidence: list[dict], *,
                         include_gt: bool = False) -> tuple[str, bool]:
    survivors, filtered = split_hypotheses(hypotheses, evidence)
    only_inconclusive = not survivors
    agent_list = "\n".join(f"  {i}. {a}" for i, a in enumerate(agents, 1))

    if only_inconclusive:
        body = ("NO HYPOTHESIS HAS CONCLUSIVE SUPPORTING EVIDENCE. All are listed below with what the "
                "investigator found. Choose the best-supported one and keep the confidence low.\n\n"
                + "\n\n".join(_render_hypothesis(j, hypotheses[j], evidence[j], True)
                              for j in range(len(hypotheses))))
    else:
        body = "HYPOTHESES WITH CONCLUSIVE EVIDENCE:\n\n" + "\n\n".join(
            _render_hypothesis(j, hypotheses[j], evidence[j], True) for j in survivors)
        if filtered:
            body += "\n\nFILTERED (evidence empty or inconclusive; shown for completeness only):\n" + "\n".join(
                _render_hypothesis(j, hypotheses[j], None, False) for j in filtered)

    prompt = f"""You are the arbiter for the diagnosis of one failed multi-agent run. Investigators have checked the hypotheses below; you give the verdict.

TASK GIVEN TO THE AGENTS:
{task_text(record, include_gt)}

FAILURE SYMPTOM:
The run ended with an incorrect final answer.

AGENTS IN THIS RUN:
{agent_list}

WHAT YOU DECIDE:
The decisive error is {DECISIVE_STEP}. Steps are numbered from 0 and absolute; the trace has {len(steps)} steps.

{body}

Weigh the evidence, not the rhetoric: a hypothesis backed by a concrete log or a violated condition outranks one that only asserts. Choose the decisive error, give a confidence from 0.0 to 1.0, and say whether the diagnosed pattern is novel and robust enough to be remembered for future runs (reproducible evidence, a lesson that generalises beyond this trace). The "signature" abstracts the pattern: the failure mode, the tool or API involved, the argument types, and context slots such as the agent's role and the task domain. The "guard" is one sentence a future run could check to avoid the same mistake.

RESPOND IN JSON FORMAT:
{{
  "agent": "<agent name from the list>",
  "step": <absolute step number>,
  "mode": "<failure mode id>",
  "confidence": <0.0 to 1.0>,
  "reason": "<what went wrong and why this is the decisive error>",
  "chosen_hypothesis": <hypothesis number, or -1 if none>,
  "novel_and_robust": <true or false>,
  "signature": {{"mast_mode": "<failure mode id>", "tool": "<tool or api>", "args": ["<argument types>"], "context": ["<agent role>", "<task domain>"]}},
  "guard": "<one sentence>"
}}

Only output valid JSON, no other text."""
    return prompt, only_inconclusive


def _signature(data, mode: str) -> dict:
    sig = data if isinstance(data, dict) else {}
    def _strs(value):
        if isinstance(value, list):
            return [str(v)[:100] for v in value if v is not None][:8]
        return [str(value)[:100]] if value else []
    return {"tool": str(sig.get("tool", "") or "unknown")[:100], "api": "default",
            "arg_schema": _strs(sig.get("args")), "context_slots": _strs(sig.get("context")),
            "err_family": normalize_mode(sig.get("mast_mode")) or mode}


def parse_verdict(raw: str | None, steps: list[Step], agents: list[str]) -> dict | None:
    data = extract_json_object(raw)
    if data is None:
        return None
    try:
        step = coerce_step(data.get("step"))
    except (TypeError, ValueError):
        return None
    if not (0 <= step < len(steps)):
        return None
    known = {a.lower(): a for a in agents}
    agent = str(data.get("agent", "") or "").strip()
    agent_fixed = agent.lower() not in known
    agent = _speaker(steps, step) if agent_fixed else known[agent.lower()]
    mode = normalize_mode(data.get("mode")) or "unknown"
    try:
        chosen = int(data.get("chosen_hypothesis", -1))
    except (TypeError, ValueError):
        chosen = -1
    return {"agent": agent, "agent_fixed": agent_fixed, "step": step, "mode": mode,
            "confidence": _clamp(data.get("confidence"), default=0.0),
            "reason": str(data.get("reason", "") or "")[:2000],
            "chosen_hypothesis": chosen,
            "novel_and_robust": _as_bool(data.get("novel_and_robust", False)),
            "signature": _signature(data.get("signature"), mode),
            "guard": str(data.get("guard", "") or "")[:500]}


def select_fallback(hypotheses: list[dict], evidence: list[dict]) -> int:
    """The hypothesis that stands when the Arbiter fails: best evidence, else first."""
    survivors, _ = split_hypotheses(hypotheses, evidence)
    if survivors:
        return max(survivors, key=lambda j: (evidence[j]["confidence"], -j))
    return 0


def memory_candidate(verdict: dict, evidence: dict | None, tau: float = 0.7) -> dict:
    """What a verified-before-write gate would see (paper eq. 2); computed, unused."""
    verified = bool(evidence and evidence.get("conclusive") and evidence.get("supports_hypothesis"))
    return {"eligible": verified and verdict["confidence"] >= tau and verdict["novel_and_robust"],
            "verified": verified, "confidence": verdict["confidence"],
            "novel_and_robust": verdict["novel_and_robust"],
            "signature": verdict["signature"], "guard": verdict["guard"], "tau": tau}


__all__ = [
    "TOOLS", "DECISIVE_STEP", "NO_BARE_ASSERTION", "MEMORY_HEADER", "default_tool",
    "build_strategist_prompt", "parse_hypotheses", "evidence_window",
    "build_investigator_prompt", "parse_evidence", "split_hypotheses",
    "build_arbiter_prompt", "parse_verdict", "select_fallback", "memory_candidate",
    "messages", "step_label",
]
