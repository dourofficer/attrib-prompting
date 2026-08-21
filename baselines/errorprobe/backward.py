"""ErrorProbe backward tracing, ported from the vendored code as generators.

The vendored ``BackwardTracer`` (``simplified_mas/backward_tracer.py``) walks a
failed trace from its last turn toward its first: a *relevance* LLM call per
turn, a *root-cause* LLM call for relevant turns, findings accumulated in an
``ErrorTracingMemory`` (LLM summarization once it holds more than 5
observations, LLM pruning past 10 when the 15-turn maintenance interval or the
50-observation cap fires, an LLM merge probe whose answer only matters when it
says "none"), and a final *synthesis* call that produces the prediction.

This port keeps that control flow, every prompt byte and every parsing rule
identical — asserted by ``tests/test_errorprobe_pipeline.py``, which drives the
real vendored ``BackwardTracer`` with a scripted client and checks both sides
issue the same prompt sequence and return the same prediction. What changes is
plumbing only: each ``llm_client.invoke(prompt)`` becomes a yielded round
``(meta, [messages(prompt)])`` so the shared runner supplies the responses
(``methods.py`` relays these, logging every response into ``calls``). Responses
arrive already ``strip_think``-ed. The vendored fallbacks for a *failed* LLM
call (network errors → heuristic summarization, etc.) have no equivalent here
because backend errors abort the trajectory at the runner and the rerun
resumes it (GUIDE.md "Resume"); the fallbacks for an *unparseable* response
are kept verbatim.

Two adaptation glue points, both in ``TraceIndex``-land and documented in
``IMPLEMENTATION.md``: the agent of a turn comes from the repo's identity rule
(``role`` with the vendored fallback cleanup — the vendored class reads only
``name``, which this repo's data does not use for agents), and the synthesized
``mistake_step`` is coerced to int where the vendored process would crash on a
non-integer.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime

from .prompts import agent_from_turn, messages

# The vendored TracingConfig as AnalyzerAgent constructs it (llm_agents.py):
# max_turns_to_examine=100, context_window_radius=5, confidence_threshold=0.75,
# min_turns_before_stopping=10; the dataclass defaults fill the rest.
MAX_TURNS_TO_EXAMINE = 100
CONTEXT_WINDOW_RADIUS = 5
CONFIDENCE_THRESHOLD = 0.75
MIN_TURNS_BEFORE_STOPPING = 10
ENABLE_PRUNING = True
PRUNING_INTERVAL = 15
MAX_OBSERVATIONS = 50


# ── TraceIndex (simplified_mas/trace_index.py) ───────────────────────────────

@dataclass
class Turn:
    index: int
    role: str
    agent: str
    content: str
    action: str | None = None
    tool_calls: list = field(default_factory=list)


def _extract_action(turn_data: dict) -> str | None:
    """Verbatim: classify a turn by the vendored (MetaGPT-shaped) patterns."""
    content = turn_data.get("content", "")
    if '"command_name"' in content or "'command_name'" in content:
        if "Editor." in content or "editor." in content:
            return "editor"
        elif "Plan." in content or "plan." in content:
            return "planning"
        elif "TeamLeader." in content or "teamleader." in content:
            return "coordination"
        elif "RoleZero." in content:
            return "communication"
        else:
            return "action"
    if "thinking:" in content.lower() or "# past experience" in content.lower():
        return "thinking"
    return None


def _extract_tool_calls(turn_data: dict) -> list[str]:
    """Verbatim: the vendored tool-name patterns."""
    content = turn_data.get("content", "")
    tools = []
    tool_patterns = [
        "Editor.", "Plan.", "TeamLeader.", "RoleZero.",
        "ProductManager.", "Architect.", "Engineer.",
    ]
    for pattern in tool_patterns:
        if pattern in content:
            tools.append(pattern.replace(".", ""))
    return tools


class TraceIndex:
    """Full-trace turn index. Agent identity is the repo rule, not ``name``."""

    def __init__(self, trace: dict):
        self.question = trace.get("question", "")
        self.turns: list[Turn] = []
        for idx, turn_data in enumerate(trace.get("history", [])):
            self.turns.append(Turn(
                index=idx,
                role=turn_data.get("role", "unknown"),
                agent=agent_from_turn(turn_data),
                content=turn_data.get("content", ""),
                action=_extract_action(turn_data),
                tool_calls=_extract_tool_calls(turn_data),
            ))


# ── ErrorTracingMemory (simplified_mas/error_tracing_memory.py) ──────────────

@dataclass
class Observation:
    turn_idx: int
    agent: str
    content: str
    importance: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    tags: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "turn_idx": self.turn_idx,
            "agent": self.agent,
            "content": self.content,
            "importance": self.importance,
            "timestamp": self.timestamp,
            "tags": self.tags,
        }


@dataclass
class Hypothesis:
    error_type: str
    agent: str
    step: int
    confidence: float
    evidence: list
    reasoning: str = ""
    supporting_turns: list = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "error_type": self.error_type,
            "agent": self.agent,
            "step": self.step,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "reasoning": self.reasoning,
            "supporting_turns": self.supporting_turns,
            "timestamp": self.timestamp,
        }

    def __str__(self) -> str:
        return f"{self.error_type} in {self.agent} at step {self.step} (conf={self.confidence:.2f})"


class TracingMemory:
    """The vendored working memory; LLM-touching methods are sub-generators.

    In the vendored run an LLM client is always present, so every
    ``self.llm_client and …`` guard reduces to its length condition — that is
    what the ports below keep.
    """

    def __init__(self, max_observations: int = MAX_OBSERVATIONS):
        self.observations: list[Observation] = []
        self.hypotheses: list[Hypothesis] = []
        self.trace_path: list[int] = []
        self.max_observations = max_observations
        self._cached_context: str | None = None
        self._cache_valid: bool = False

    # -- core operations ------------------------------------------------------

    def add_observation(self, turn_idx, agent, content, importance="medium", tags=None):
        """Sub-generator: appending may trigger the (LLM-based) auto-prune."""
        self.observations.append(Observation(
            turn_idx=turn_idx, agent=agent, content=content,
            importance=importance, tags=tags or [],
        ))
        self._cache_valid = False
        if len(self.observations) > self.max_observations:
            yield from self.prune_irrelevant_observations()

    def add_hypothesis(self, error_type, agent, step, confidence, evidence,
                       reasoning="", supporting_turns=None) -> None:
        self.hypotheses.append(Hypothesis(
            error_type=error_type, agent=agent, step=step, confidence=confidence,
            evidence=evidence, reasoning=reasoning,
            supporting_turns=supporting_turns or [],
        ))
        self._cache_valid = False
        self.hypotheses.sort(key=lambda h: h.confidence, reverse=True)

    def mark_turn_examined(self, turn_idx: int) -> None:
        if turn_idx not in self.trace_path:
            self.trace_path.append(turn_idx)

    def get_top_hypothesis(self) -> Hypothesis | None:
        return self.hypotheses[0] if self.hypotheses else None

    def get_hypotheses(self, min_confidence: float = 0.0) -> list[Hypothesis]:
        return [h for h in self.hypotheses if h.confidence >= min_confidence]

    def get_observations_by_importance(self, importance: str) -> list[Observation]:
        return [o for o in self.observations if o.importance == importance]

    # -- context generation ---------------------------------------------------

    def get_context_for_llm(self, max_tokens: int = 2000):
        """Sub-generator returning the condensed findings summary."""
        if self._cache_valid and self._cached_context:
            return self._cached_context
        if len(self.observations) > 5:
            context = yield from self._llm_summarize_context(max_tokens)
        else:
            context = self._heuristic_summarize_context(max_tokens)
        self._cached_context = context
        self._cache_valid = True
        return context

    def _heuristic_summarize_context(self, max_tokens: int) -> str:
        parts = []
        parts.append(f"=== Traced {len(self.trace_path)} turns ===\n")
        if self.hypotheses:
            parts.append("HYPOTHESES:")
            for i, hyp in enumerate(self.hypotheses[:3], 1):
                parts.append(
                    f"{i}. {hyp.error_type} in {hyp.agent} at step {hyp.step} "
                    f"(confidence: {hyp.confidence:.2f})"
                )
                if hyp.reasoning:
                    parts.append(f"   Reasoning: {hyp.reasoning[:100]}")
            parts.append("")
        high_importance = [o for o in self.observations if o.importance == "high"]
        if high_importance:
            parts.append("KEY OBSERVATIONS:")
            for obs in high_importance[-5:]:
                parts.append(f"- Turn {obs.turn_idx} [{obs.agent}]: {obs.content[:100]}")
            parts.append("")
        if len(self.observations) > len(high_importance):
            parts.append("RECENT OBSERVATIONS:")
            recent = [o for o in self.observations[-3:] if o not in high_importance]
            for obs in recent:
                parts.append(f"- Turn {obs.turn_idx} [{obs.agent}]: {obs.content[:80]}")
        return "\n".join(parts)

    def _llm_summarize_context(self, max_tokens: int):
        prompt_parts = [
            "Summarize these error tracing findings into a concise context (3-5 sentences).",
            "Focus on: key patterns, potential error causes, and agent behaviors.\n",
            "=== HYPOTHESES ===",
        ]
        for i, hyp in enumerate(self.hypotheses[:5], 1):
            prompt_parts.append(
                f"{i}. {hyp.error_type} in {hyp.agent} at step {hyp.step} "
                f"(confidence: {hyp.confidence:.2f})"
            )
            if hyp.evidence:
                prompt_parts.append(f"   Evidence: {', '.join(hyp.evidence[:3])}")
        prompt_parts.append("\n=== OBSERVATIONS ===")
        for obs in self.observations[-10:]:
            prompt_parts.append(
                f"Turn {obs.turn_idx} [{obs.agent}] ({obs.importance}): {obs.content[:150]}"
            )
        prompt_parts.append("\n=== SUMMARY ===")
        prompt_parts.append("Provide a concise summary:")
        prompt = "\n".join(prompt_parts)

        response = (yield ({"role": "summarize"}, [messages(prompt)]))[0]
        return response.strip()

    # -- pruning & merging ----------------------------------------------------

    def prune_irrelevant_observations(self):
        """Sub-generator returning the number of observations removed."""
        if len(self.observations) > 10:
            removed = yield from self._llm_prune_observations()
            return removed
        return self._heuristic_prune_observations()

    def _heuristic_prune_observations(self) -> int:
        if len(self.observations) <= 20:
            return 0
        initial_count = len(self.observations)
        turn_indices = [o.turn_idx for o in self.observations]
        recent_threshold = max(turn_indices) - 20 if turn_indices else 0
        self.observations = [
            o for o in self.observations
            if o.importance == "high" or o.turn_idx >= recent_threshold
        ]
        removed = initial_count - len(self.observations)
        if removed > 0:
            self._cache_valid = False
        return removed

    def _llm_prune_observations(self):
        if not self.hypotheses:
            return self._heuristic_prune_observations()
        top_hypothesis = self.get_top_hypothesis()
        prompt_parts = [
            "You are analyzing error traces. We have a hypothesis about the error:",
            f"HYPOTHESIS: {top_hypothesis.error_type} in {top_hypothesis.agent} at step {top_hypothesis.step}",
            f"Reasoning: {top_hypothesis.reasoning}\n",
            "Which of these observations are RELEVANT to this hypothesis?",
            "Respond with ONLY the observation numbers that are relevant (comma-separated).\n",
            "OBSERVATIONS:",
        ]
        for i, obs in enumerate(self.observations, 1):
            prompt_parts.append(
                f"{i}. Turn {obs.turn_idx} [{obs.agent}]: {obs.content[:100]}"
            )
        prompt_parts.append("\nRelevant observation numbers:")
        prompt = "\n".join(prompt_parts)

        response = (yield ({"role": "prune"}, [messages(prompt)]))[0]

        relevant_nums = set()
        for token in response.replace(",", " ").split():
            try:
                num = int(token.strip())
                if 1 <= num <= len(self.observations):
                    relevant_nums.add(num - 1)
            except ValueError:
                continue
        initial_count = len(self.observations)
        self.observations = [
            o for i, o in enumerate(self.observations)
            if i in relevant_nums or self.observations[i].importance == "high"
        ]
        removed = initial_count - len(self.observations)
        if removed > 0:
            self._cache_valid = False
        return removed

    def merge_similar_hypotheses(self):
        """Sub-generator; the LLM answer matters only when it says "none"."""
        if len(self.hypotheses) <= 1:
            return 0
        prompt_parts = [
            "You are analyzing error hypotheses. Are any of these essentially describing the same error?",
            "If yes, suggest which to merge.\n",
            "HYPOTHESES:",
        ]
        for i, hyp in enumerate(self.hypotheses, 1):
            prompt_parts.append(
                f"{i}. {hyp.error_type} in {hyp.agent} at step {hyp.step} "
                f"(confidence: {hyp.confidence:.2f})"
            )
            if hyp.reasoning:
                prompt_parts.append(f"   Reasoning: {hyp.reasoning[:150]}")
        prompt_parts.append("\nWhich hypotheses should be merged? Format: '1,2' or '3,4,5' or 'none'")
        prompt = "\n".join(prompt_parts)

        response = (yield ({"role": "merge"}, [messages(prompt)]))[0]
        if "none" in response.lower():
            return 0
        return self._heuristic_merge_hypotheses()

    def _heuristic_merge_hypotheses(self) -> int:
        initial_count = len(self.hypotheses)
        merged = []
        seen = set()
        for hyp in self.hypotheses:
            key = (hyp.error_type, hyp.agent)
            if key not in seen:
                seen.add(key)
                merged.append(hyp)
            else:
                existing = next(h for h in merged if (h.error_type, h.agent) == key)
                if hyp.confidence > existing.confidence:
                    existing.confidence = hyp.confidence
                existing.evidence = list(set(existing.evidence + hyp.evidence))
                existing.supporting_turns = list(set(existing.supporting_turns + hyp.supporting_turns))
        self.hypotheses = merged
        removed = initial_count - len(self.hypotheses)
        if removed > 0:
            self._cache_valid = False
        return removed

    def get_summary(self) -> dict:
        return {
            "observations_count": len(self.observations),
            "hypotheses_count": len(self.hypotheses),
            "turns_examined": len(self.trace_path),
            "top_hypothesis": self.get_top_hypothesis().to_dict() if self.hypotheses else None,
            "high_importance_observations": len(self.get_observations_by_importance("high")),
            "medium_importance_observations": len(self.get_observations_by_importance("medium")),
            "low_importance_observations": len(self.get_observations_by_importance("low")),
        }


# ── BackwardTracer (simplified_mas/backward_tracer.py) ───────────────────────

def _get_context_window(trace_index: TraceIndex, center_idx: int,
                        radius: int = CONTEXT_WINDOW_RADIUS) -> str:
    start_idx = max(0, center_idx - radius)
    end_idx = min(len(trace_index.turns), center_idx + radius + 1)
    window_turns = trace_index.turns[start_idx:end_idx]
    lines = []
    for i, turn in enumerate(window_turns, start=start_idx):
        marker = ">>>" if i == center_idx else "   "
        lines.append(
            f"{marker} Turn {turn.index} [{turn.agent}]: "
            f"{turn.content[:150]}..."
        )
    return "\n".join(lines)


def _check_relevance(turn: Turn, turn_idx: int, memory_context: str, context_window: str):
    """Sub-generator → (is_relevant, score)."""
    prompt = f"""You are tracing backward through a failed task execution to find the root cause.

CURRENT FINDINGS:
{memory_context}

CURRENT TURN (Turn {turn.index}):
Agent: {turn.agent}
Content: {turn.content[:500]}

CONTEXT:
{context_window}

Is this turn RELEVANT to understanding the error? Consider:
- Does it show unusual behavior?
- Does it relate to current hypotheses?
- Does it involve decision points or actions?

Respond in JSON format:
{{
  "is_relevant": true/false,
  "score": 0.0-1.0,
  "reason": "brief explanation"
}}"""

    response = (yield ({"role": "relevance", "turn_idx": turn_idx}, [messages(prompt)]))[0]
    try:
        result = json.loads(response.strip())
        return result.get("is_relevant", False), float(result.get("score", 0.0))
    except Exception:
        # Vendored fallback: any turn with an action or tool call is relevant.
        has_action = turn.action is not None or len(turn.tool_calls) > 0
        return has_action, 0.5


def _analyze_for_root_cause(turn: Turn, turn_idx: int, memory_context: str,
                            context_window: str, question: str):
    """Sub-generator → (is_potential_cause, error_type, confidence, reasoning, observations)."""
    prompt = f"""You are analyzing a potentially problematic turn in a multi-agent system execution.

TASK:
{question}

CURRENT FINDINGS:
{memory_context}

TURN BEING ANALYZED (Turn {turn.index}):
Agent: {turn.agent}
Role: {turn.role}
Content: {turn.content[:800]}
{f'Action: {turn.action}' if turn.action else ''}
{f'Tool Calls: {turn.tool_calls}' if turn.tool_calls else ''}

SURROUNDING CONTEXT:
{context_window}

ANALYSIS QUESTIONS:
1. Could this turn be the ROOT CAUSE of the error?
2. If yes, what type of error occurred?
   - logic_error: Flawed reasoning or incorrect algorithm
   - incomplete_verification: Insufficient testing or validation
   - reasoning_action_mismatch: Action doesn't match stated reasoning
   - step_repetition: Unnecessary repetition of steps
   - premature_termination: Stopped before completing task
   - fail_clarification: Failed to seek/provide needed clarification
   - fail_task_spec: Didn't follow task specifications
   - other: Specify in reasoning
3. How confident are you this is the root cause? (0.0-1.0)
4. What observations support this conclusion?

Respond in JSON format:
{{
  "is_potential_cause": true/false,
  "error_type": "type_name" or null,
  "confidence": 0.0-1.0,
  "reasoning": "detailed explanation of why this is/isn't the root cause",
  "observations": ["observation 1", "observation 2", ...]
}}"""

    response = (yield ({"role": "root_cause", "turn_idx": turn_idx}, [messages(prompt)]))[0]
    try:
        result = json.loads(response.strip())
        return (
            result.get("is_potential_cause", False),
            result.get("error_type"),
            float(result.get("confidence", 0.0)),
            result.get("reasoning", ""),
            result.get("observations", []),
        )
    except Exception as exc:
        return False, None, 0.0, f"Analysis failed: {str(exc)}", []


def _should_stop_tracing(memory: TracingMemory, turns_examined: int) -> bool:
    if turns_examined < MIN_TURNS_BEFORE_STOPPING:
        return False
    if turns_examined >= MAX_TURNS_TO_EXAMINE:
        return True
    top_hypothesis = memory.get_top_hypothesis()
    if top_hypothesis and top_hypothesis.confidence >= CONFIDENCE_THRESHOLD:
        return True
    high_conf_hyps = memory.get_hypotheses(min_confidence=0.6)
    if len(high_conf_hyps) >= 2:
        if len(set(h.step for h in high_conf_hyps)) <= 1:
            return True
    return False


def _fallback_prediction(memory: TracingMemory, trace_index: TraceIndex) -> dict:
    high_imp_obs = memory.get_observations_by_importance("high")
    if high_imp_obs:
        latest_obs = high_imp_obs[-1]
        return {
            "mistake_agent": latest_obs.agent,
            "mistake_step": latest_obs.turn_idx,
            "mistake_type": "unknown",
            "mistake_reason": f"Based on observation: {latest_obs.content}",
            "confidence": 0.3,
            "method": "backward_tracing_fallback",
        }
    final_turn = trace_index.turns[-1] if trace_index.turns else None
    if final_turn:
        return {
            "mistake_agent": final_turn.agent,
            "mistake_step": final_turn.index,
            "mistake_type": "unknown",
            "mistake_reason": "No clear hypothesis found; defaulting to final turn",
            "confidence": 0.1,
            "method": "backward_tracing_fallback",
        }
    return {
        "mistake_agent": "Unknown",
        "mistake_step": 0,
        "mistake_type": "unknown",
        "mistake_reason": "Unable to determine error cause",
        "confidence": 0.0,
        "method": "backward_tracing_fallback",
    }


def _synthesize_final_answer(memory: TracingMemory, trace_index: TraceIndex, question: str):
    """Sub-generator → the final prediction dict."""
    top_hypothesis = memory.get_top_hypothesis()
    if not top_hypothesis:
        return _fallback_prediction(memory, trace_index)

    memory_summary = yield from memory.get_context_for_llm()

    prompt = f"""Based on the complete backward trace analysis, provide the final error diagnosis.

TASK:
{question}

COMPLETE FINDINGS:
{memory_summary}

TOP HYPOTHESIS:
{top_hypothesis}

Provide the final diagnosis in JSON format:
{{
  "mistake_agent": "agent name",
  "mistake_step": step_number,
  "mistake_type": "error_type",
  "mistake_reason": "detailed explanation",
  "confidence": 0.0-1.0
}}

The diagnosis should:
1. Identify the specific agent that made the error
2. Identify the exact step number where the error occurred
3. Classify the error type
4. Explain clearly what went wrong and why"""

    response = (yield ({"role": "synthesize"}, [messages(prompt)]))[0]
    try:
        result = json.loads(response.strip())
        return {
            "mistake_agent": result.get("mistake_agent", top_hypothesis.agent),
            "mistake_step": result.get("mistake_step", top_hypothesis.step),
            "mistake_type": result.get("mistake_type", top_hypothesis.error_type),
            "mistake_reason": result.get("mistake_reason", top_hypothesis.reasoning),
            "confidence": result.get("confidence", top_hypothesis.confidence),
            "supporting_evidence": top_hypothesis.evidence,
            "method": "backward_tracing",
        }
    except Exception:
        return {
            "mistake_agent": top_hypothesis.agent,
            "mistake_step": top_hypothesis.step,
            "mistake_type": top_hypothesis.error_type,
            "mistake_reason": top_hypothesis.reasoning,
            "confidence": top_hypothesis.confidence,
            "supporting_evidence": top_hypothesis.evidence,
            "method": "backward_tracing",
        }


def backward_trace(trace_index: TraceIndex, question: str):
    """The vendored ``BackwardTracer.trace`` loop as one sub-generator."""
    memory = TracingMemory()
    final_turn_idx = len(trace_index.turns) - 1

    yield from memory.add_observation(
        turn_idx=final_turn_idx,
        agent=trace_index.turns[final_turn_idx].agent if trace_index.turns else "Unknown",
        content="Task execution completed but result is incorrect",
        importance="high",
        tags=["symptom", "final_turn"],
    )

    turns_examined = 0
    for turn_idx in reversed(range(final_turn_idx + 1)):
        turn = trace_index.turns[turn_idx]
        memory.mark_turn_examined(turn_idx)
        turns_examined += 1

        memory_context = yield from memory.get_context_for_llm()
        context_window = _get_context_window(trace_index, turn_idx)

        is_relevant, _score = yield from _check_relevance(
            turn, turn_idx, memory_context, context_window)

        if is_relevant:
            (is_potential_cause, error_type, confidence, reasoning,
             observations) = yield from _analyze_for_root_cause(
                turn, turn_idx, memory_context, context_window, question)

            for obs_content in observations:
                yield from memory.add_observation(
                    turn_idx=turn_idx,
                    agent=turn.agent,
                    content=obs_content,
                    importance="high" if is_potential_cause else "medium",
                )
            if is_potential_cause and error_type:
                memory.add_hypothesis(
                    error_type=error_type,
                    agent=turn.agent,
                    step=turn.index,
                    confidence=confidence,
                    evidence=observations,
                    reasoning=reasoning,
                    supporting_turns=[turn_idx],
                )

        if ENABLE_PRUNING and turns_examined % PRUNING_INTERVAL == 0:
            yield from memory.prune_irrelevant_observations()
            yield from memory.merge_similar_hypotheses()

        if _should_stop_tracing(memory, turns_examined):
            break

    prediction = yield from _synthesize_final_answer(memory, trace_index, question)
    prediction["turns_examined"] = turns_examined
    prediction["total_turns"] = len(trace_index.turns)
    prediction["memory_summary"] = memory.get_summary()
    return prediction
