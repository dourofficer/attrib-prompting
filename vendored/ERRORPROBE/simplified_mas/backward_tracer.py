"""
BackwardTracer - Core algorithm for backward error tracing

This component implements the backward tracing algorithm that:
1. Starts from the final turn (symptom: task failed)
2. Traces backward turn by turn
3. Uses LLM to determine relevance and identify root causes
4. Accumulates findings in ErrorTracingMemory
5. Stops when high-confidence root cause is found

The algorithm progressively builds understanding by examining turns
in reverse order and maintaining context through ErrorTracingMemory.
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import json

from .trace_index import TraceIndex, Turn
from .error_tracing_memory import ErrorTracingMemory, Observation, Hypothesis


@dataclass
class TracingConfig:
    """Configuration for backward tracing"""
    max_turns_to_examine: int = 100  # Stop after examining this many turns
    context_window_radius: int = 5  # Turns before/after for context
    confidence_threshold: float = 0.75  # Stop if hypothesis exceeds this
    min_turns_before_stopping: int = 10  # Minimum turns to examine before stopping
    relevance_threshold: float = 0.5  # Threshold for relevance detection
    enable_pruning: bool = True  # Enable memory pruning
    pruning_interval: int = 15  # Prune every N turns


@dataclass
class TurnAnalysis:
    """Result of analyzing a single turn"""
    turn_idx: int
    is_relevant: bool
    relevance_score: float
    observations: List[str]
    is_potential_cause: bool
    error_type: Optional[str] = None
    confidence: float = 0.0
    reasoning: str = ""


class BackwardTracer:
    """
    Core backward tracing algorithm.

    Traces backward from final turn to identify root cause of errors.
    Uses LLM to analyze relevance and identify potential error causes.
    Accumulates findings in ErrorTracingMemory for progressive understanding.
    """

    def __init__(self, llm_client, config: Optional[TracingConfig] = None):
        """
        Initialize backward tracer.

        Args:
            llm_client: LLM client for analysis
            config: Tracing configuration
        """
        self.llm_client = llm_client
        self.config = config or TracingConfig()

    def trace(
        self,
        trace_index: TraceIndex,
        question: str,
        ground_truth: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Main entry point: Trace backward to find error root cause.

        Args:
            trace_index: Indexed trace to analyze
            question: The task/question that was attempted
            ground_truth: Optional ground truth for debugging

        Returns:
            Prediction dict with agent, step, error_type, confidence, reasoning
        """
        # Initialize memory with LLM
        memory = ErrorTracingMemory(llm_client=self.llm_client)

        # Start from the final turn
        final_turn_idx = len(trace_index.turns) - 1

        # Add initial observation about task failure
        memory.add_observation(
            turn_idx=final_turn_idx,
            agent=trace_index.turns[final_turn_idx].agent if trace_index.turns else "Unknown",
            content="Task execution completed but result is incorrect",
            importance="high",
            tags=["symptom", "final_turn"]
        )

        turns_examined = 0

        # Backward loop
        for turn_idx in reversed(range(final_turn_idx + 1)):
            turn = trace_index.turns[turn_idx]
            memory.mark_turn_examined(turn_idx)
            turns_examined += 1

            # Get current memory context
            memory_context = memory.get_context_for_llm()

            # Analyze this turn
            analysis = self._analyze_turn(
                turn=turn,
                turn_idx=turn_idx,
                memory_context=memory_context,
                trace_index=trace_index,
                question=question
            )

            # Process analysis results
            if analysis.is_relevant:
                # Add observations
                for obs_content in analysis.observations:
                    memory.add_observation(
                        turn_idx=turn_idx,
                        agent=turn.agent,
                        content=obs_content,
                        importance="high" if analysis.is_potential_cause else "medium"
                    )

                # Add hypothesis if this might be the cause
                if analysis.is_potential_cause and analysis.error_type:
                    memory.add_hypothesis(
                        error_type=analysis.error_type,
                        agent=turn.agent,
                        step=turn.index,
                        confidence=analysis.confidence,
                        evidence=analysis.observations,
                        reasoning=analysis.reasoning,
                        supporting_turns=[turn_idx]
                    )

            # Periodic memory maintenance
            if self.config.enable_pruning and turns_examined % self.config.pruning_interval == 0:
                memory.prune_irrelevant_observations()
                memory.merge_similar_hypotheses()

            # Check stopping criteria
            if self._should_stop_tracing(memory, turns_examined):
                break

        # Synthesize final answer from accumulated knowledge
        prediction = self._synthesize_final_answer(memory, trace_index, question)

        # Add metadata
        prediction["turns_examined"] = turns_examined
        prediction["total_turns"] = len(trace_index.turns)
        prediction["memory_summary"] = memory.get_summary()

        return prediction

    def _analyze_turn(
        self,
        turn: Turn,
        turn_idx: int,
        memory_context: str,
        trace_index: TraceIndex,
        question: str
    ) -> TurnAnalysis:
        """
        Analyze a single turn for relevance and potential error.

        Args:
            turn: Turn to analyze
            turn_idx: Turn index
            memory_context: Current memory context
            trace_index: Full trace index
            question: Original task question

        Returns:
            TurnAnalysis with findings
        """
        # Get surrounding context
        context_window = self._get_context_window(trace_index, turn_idx)

        # First: Quick relevance check
        is_relevant, relevance_score = self._check_relevance(
            turn=turn,
            memory_context=memory_context,
            context_window=context_window
        )

        if not is_relevant:
            return TurnAnalysis(
                turn_idx=turn_idx,
                is_relevant=False,
                relevance_score=relevance_score,
                observations=[],
                is_potential_cause=False
            )

        # Second: Detailed analysis for potential root cause
        (
            is_potential_cause,
            error_type,
            confidence,
            reasoning,
            observations
        ) = self._analyze_for_root_cause(
            turn=turn,
            memory_context=memory_context,
            context_window=context_window,
            question=question
        )

        return TurnAnalysis(
            turn_idx=turn_idx,
            is_relevant=True,
            relevance_score=relevance_score,
            observations=observations,
            is_potential_cause=is_potential_cause,
            error_type=error_type,
            confidence=confidence,
            reasoning=reasoning
        )

    def _check_relevance(
        self,
        turn: Turn,
        memory_context: str,
        context_window: str
    ) -> Tuple[bool, float]:
        """
        Quick check if turn is relevant to error investigation.

        Args:
            turn: Turn to check
            memory_context: Current findings summary
            context_window: Surrounding turns for context

        Returns:
            (is_relevant, relevance_score)
        """
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

        try:
            response = self.llm_client.invoke(prompt, max_tokens=200)

            # Parse JSON response
            result = json.loads(response.strip())
            is_relevant = result.get("is_relevant", False)
            score = float(result.get("score", 0.0))

            return is_relevant, score

        except (json.JSONDecodeError, Exception) as e:
            # Fallback: Consider all turns with actions as relevant
            has_action = turn.action is not None or len(turn.tool_calls) > 0
            return has_action, 0.5

    def _analyze_for_root_cause(
        self,
        turn: Turn,
        memory_context: str,
        context_window: str,
        question: str
    ) -> Tuple[bool, Optional[str], float, str, List[str]]:
        """
        Detailed analysis: Is this turn the root cause?

        Args:
            turn: Turn to analyze
            memory_context: Current findings
            context_window: Surrounding turns
            question: Original task

        Returns:
            (is_potential_cause, error_type, confidence, reasoning, observations)
        """
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

        try:
            response = self.llm_client.invoke(prompt, max_tokens=500)

            # Parse JSON response
            result = json.loads(response.strip())

            is_potential_cause = result.get("is_potential_cause", False)
            error_type = result.get("error_type")
            confidence = float(result.get("confidence", 0.0))
            reasoning = result.get("reasoning", "")
            observations = result.get("observations", [])

            return is_potential_cause, error_type, confidence, reasoning, observations

        except (json.JSONDecodeError, Exception) as e:
            # Fallback: Mark as not a root cause
            return False, None, 0.0, f"Analysis failed: {str(e)}", []

    def _get_context_window(
        self,
        trace_index: TraceIndex,
        center_idx: int,
        radius: Optional[int] = None
    ) -> str:
        """
        Get surrounding turns for context.

        Args:
            trace_index: Full trace
            center_idx: Center turn index
            radius: How many turns before/after (default: config value)

        Returns:
            Formatted context window
        """
        radius = radius or self.config.context_window_radius

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

    def _should_stop_tracing(
        self,
        memory: ErrorTracingMemory,
        turns_examined: int
    ) -> bool:
        """
        Determine if we should stop tracing backward.

        Stopping criteria:
        1. High-confidence hypothesis found (confidence > threshold)
        2. Examined enough turns (turns > max_turns_to_examine)
        3. Examined minimum turns AND have reasonable hypothesis

        Args:
            memory: Current memory state
            turns_examined: Number of turns examined so far

        Returns:
            True if should stop, False otherwise
        """
        # Don't stop too early
        if turns_examined < self.config.min_turns_before_stopping:
            return False

        # Stop if we've examined too many turns
        if turns_examined >= self.config.max_turns_to_examine:
            return True

        # Stop if we have high-confidence hypothesis
        top_hypothesis = memory.get_top_hypothesis()
        if top_hypothesis and top_hypothesis.confidence >= self.config.confidence_threshold:
            return True

        # Stop if we have multiple converging hypotheses
        high_conf_hyps = memory.get_hypotheses(min_confidence=0.6)
        if len(high_conf_hyps) >= 2:
            # Check if they point to similar location (within 5 steps)
            if len(set(h.step for h in high_conf_hyps)) <= 1:
                return True

        return False

    def _synthesize_final_answer(
        self,
        memory: ErrorTracingMemory,
        trace_index: TraceIndex,
        question: str
    ) -> Dict[str, Any]:
        """
        Synthesize final prediction from accumulated knowledge.

        Args:
            memory: Complete memory after tracing
            trace_index: Full trace
            question: Original task

        Returns:
            Final prediction dictionary
        """
        # Get top hypothesis
        top_hypothesis = memory.get_top_hypothesis()

        if not top_hypothesis:
            # Fallback: No hypothesis found, make best guess
            return self._fallback_prediction(memory, trace_index)

        # Build comprehensive reasoning from memory
        memory_summary = memory.get_context_for_llm()

        # Use LLM to synthesize final answer
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

        try:
            response = self.llm_client.invoke(prompt, max_tokens=400)
            result = json.loads(response.strip())

            return {
                "mistake_agent": result.get("mistake_agent", top_hypothesis.agent),
                "mistake_step": result.get("mistake_step", top_hypothesis.step),
                "mistake_type": result.get("mistake_type", top_hypothesis.error_type),
                "mistake_reason": result.get("mistake_reason", top_hypothesis.reasoning),
                "confidence": result.get("confidence", top_hypothesis.confidence),
                "supporting_evidence": top_hypothesis.evidence,
                "method": "backward_tracing"
            }

        except (json.JSONDecodeError, Exception) as e:
            # Fallback to hypothesis directly
            return {
                "mistake_agent": top_hypothesis.agent,
                "mistake_step": top_hypothesis.step,
                "mistake_type": top_hypothesis.error_type,
                "mistake_reason": top_hypothesis.reasoning,
                "confidence": top_hypothesis.confidence,
                "supporting_evidence": top_hypothesis.evidence,
                "method": "backward_tracing"
            }

    def _fallback_prediction(
        self,
        memory: ErrorTracingMemory,
        trace_index: TraceIndex
    ) -> Dict[str, Any]:
        """
        Generate fallback prediction when no hypothesis found.

        Args:
            memory: Memory state
            trace_index: Full trace

        Returns:
            Fallback prediction
        """
        # Look for high-importance observations
        high_imp_obs = memory.get_observations_by_importance("high")

        if high_imp_obs:
            # Use most recent high-importance observation
            latest_obs = high_imp_obs[-1]
            return {
                "mistake_agent": latest_obs.agent,
                "mistake_step": latest_obs.turn_idx,
                "mistake_type": "unknown",
                "mistake_reason": f"Based on observation: {latest_obs.content}",
                "confidence": 0.3,
                "method": "backward_tracing_fallback"
            }

        # Ultimate fallback: last turn
        final_turn = trace_index.turns[-1] if trace_index.turns else None
        if final_turn:
            return {
                "mistake_agent": final_turn.agent,
                "mistake_step": final_turn.index,
                "mistake_type": "unknown",
                "mistake_reason": "No clear hypothesis found; defaulting to final turn",
                "confidence": 0.1,
                "method": "backward_tracing_fallback"
            }

        # Absolute fallback
        return {
            "mistake_agent": "Unknown",
            "mistake_step": 0,
            "mistake_type": "unknown",
            "mistake_reason": "Unable to determine error cause",
            "confidence": 0.0,
            "method": "backward_tracing_fallback"
        }
