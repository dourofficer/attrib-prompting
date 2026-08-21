"""
ErrorTracingMemory - Working memory for backward error tracing

This component accumulates findings as we trace backward through execution history.
It stores observations, hypotheses, and evidence, and provides intelligent
summarization and pruning capabilities.

Design: Hybrid approach
- Core: Pure data structure (works without LLM)
- Smart: Optional LLM-based features for summarization and pruning
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import json


@dataclass
class Observation:
    """Single observation made during backward tracing"""
    turn_idx: int
    agent: str
    content: str
    importance: str  # "high", "medium", "low"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_idx": self.turn_idx,
            "agent": self.agent,
            "content": self.content,
            "importance": self.importance,
            "timestamp": self.timestamp,
            "tags": self.tags
        }

    def __str__(self) -> str:
        return f"Turn {self.turn_idx} [{self.agent}]: {self.content[:80]}..."


@dataclass
class Hypothesis:
    """Hypothesis about the root cause of the error"""
    error_type: str
    agent: str
    step: int
    confidence: float  # 0.0 to 1.0
    evidence: List[str]
    reasoning: str = ""
    supporting_turns: List[int] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_type": self.error_type,
            "agent": self.agent,
            "step": self.step,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "reasoning": self.reasoning,
            "supporting_turns": self.supporting_turns,
            "timestamp": self.timestamp
        }

    def __str__(self) -> str:
        return f"{self.error_type} in {self.agent} at step {self.step} (conf={self.confidence:.2f})"


class ErrorTracingMemory:
    """
    Working memory for backward error tracing.

    Stores and organizes findings accumulated during backward traversal:
    - Observations: What we noticed at each turn
    - Hypotheses: Potential root causes with confidence scores
    - Evidence: Supporting information for each hypothesis
    - Trace path: Sequence of turns we've examined

    Features:
    - Basic: Works without LLM (heuristic summarization)
    - Smart: Uses LLM for intelligent summarization and pruning
    """

    def __init__(self, llm_client=None, max_observations: int = 50):
        """
        Initialize error tracing memory.

        Args:
            llm_client: Optional LLM client for smart features
            max_observations: Maximum observations to keep (triggers pruning)
        """
        # Core data structures
        self.observations: List[Observation] = []
        self.hypotheses: List[Hypothesis] = []
        self.trace_path: List[int] = []  # Turn indices we've examined

        # Configuration
        self.llm_client = llm_client
        self.max_observations = max_observations

        # Cached summaries
        self._cached_context: Optional[str] = None
        self._cache_valid: bool = False

    # ========================================================================
    # CORE OPERATIONS (No LLM)
    # ========================================================================

    def add_observation(
        self,
        turn_idx: int,
        agent: str,
        content: str,
        importance: str = "medium",
        tags: Optional[List[str]] = None
    ) -> None:
        """
        Add an observation from backward tracing.

        Args:
            turn_idx: Turn index where observation was made
            agent: Agent involved in this turn
            content: Observation content
            importance: "high", "medium", or "low"
            tags: Optional tags for categorization
        """
        obs = Observation(
            turn_idx=turn_idx,
            agent=agent,
            content=content,
            importance=importance,
            tags=tags or []
        )
        self.observations.append(obs)
        self._cache_valid = False

        # Auto-prune if we exceed max
        if len(self.observations) > self.max_observations:
            self._auto_prune()

    def add_hypothesis(
        self,
        error_type: str,
        agent: str,
        step: int,
        confidence: float,
        evidence: List[str],
        reasoning: str = "",
        supporting_turns: Optional[List[int]] = None
    ) -> None:
        """
        Add a hypothesis about the root cause.

        Args:
            error_type: Type of error (e.g., "logic_error", "incomplete_verification")
            agent: Agent that made the error
            step: Step number where error occurred
            confidence: Confidence score (0.0 to 1.0)
            evidence: List of evidence supporting this hypothesis
            reasoning: Explanation of why this is the error
            supporting_turns: Turn indices that support this hypothesis
        """
        hyp = Hypothesis(
            error_type=error_type,
            agent=agent,
            step=step,
            confidence=confidence,
            evidence=evidence,
            reasoning=reasoning,
            supporting_turns=supporting_turns or []
        )
        self.hypotheses.append(hyp)
        self._cache_valid = False

        # Keep hypotheses sorted by confidence (highest first)
        self.hypotheses.sort(key=lambda h: h.confidence, reverse=True)

    def update_hypothesis(self, hypothesis_idx: int, **updates) -> None:
        """
        Update an existing hypothesis.

        Args:
            hypothesis_idx: Index of hypothesis to update
            **updates: Fields to update (confidence, evidence, etc.)
        """
        if 0 <= hypothesis_idx < len(self.hypotheses):
            hyp = self.hypotheses[hypothesis_idx]
            for key, value in updates.items():
                if hasattr(hyp, key):
                    setattr(hyp, key, value)

            # Re-sort after update
            self.hypotheses.sort(key=lambda h: h.confidence, reverse=True)
            self._cache_valid = False

    def mark_turn_examined(self, turn_idx: int) -> None:
        """Mark a turn as examined in our trace path."""
        if turn_idx not in self.trace_path:
            self.trace_path.append(turn_idx)

    def get_top_hypothesis(self) -> Optional[Hypothesis]:
        """Get the hypothesis with highest confidence."""
        return self.hypotheses[0] if self.hypotheses else None

    def get_hypotheses(self, min_confidence: float = 0.0) -> List[Hypothesis]:
        """Get all hypotheses above a confidence threshold."""
        return [h for h in self.hypotheses if h.confidence >= min_confidence]

    def get_observations_by_importance(self, importance: str) -> List[Observation]:
        """Get observations filtered by importance level."""
        return [o for o in self.observations if o.importance == importance]

    def get_observations_for_turn(self, turn_idx: int) -> List[Observation]:
        """Get all observations for a specific turn."""
        return [o for o in self.observations if o.turn_idx == turn_idx]

    # ========================================================================
    # CONTEXT GENERATION (Hybrid: Heuristic + Optional LLM)
    # ========================================================================

    def get_context_for_llm(self, max_tokens: int = 2000) -> str:
        """
        Get condensed context summary for next LLM call.

        Uses LLM-based summarization if available, otherwise falls back
        to heuristic summarization.

        Args:
            max_tokens: Target maximum tokens (approximate)

        Returns:
            Condensed summary of current findings
        """
        # Return cached if valid
        if self._cache_valid and self._cached_context:
            return self._cached_context

        # Generate new context
        if self.llm_client and len(self.observations) > 5:
            context = self._llm_summarize_context(max_tokens)
        else:
            context = self._heuristic_summarize_context(max_tokens)

        # Cache and return
        self._cached_context = context
        self._cache_valid = True
        return context

    def _heuristic_summarize_context(self, max_tokens: int) -> str:
        """
        Simple heuristic summarization (no LLM).

        Strategy:
        1. Include all high-confidence hypotheses
        2. Include recent high-importance observations
        3. Include observation count and turn range
        """
        parts = []

        # Summary header
        parts.append(f"=== Traced {len(self.trace_path)} turns ===\n")

        # Hypotheses (highest confidence first)
        if self.hypotheses:
            parts.append("HYPOTHESES:")
            for i, hyp in enumerate(self.hypotheses[:3], 1):  # Top 3
                parts.append(
                    f"{i}. {hyp.error_type} in {hyp.agent} at step {hyp.step} "
                    f"(confidence: {hyp.confidence:.2f})"
                )
                if hyp.reasoning:
                    parts.append(f"   Reasoning: {hyp.reasoning[:100]}")
            parts.append("")

        # Key observations (high importance + recent)
        high_importance = [o for o in self.observations if o.importance == "high"]
        if high_importance:
            parts.append("KEY OBSERVATIONS:")
            for obs in high_importance[-5:]:  # Last 5 high-importance
                parts.append(f"- Turn {obs.turn_idx} [{obs.agent}]: {obs.content[:100]}")
            parts.append("")

        # Recent observations
        if len(self.observations) > len(high_importance):
            parts.append("RECENT OBSERVATIONS:")
            recent = [o for o in self.observations[-3:] if o not in high_importance]
            for obs in recent:
                parts.append(f"- Turn {obs.turn_idx} [{obs.agent}]: {obs.content[:80]}")

        return "\n".join(parts)

    def _llm_summarize_context(self, max_tokens: int) -> str:
        """
        LLM-based intelligent summarization.

        Uses LLM to condense observations and hypotheses into
        a focused summary highlighting key patterns and findings.
        """
        # Build prompt with all observations and hypotheses
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
        for obs in self.observations[-10:]:  # Last 10 observations
            prompt_parts.append(
                f"Turn {obs.turn_idx} [{obs.agent}] ({obs.importance}): {obs.content[:150]}"
            )

        prompt_parts.append("\n=== SUMMARY ===")
        prompt_parts.append("Provide a concise summary:")

        prompt = "\n".join(prompt_parts)

        # Call LLM for summarization
        try:
            response = self.llm_client.invoke(prompt, max_tokens=500)
            return response.strip()
        except Exception as e:
            # Fallback to heuristic if LLM fails
            return self._heuristic_summarize_context(max_tokens)

    # ========================================================================
    # SMART FEATURES (LLM-based)
    # ========================================================================

    def prune_irrelevant_observations(self) -> int:
        """
        Remove observations that aren't contributing to current hypotheses.

        Uses LLM to determine relevance if available, otherwise uses
        simple heuristics (age + importance).

        Returns:
            Number of observations removed
        """
        if self.llm_client and len(self.observations) > 10:
            return self._llm_prune_observations()
        else:
            return self._heuristic_prune_observations()

    def _heuristic_prune_observations(self) -> int:
        """
        Simple heuristic pruning.

        Strategy:
        - Keep all "high" importance observations
        - Keep recent observations (last 20)
        - Remove old "low" importance observations
        """
        if len(self.observations) <= 20:
            return 0

        initial_count = len(self.observations)

        # Filter: keep high importance or recent
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

    def _llm_prune_observations(self) -> int:
        """
        LLM-guided intelligent pruning.

        Asks LLM to identify which observations are relevant to
        current hypotheses.
        """
        if not self.hypotheses:
            # No hypotheses yet, can't prune intelligently
            return self._heuristic_prune_observations()

        top_hypothesis = self.get_top_hypothesis()

        # Build prompt
        prompt_parts = [
            "You are analyzing error traces. We have a hypothesis about the error:",
            f"HYPOTHESIS: {top_hypothesis.error_type} in {top_hypothesis.agent} at step {top_hypothesis.step}",
            f"Reasoning: {top_hypothesis.reasoning}\n",
            "Which of these observations are RELEVANT to this hypothesis?",
            "Respond with ONLY the observation numbers that are relevant (comma-separated).\n",
            "OBSERVATIONS:"
        ]

        for i, obs in enumerate(self.observations, 1):
            prompt_parts.append(
                f"{i}. Turn {obs.turn_idx} [{obs.agent}]: {obs.content[:100]}"
            )

        prompt_parts.append("\nRelevant observation numbers:")
        prompt = "\n".join(prompt_parts)

        try:
            response = self.llm_client.invoke(prompt, max_tokens=200)

            # Parse response
            relevant_nums = set()
            for token in response.replace(",", " ").split():
                try:
                    num = int(token.strip())
                    if 1 <= num <= len(self.observations):
                        relevant_nums.add(num - 1)  # Convert to 0-indexed
                except ValueError:
                    continue

            # Keep only relevant observations
            initial_count = len(self.observations)
            self.observations = [
                o for i, o in enumerate(self.observations)
                if i in relevant_nums or self.observations[i].importance == "high"
            ]

            removed = initial_count - len(self.observations)
            if removed > 0:
                self._cache_valid = False

            return removed

        except Exception as e:
            # Fallback to heuristic if LLM fails
            return self._heuristic_prune_observations()

    def _auto_prune(self) -> None:
        """Automatically prune when observations exceed max."""
        self.prune_irrelevant_observations()

    def merge_similar_hypotheses(self) -> int:
        """
        Merge similar hypotheses to reduce redundancy.

        Uses LLM to identify similar hypotheses if available.

        Returns:
            Number of hypotheses merged
        """
        if len(self.hypotheses) <= 1:
            return 0

        if self.llm_client:
            return self._llm_merge_hypotheses()
        else:
            return self._heuristic_merge_hypotheses()

    def _heuristic_merge_hypotheses(self) -> int:
        """
        Simple heuristic merging.

        Strategy: Merge hypotheses with same error_type and agent.
        """
        initial_count = len(self.hypotheses)

        merged = []
        seen = set()

        for hyp in self.hypotheses:
            key = (hyp.error_type, hyp.agent)
            if key not in seen:
                seen.add(key)
                merged.append(hyp)
            else:
                # Merge into existing
                existing = next(h for h in merged if (h.error_type, h.agent) == key)
                # Keep higher confidence
                if hyp.confidence > existing.confidence:
                    existing.confidence = hyp.confidence
                # Merge evidence
                existing.evidence = list(set(existing.evidence + hyp.evidence))
                # Merge supporting turns
                existing.supporting_turns = list(set(existing.supporting_turns + hyp.supporting_turns))

        self.hypotheses = merged
        removed = initial_count - len(self.hypotheses)

        if removed > 0:
            self._cache_valid = False

        return removed

    def _llm_merge_hypotheses(self) -> int:
        """
        LLM-guided hypothesis merging.

        Asks LLM to identify which hypotheses are essentially the same.
        """
        # Build prompt
        prompt_parts = [
            "You are analyzing error hypotheses. Are any of these essentially describing the same error?",
            "If yes, suggest which to merge.\n",
            "HYPOTHESES:"
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

        try:
            response = self.llm_client.invoke(prompt, max_tokens=200)

            if "none" in response.lower():
                return 0

            # Parse merge groups (e.g., "1,2" means merge hypotheses 1 and 2)
            # For now, do simple heuristic merge
            # TODO: Implement full LLM-guided merging
            return self._heuristic_merge_hypotheses()

        except Exception as e:
            return self._heuristic_merge_hypotheses()

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def get_summary(self) -> Dict[str, Any]:
        """Get a complete summary of memory state."""
        return {
            "observations_count": len(self.observations),
            "hypotheses_count": len(self.hypotheses),
            "turns_examined": len(self.trace_path),
            "top_hypothesis": self.get_top_hypothesis().to_dict() if self.hypotheses else None,
            "high_importance_observations": len(self.get_observations_by_importance("high")),
            "medium_importance_observations": len(self.get_observations_by_importance("medium")),
            "low_importance_observations": len(self.get_observations_by_importance("low"))
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize memory to dictionary."""
        return {
            "observations": [o.to_dict() for o in self.observations],
            "hypotheses": [h.to_dict() for h in self.hypotheses],
            "trace_path": self.trace_path,
            "summary": self.get_summary()
        }

    def save_to_file(self, filepath: str) -> None:
        """Save memory state to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def __repr__(self) -> str:
        return (
            f"ErrorTracingMemory("
            f"observations={len(self.observations)}, "
            f"hypotheses={len(self.hypotheses)}, "
            f"turns_examined={len(self.trace_path)})"
        )
