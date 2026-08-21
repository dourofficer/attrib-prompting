"""
Simplified LLM-Based Multi-Agent System

Architecture:
- Analyzer Agent (LLM): Analyzes traces, identifies errors
- Verifier Agent (LLM): Verifies error hypotheses
- Memory Manager (Algorithm): VBW + RFI-Δ memory management

NEW: Backward Tracing Mode (Optional)
- TraceIndex: Full trace indexing for fast access
- BackwardTracer: Traces backward from symptom to root cause
- ErrorTracingMemory: Accumulates findings during tracing
"""

import os
import json
from typing import Dict, List, Any, Optional
import litellm
from litellm import completion

# Import algorithmic components
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import EPMManager, ErrorSignature, ErrorSpan, ErrorPatch, Evidence, ErrorHypothesis, ERROR_FAMILIES, get_all_mast_families
from utils import extract_agent_names, format_agent_options, get_trace_length

# Import backward tracing components
from .trace_index import TraceIndex
from .backward_tracer import BackwardTracer, TracingConfig
from .failure_mode_detector import FailureModeDetector


class LLMClient:
    """
    LLM client wrapper for backward tracing.

    Provides simple invoke() interface expected by BackwardTracer.
    """

    def __init__(self, model_id: str, temperature: float = 0.7):
        self.model_id = model_id
        self.temperature = temperature

    def invoke(self, prompt: str, max_tokens: int = 1000) -> str:
        """
        Simple LLM invocation.

        Args:
            prompt: The prompt to send
            max_tokens: Maximum tokens in response

        Returns:
            LLM response text
        """
        try:
            response = completion(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=self.temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"LLM invocation failed: {e}")


class LLMConfig:
    """LLM configuration"""
    def __init__(self, config_path: str = "config.yaml"):
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)

        raw_model_id = config['model']['model_id']
        self.max_tokens = config['model'].get('max_tokens', 4000)
        self.temperature = config['model'].get('temperature', 0.7)

        # NEW: Backward tracing configuration
        self.use_backward_tracing = config.get('backward_tracing', {}).get('enabled', False)

        # LLM access is handled by LiteLLM. Provide credentials via the
        # standard environment variables for your provider, e.g.:
        #   OPENAI_API_KEY, ANTHROPIC_API_KEY, ...
        # See https://docs.litellm.ai/docs/providers for details.
        self.model_id = raw_model_id

        print(f"✓ LLM Config loaded: {self.model_id}")
        if self.use_backward_tracing:
            print(f"✓ Backward Tracing: ENABLED")


class AnalyzerAgent:
    """
    LLM-based agent that analyzes failed traces

    Uses LLM to:
    - Understand the trace context
    - Identify where the error occurred
    - Classify the error type
    - Provide reasoning

    Supports TWO modes:
    1. LEGACY: Truncated history (last 15 turns)
    2. NEW: Backward tracing (full trace access)
    """

    def __init__(self, llm_config: LLMConfig):
        self.config = llm_config
        # Use MAST taxonomy (14 empirically-validated failure modes)
        # Grouped by: FC1 (Design/Spec), FC2 (Coordination), FC3 (Verification)
        self.error_families = get_all_mast_families()

        # BASELINE: No failure mode detector
        # self.failure_detector = FailureModeDetector()  # Disabled for baseline

        # Initialize backward tracer if enabled
        self.backward_tracer = None
        if llm_config.use_backward_tracing:
            llm_client = LLMClient(llm_config.model_id, llm_config.temperature)
            tracing_config = TracingConfig(
                max_turns_to_examine=100,
                context_window_radius=5,
                confidence_threshold=0.75,
                min_turns_before_stopping=10
            )
            self.backward_tracer = BackwardTracer(llm_client, tracing_config)

    def analyze_trace(self, trace: Dict[str, Any], similar_patterns: List[Dict] = None) -> Dict[str, Any]:
        """
        Analyze a failed trace using LLM with optional position guidance

        Args:
            trace: The failed execution trace
            similar_patterns: Optional list of similar patterns from memory for position hints

        Returns:
        {
            "error_step": (start, end),
            "error_family": str,
            "error_reason": str,
            "confidence": float,
            "tool_used": str,
            "context": Dict
        }
        """
        # NEW: Use backward tracing if enabled
        if self.backward_tracer:
            return self._analyze_with_backward_tracing(trace)

        # LEGACY: Use truncated history approach
        return self._analyze_with_truncated_history(trace, similar_patterns)

    def _analyze_with_backward_tracing(self, trace: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze trace using backward tracing (FULL TRACE ACCESS).

        This is the NEW approach that:
        - Accesses the complete trace history
        - Traces backward from symptom to root cause
        - Accumulates findings progressively
        """
        try:
            # Build TraceIndex from trace
            trace_index = TraceIndex(trace)

            # Get question/task
            question = trace.get('question', 'Unknown task')

            # Run backward tracing
            prediction = self.backward_tracer.trace(trace_index, question)

            # Convert to expected format
            return {
                "error_step": (prediction.get('mistake_step', 0), prediction.get('mistake_step', 0)),
                "error_family": prediction.get('mistake_type', 'unknown'),
                "error_reason": prediction.get('mistake_reason', 'No reason provided'),
                "confidence": prediction.get('confidence', 0.5),
                "tool_used": prediction.get('mistake_agent', 'unknown'),
                "context": {
                    "method": "backward_tracing",
                    "turns_examined": prediction.get('turns_examined', 0),
                    "total_turns": prediction.get('total_turns', 0),
                    "memory_summary": prediction.get('memory_summary', {})
                }
            }

        except Exception as e:
            print(f"Backward tracing failed: {e}")
            # Fallback to legacy method
            return self._analyze_with_truncated_history(trace, None)

    def _analyze_with_truncated_history(self, trace: Dict[str, Any], similar_patterns: List[Dict] = None) -> Dict[str, Any]:
        """
        Analyze trace using LEGACY truncated history approach (last 15 turns).

        This is the OLD approach with the 15-turn limitation.
        """
        # Build prompt with similar patterns for position guidance
        prompt = self._build_analysis_prompt(trace, similar_patterns)

        # Call LLM
        try:
            response = completion(
                model=self.config.model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=self.config.temperature
            )

            # Parse response
            result = self._parse_llm_response(response.choices[0].message.content)
            result["context"]["method"] = "truncated_history"
            return result

        except Exception as e:
            print(f"LLM call failed: {e}")
            return {
                "error_step": (0, 0),
                "error_family": "unknown",
                "error_reason": f"Analysis failed: {str(e)}",
                "confidence": 0.0,
                "tool_used": "unknown",
                "context": {}
            }

    def _build_analysis_prompt(self, trace: Dict[str, Any], similar_patterns: List[Dict] = None) -> str:
        """Build simple baseline prompt - no Phase 1 enhancements"""
        # Extract conversation history - BASELINE: Last 15 turns only
        history = trace.get('history', [])
        task_goal = trace.get('question', 'Unknown task')

        # Get agent names for validation - support both data formats
        agent_names = set()
        for turn in history:
            agent = turn.get('name')
            if not agent:  # Fall back to extracting from 'role' field
                role_str = turn.get('role', '')
                agent = role_str.split('(')[0].strip() if role_str else None
                if agent and agent.lower() not in ['human', 'assistant', 'user', 'system', '']:
                    agent_names.add(agent)
            elif agent and agent != 'Unknown':
                agent_names.add(agent)
        agent_names = list(agent_names) if agent_names else ['Unknown']
        agent_options = format_agent_options(agent_names) if agent_names else "No specific agents identified - use 'unknown'"

        # Format history - BASELINE: Simple format, last 15 turns, 500 char limit
        conversation = []
        for idx, turn in enumerate(history[-15:]):  # Last 15 turns only
            # Support both data formats:
            # - Automatic format: agent name in 'name' field
            # - Hand-crafted format: agent name in 'role' field (e.g., "Orchestrator (thought)")
            agent = turn.get('name')
            if not agent:  # Fall back to extracting from 'role' field
                role_str = turn.get('role', '')
                # Extract agent name before any parenthesis (e.g., "Orchestrator (thought)" -> "Orchestrator")
                agent = role_str.split('(')[0].strip() if role_str else 'Unknown'
                # Handle standard roles like "human", "assistant"
                if agent.lower() in ['human', 'assistant', 'user', 'system']:
                    agent = 'Unknown'
            if not agent or agent == '':
                agent = 'Unknown'

            role = turn.get('role', 'unknown')
            content = turn.get('content', '')[:500]  # 500 char truncation

            conversation.append(f"""Turn {idx}:
  Agent: {agent}
  Content: {content}
---""")

        conv_text = "\n".join(conversation)

        # Get error families for taxonomy
        mast_descriptions = []
        for family in self.error_families:
            desc = ERROR_FAMILIES.get(family, family)
            mast_descriptions.append(f"- {family}: {desc}")

        error_types_text = "\n".join(mast_descriptions)

        # Build simple baseline prompt
        prompt = f"""You are an expert at analyzing failed multi-agent execution traces.

TASK:
{task_goal}

CONVERSATION HISTORY (last 15 turns):
{conv_text}

{agent_options}

Based on the above trace, identify the error:
1. Which turn(s) contain the error?
2. Which agent caused the error?
3. What type of error occurred?
4. Why did it happen?

ERROR TAXONOMY:
{error_types_text}

RESPOND IN JSON FORMAT:
{{
  "error_start_turn": <number>,
  "error_end_turn": <number>,
  "error_family": "<type from taxonomy above>",
  "error_agent": "<agent name - must be from available agents list>",
  "error_reason": "<explanation>",
  "confidence": <0.0 to 1.0>
}}

Only output valid JSON, no other text."""

        return prompt

    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM JSON response - handles both single step and range formats"""
        try:
            # Extract JSON from response
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_str = response_text.split("```")[1].strip()
            else:
                json_str = response_text.strip()

            data = json.loads(json_str)

            # Support both new "error_agent" and legacy "tool_or_component"
            agent = data.get("error_agent") or data.get("tool_or_component", "unknown")

            # Parse error step - support multiple formats
            if "error_step" in data:
                # Option 1: Single precise step (PREFERRED)
                step = data["error_step"]
                error_step = (step, step)  # Convert to range format for consistency
            elif "error_step_start" in data and "error_step_end" in data:
                # Option 2: Range format
                error_step = (data["error_step_start"], data["error_step_end"])
            elif "error_start_turn" in data or "error_end_turn" in data:
                # Legacy format (backward compatibility)
                error_step = (data.get("error_start_turn", 0), data.get("error_end_turn", 0))
            else:
                # Fallback
                error_step = (0, 0)

            return {
                "error_step": error_step,
                "error_family": data.get("error_family", "unknown"),
                "error_reason": data.get("error_reason", "No reason provided"),
                "confidence": float(data.get("confidence", 0.5)),
                "tool_used": agent,  # Use extracted agent name
                "context": data
            }
        except Exception as e:
            print(f"Failed to parse LLM response: {e}")
            print(f"Response was: {response_text[:200]}")
            return {
                "error_step": (0, 0),
                "error_family": "unknown",
                "error_reason": "Failed to parse LLM response",
                "confidence": 0.0,
                "tool_used": "unknown",
                "context": {}
            }

    # ==== PHASE 1.3 HELPER METHODS ====

    def _extract_agent_roles(self, history: List[Dict]) -> Dict[str, str]:
        """Extract agent names and their roles from trace"""
        agent_roles = {}
        for turn in history:
            agent = turn.get('name')
            role = turn.get('role', 'Unknown')
            if agent and agent not in agent_roles:
                agent_roles[agent] = role
        return agent_roles

    def _detect_temporal_patterns(self, history: List[Dict]) -> List[str]:
        """Detect temporal patterns: loops, stalls, rushes"""
        patterns = []

        if len(history) < 3:
            return patterns

        # Pattern 1: Step repetition (same step number multiple times)
        steps = [turn.get('step', idx) for idx, turn in enumerate(history)]
        step_counts = {}
        for step in steps:
            step_counts[step] = step_counts.get(step, 0) + 1

        repeated_steps = [step for step, count in step_counts.items() if count > 1]
        if repeated_steps:
            patterns.append(f"Step repetition detected: {len(repeated_steps)} steps repeated (possible loop/retry)")

        # Pattern 2: Very short trace (possible premature termination)
        if len(history) < 5:
            patterns.append(f"Very short trace ({len(history)} turns) - possible premature termination")

        # Pattern 3: Very long trace (possible inefficiency or loop)
        if len(history) > 50:
            patterns.append(f"Very long trace ({len(history)} turns) - possible inefficiency or stuck in loop")

        # Pattern 4: Agent switching frequency
        agent_switches = 0
        prev_agent = None
        for turn in history:
            agent = turn.get('name')
            if prev_agent and prev_agent != agent:
                agent_switches += 1
            prev_agent = agent

        avg_switches_per_10_turns = (agent_switches / len(history)) * 10
        if avg_switches_per_10_turns < 2:
            patterns.append(f"Low agent collaboration (only {agent_switches} handoffs in {len(history)} turns)")
        elif avg_switches_per_10_turns > 8:
            patterns.append(f"High agent switching ({agent_switches} handoffs) - possible confusion/coordination issues")

        return patterns

    def _build_interaction_summary(self, history: List[Dict]) -> str:
        """Build summary of agent interactions"""
        if not history:
            return "No interactions"

        # Track agent sequence
        agent_sequence = []
        prev_agent = None
        for turn in history:
            agent = turn.get('name', 'Unknown')
            if agent != prev_agent:
                agent_sequence.append(agent)
                prev_agent = agent

        # Build summary
        unique_agents = len(set(agent_sequence))
        total_turns = len(history)
        handoffs = len(agent_sequence) - 1

        # Format sequence (limit to first 10 transitions)
        sequence_str = " → ".join(agent_sequence[:min(10, len(agent_sequence))])
        if len(agent_sequence) > 10:
            sequence_str += f" ... ({len(agent_sequence) - 10} more transitions)"

        summary = f"{unique_agents} agents, {total_turns} turns, {handoffs} handoffs\nSequence: {sequence_str}"

        return summary


class VerifierAgent:
    """
    LLM-based agent that verifies error hypotheses

    Uses LLM to:
    - Review the analyzer's findings
    - Provide additional evidence
    - Suggest fixes
    """

    def __init__(self, llm_config: LLMConfig):
        self.config = llm_config

    def verify_hypothesis(self,
                         trace: Dict[str, Any],
                         analysis: Dict[str, Any]) -> Evidence:
        """
        Verify an error hypothesis using LLM

        Returns Evidence with verification details
        """
        prompt = self._build_verification_prompt(trace, analysis)

        try:
            response = completion(
                model=self.config.model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500,
                temperature=0.3  # Lower temp for verification
            )

            result = self._parse_verification_response(response.choices[0].message.content)
            return result

        except Exception as e:
            print(f"Verification failed: {e}")
            return Evidence(
                ablation_gain=0.0,
                static_hits=0,
                replays=0,
                verified=False,
                probe_details={"error": str(e)}
            )

    def _build_verification_prompt(self, trace: Dict[str, Any], analysis: Dict[str, Any]) -> str:
        """Build verification prompt"""
        history = trace.get('history', [])
        error_start, error_end = analysis['error_step']

        # Get context around error
        context_turns = []
        for idx in range(max(0, error_start - 2), min(len(history), error_end + 3)):
            turn = history[idx]
            role = turn.get('role', 'unknown')
            content = turn.get('content', '')[:300]
            marker = " <<<ERROR TURN" if error_start <= idx <= error_end else ""
            context_turns.append(f"Turn {idx} [{role}]{marker}: {content}")

        context_text = "\n".join(context_turns)

        prompt = f"""You are verifying an error analysis.

PROPOSED ERROR ANALYSIS:
- Error turns: {error_start} to {error_end}
- Error type: {analysis['error_family']}
- Reason: {analysis['error_reason']}
- Confidence: {analysis['confidence']}

TRACE CONTEXT:
{context_text}

TASK: Verify if this analysis is correct by:
1. Checking if the error turns make sense
2. Assessing if the error type is appropriate
3. Evaluating the reasoning quality
4. Estimating impact if this error were fixed (0.0 to 1.0)
5. Suggesting a fix (guard, rewrite, or prompt hint)

RESPOND IN JSON:
{{
  "verified": <true/false>,
  "verification_confidence": <0.0 to 1.0>,
  "impact_if_fixed": <0.0 to 1.0>,
  "evidence_count": <number of supporting evidence pieces>,
  "suggested_fix": {{
    "type": "guard|rewrite|hint",
    "fix_text": "<actual fix>"
  }},
  "verification_notes": "<why verified or not>"
}}

Only output valid JSON."""

        return prompt

    def _parse_verification_response(self, response_text: str) -> Evidence:
        """Parse verification response into Evidence"""
        try:
            # Extract JSON
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_str = response_text.split("```")[1].strip()
            else:
                json_str = response_text.strip()

            data = json.loads(json_str)

            return Evidence(
                ablation_gain=float(data.get("impact_if_fixed", 0.0)),
                static_hits=int(data.get("evidence_count", 0)),
                replays=1,
                verified=bool(data.get("verified", False)),
                probe_details={
                    "verification_confidence": data.get("verification_confidence", 0.0),
                    "suggested_fix": data.get("suggested_fix", {}),
                    "notes": data.get("verification_notes", "")
                }
            )
        except Exception as e:
            print(f"Failed to parse verification: {e}")
            return Evidence(
                ablation_gain=0.0,
                static_hits=0,
                replays=0,
                verified=False,
                probe_details={"parse_error": str(e)}
            )


class SimplifiedMAS:
    """
    Simplified Multi-Agent System with LLM reasoning + algorithmic memory

    Components:
    1. Analyzer Agent (LLM) - analyzes traces
    2. Verifier Agent (LLM) - verifies findings
    3. Memory Manager (Algorithm) - VBW + RFI-Δ storage
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.llm_config = LLMConfig(config_path)
        self.analyzer = AnalyzerAgent(self.llm_config)
        self.verifier = VerifierAgent(self.llm_config)
        self.memory = EPMManager()

        self.stats = {
            "traces_analyzed": 0,
            "hypotheses_verified": 0,
            "patterns_stored": 0,
            "total_cost": 0.0
        }

    def analyze_trace(self, trace: Dict[str, Any], source_run: str = "") -> Dict[str, Any]:
        """
        Analyze a failed trace with LLM reasoning

        Returns complete analysis with memory updates
        """
        self.stats["traces_analyzed"] += 1

        # Step 0: Retrieve similar patterns from memory for position guidance
        similar_patterns = []
        trace_length = get_trace_length(trace)
        if self.memory.entries:
            # Get top 3 most relevant patterns by RFI-Δ score
            sorted_patterns = sorted(
                [entry.to_dict() for entry in self.memory.entries.values()],
                key=lambda x: x['scores'].get('S', 0),
                reverse=True
            )[:3]
            similar_patterns = sorted_patterns

        # Step 1: Analyzer generates hypothesis (with position hints if available)
        print(f"  [Analyzer] Analyzing trace...")
        if similar_patterns:
            print(f"  [Analyzer] Using position guidance from {len(similar_patterns)} similar patterns")
        analysis = self.analyzer.analyze_trace(trace, similar_patterns)

        # Step 2: Verifier checks the hypothesis
        print(f"  [Verifier] Verifying hypothesis: {analysis['error_family']}...")
        evidence = self.verifier.verify_hypothesis(trace, analysis)
        print(f"  [Verifier] Result: verified={evidence.verified}, gain={evidence.ablation_gain:.2f}, hits={evidence.static_hits}")

        # Step 3: Build hypothesis structure with position information
        error_span = ErrorSpan(
            start_turn=analysis['error_step'][0],
            end_turn=analysis['error_step'][1],
            role_path=[analysis['tool_used']]
        )

        # Add position information for future guidance
        if trace_length > 0:
            error_span.add_position_info(trace_length)

        hypothesis = ErrorHypothesis(
            span=error_span,
            family=analysis['error_family'],
            rationale=analysis['error_reason'],
            signature=ErrorSignature(
                tool=analysis['tool_used'],
                api="default",
                arg_schema=[],
                context_slots=[],
                err_family=analysis['error_family']
            ),
            patch=self._create_patch_from_fix(evidence.probe_details.get('suggested_fix', {})),
            confidence=analysis['confidence'] * evidence.probe_details.get('verification_confidence', 0.5)
        )

        # Step 4: Memory manager decides whether to store (VBW gate)
        if evidence.verified:
            stored = self.memory.write_from_hypothesis(hypothesis, evidence, source_run)
            if stored:
                self.stats["hypotheses_verified"] += 1
                self.stats["patterns_stored"] += 1
                print(f"  [Memory] ✓ Pattern stored (VBW passed)")
            else:
                print(f"  [Memory] ✗ Rejected by VBW gate")

        # Return complete analysis
        return {
            "analysis": analysis,
            "evidence": evidence.to_dict(),
            "hypothesis": hypothesis.to_dict(),
            "memory_stored": evidence.verified and self.memory.write_from_hypothesis(hypothesis, evidence, source_run),
            "confidence": hypothesis.confidence
        }

    def _create_patch_from_fix(self, fix_data: Dict) -> Optional[ErrorPatch]:
        """Create ErrorPatch from LLM suggested fix"""
        if not fix_data:
            return None

        fix_type = fix_data.get('type', 'hint')
        fix_text = fix_data.get('fix_text', '')

        if fix_type == 'guard':
            return ErrorPatch(guard=fix_text)
        elif fix_type == 'rewrite':
            return ErrorPatch(rewrite=fix_text)
        else:
            return ErrorPatch(prompt_hint=fix_text)

    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics"""
        memory_stats = self.memory.get_statistics()

        return {
            "traces_analyzed": self.stats["traces_analyzed"],
            "hypotheses_verified": self.stats["hypotheses_verified"],
            "patterns_stored": self.stats["patterns_stored"],
            "verification_rate": self.stats["hypotheses_verified"] / max(self.stats["traces_analyzed"], 1),
            "memory": memory_stats
        }

    def save_memory(self, filepath: str):
        """Save memory to file"""
        self.memory.save(filepath)

    def load_memory(self, filepath: str):
        """Load memory from file"""
        self.memory = EPMManager.load(filepath)
