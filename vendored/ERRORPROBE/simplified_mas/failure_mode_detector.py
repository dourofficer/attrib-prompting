"""
MAST Failure Mode Detector

Implements detection for MAST's 14 empirically-validated failure modes across 3 categories:
- FC1: Design & Specification Failures (41.77%)
- FC2: Coordination Failures (36.94%)
- FC3: Verification Failures (21.30%)

Based on: "Why Do Multi-Agent LLM Systems Fail?" (UC Berkeley)
"""

from typing import Dict, List, Any
import re
from collections import Counter


class FailureModeDetector:
    """
    Detects MAST's 14 failure modes in multi-agent traces

    Uses pattern matching and heuristics (not LLM) for efficiency
    """

    def __init__(self):
        # Common error indicators in content
        self.error_patterns = [
            r'error', r'exception', r'failed', r'failure', r'incorrect',
            r'wrong', r'bug', r'issue', r'problem', r'fix'
        ]

        # Completion indicators
        self.completion_patterns = [
            r'done', r'completed', r'finished', r'success', r'final',
            r'submit', r'deliver'
        ]

        # Question/clarification patterns
        self.question_patterns = [
            r'\?', r'what', r'why', r'how', r'which', r'clarify',
            r'confirm', r'verify', r'check'
        ]

    def detect_all(self, turn: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, float]:
        """
        Detect all failure modes for a given turn

        Args:
            turn: Current turn data {name, role, content, step, ...}
            context: {
                'previous_turns': List of earlier turns
                'agent_roles': Dict[agent_name] = role
                'task_goal': Original task description
                'turn_idx': Current turn index in full trace
            }

        Returns:
            Dict[mode_name] = confidence_score (0.0 to 1.0)
        """
        modes = {}

        # FC1: Design & Specification Failures
        modes['fm1_1_task_spec_failure'] = self._detect_task_spec_failure(turn, context)
        modes['fm1_2_role_spec_failure'] = self._detect_role_spec_failure(turn, context)
        modes['fm1_3_step_repetition'] = self._detect_step_repetition(turn, context)
        modes['fm1_4_context_loss'] = self._detect_context_loss(turn, context)
        modes['fm1_5_completion_failure'] = self._detect_completion_failure(turn, context)

        # FC2: Coordination Failures
        modes['fm2_1_conversation_reset'] = self._detect_conversation_reset(turn, context)
        modes['fm2_2_wrong_assumptions'] = self._detect_wrong_assumptions(turn, context)
        modes['fm2_3_task_derailment'] = self._detect_task_derailment(turn, context)
        modes['fm2_4_info_withholding'] = self._detect_info_withholding(turn, context)
        modes['fm2_6_reasoning_mismatch'] = self._detect_reasoning_mismatch(turn, context)

        # FC3: Verification Failures
        modes['fm3_1_premature_termination'] = self._detect_premature_termination(turn, context)
        modes['fm3_2_incomplete_verification'] = self._detect_incomplete_verification(turn, context)
        modes['fm3_3_incorrect_verification'] = self._detect_incorrect_verification(turn, context)

        # Return only modes with significant confidence (>= 0.3)
        return {k: v for k, v in modes.items() if v >= 0.3}

    # ========== FC1: Design & Specification Failures ==========

    def _detect_task_spec_failure(self, turn: Dict, context: Dict) -> float:
        """FM-1.1: Agent fails to follow task requirements (10.98%)"""
        content = turn.get('content', '').lower()
        task_goal = context.get('task_goal', '').lower()

        # Check if agent acknowledges task but does something different
        task_keywords = set(re.findall(r'\b\w+\b', task_goal))
        content_keywords = set(re.findall(r'\b\w+\b', content))

        # Low overlap with task keywords suggests misalignment
        if len(task_keywords) > 3:
            overlap = len(task_keywords & content_keywords) / len(task_keywords)
            if overlap < 0.2:
                return 0.7

        # Check for explicit task violations
        violation_patterns = [r'instead', r'different', r'changed', r'modified']
        if any(re.search(pattern, content) for pattern in violation_patterns):
            return 0.6

        return 0.0

    def _detect_role_spec_failure(self, turn: Dict, context: Dict) -> float:
        """FM-1.2: Agent fails to follow their role specification (0.5%)"""
        agent_name = turn.get('name', '')
        expected_role = context.get('agent_roles', {}).get(agent_name, '')
        content = turn.get('content', '').lower()

        # Define role-specific action patterns
        role_actions = {
            'engineer': ['implement', 'code', 'write', 'develop', 'program'],
            'reviewer': ['review', 'check', 'verify', 'test', 'validate'],
            'manager': ['assign', 'coordinate', 'plan', 'organize'],
            'qa': ['test', 'validate', 'verify', 'check'],
            'designer': ['design', 'architecture', 'structure', 'pattern']
        }

        # Check if agent's actions match their role
        if expected_role.lower() in role_actions:
            expected_actions = role_actions[expected_role.lower()]
            has_expected_action = any(action in content for action in expected_actions)

            # If role is clear but actions don't match, flag it
            if not has_expected_action and len(content) > 50:
                return 0.5

        return 0.0

    def _detect_step_repetition(self, turn: Dict, context: Dict) -> float:
        """FM-1.3: Agent repeats the same action (17.14% - most common!)"""
        content = turn.get('content', '')
        previous_turns = context.get('previous_turns', [])

        if len(previous_turns) < 2:
            return 0.0

        # Check similarity to recent turns
        similarities = []
        for prev_turn in previous_turns[-5:]:  # Check last 5 turns
            prev_content = prev_turn.get('content', '')
            similarity = self._content_similarity(content, prev_content)
            similarities.append(similarity)

        max_similarity = max(similarities) if similarities else 0.0

        # High similarity indicates repetition
        if max_similarity > 0.8:
            return 0.9
        elif max_similarity > 0.6:
            return 0.7
        elif max_similarity > 0.4:
            return 0.5

        return 0.0

    def _detect_context_loss(self, turn: Dict, context: Dict) -> float:
        """FM-1.4: Agent loses important context from earlier turns (3.33%)"""
        content = turn.get('content', '').lower()
        previous_turns = context.get('previous_turns', [])

        # Look for questions about information already provided
        question_indicators = ['what', 'which', 'where', 'when', 'who']
        has_question = any(indicator in content for indicator in question_indicators)

        if not has_question or len(previous_turns) < 5:
            return 0.0

        # Extract key information from earlier turns
        earlier_info = set()
        for turn in previous_turns[:-3]:  # Not recent turns
            words = re.findall(r'\b\w{4,}\b', turn.get('content', '').lower())
            earlier_info.update(words[:10])  # Key terms

        # Check if question is about information already mentioned
        question_terms = set(re.findall(r'\b\w{4,}\b', content))

        if earlier_info and question_terms:
            overlap = len(earlier_info & question_terms) / len(question_terms) if question_terms else 0
            if overlap > 0.5:
                return 0.6  # Likely asking about known information

        return 0.0

    def _detect_completion_failure(self, turn: Dict, context: Dict) -> float:
        """FM-1.5: Agent doesn't recognize when task is complete (9.82%)"""
        content = turn.get('content', '').lower()
        previous_turns = context.get('previous_turns', [])

        # Check if there's evidence of completion in recent turns
        completion_found = False
        for prev_turn in previous_turns[-3:]:
            prev_content = prev_turn.get('content', '').lower()
            if any(re.search(pattern, prev_content) for pattern in self.completion_patterns):
                completion_found = True
                break

        # If completed but agent continues working, flag it
        if completion_found:
            continuing_work = any(word in content for word in ['next', 'now', 'continue', 'also', 'additionally'])
            if continuing_work:
                return 0.6

        return 0.0

    # ========== FC2: Coordination Failures ==========

    def _detect_conversation_reset(self, turn: Dict, context: Dict) -> float:
        """FM-2.1: Conversation unexpectedly resets (2.33%)"""
        content = turn.get('content', '').lower()
        previous_turns = context.get('previous_turns', [])

        if len(previous_turns) < 3:
            return 0.0

        # Check for restart indicators
        restart_patterns = [r'let\'s start', r'begin', r'from scratch', r'restart', r'reset']
        if any(re.search(pattern, content) for pattern in restart_patterns):
            return 0.7

        # Check for sudden topic change (low similarity to recent context)
        recent_content = ' '.join([t.get('content', '') for t in previous_turns[-3:]])
        similarity = self._content_similarity(content, recent_content)

        if similarity < 0.1:
            return 0.5  # Very low similarity suggests reset

        return 0.0

    def _detect_wrong_assumptions(self, turn: Dict, context: Dict) -> float:
        """FM-2.2: Proceeding with wrong assumptions instead of clarifying (11.65%)"""
        content = turn.get('content', '').lower()

        # Look for assumption indicators without clarification
        assumption_patterns = [r'assume', r'probably', r'maybe', r'likely', r'think', r'believe']
        clarification_patterns = [r'confirm', r'clarify', r'verify', r'check with']

        has_assumption = any(re.search(pattern, content) for pattern in assumption_patterns)
        seeks_clarification = any(re.search(pattern, content) for pattern in clarification_patterns)

        # Assumption without seeking clarification is risky
        if has_assumption and not seeks_clarification:
            return 0.6

        return 0.0

    def _detect_task_derailment(self, turn: Dict, context: Dict) -> float:
        """FM-2.3: Conversation derails from original task (7.15%)"""
        content = turn.get('content', '').lower()
        task_goal = context.get('task_goal', '').lower()

        # Extract keywords from task
        task_keywords = set(re.findall(r'\b\w{5,}\b', task_goal))
        content_keywords = set(re.findall(r'\b\w{5,}\b', content))

        # Very low overlap suggests derailment
        if task_keywords and content_keywords:
            overlap = len(task_keywords & content_keywords) / len(task_keywords)
            if overlap < 0.1:
                return 0.7

        return 0.0

    def _detect_info_withholding(self, turn: Dict, context: Dict) -> float:
        """FM-2.4: Agent withholds crucial information (1.66%)"""
        content = turn.get('content', '').lower()

        # Look for partial information indicators
        partial_patterns = [r'some', r'partial', r'not all', r'incomplete', r'part of']
        warning_patterns = [r'note:', r'warning:', r'important:', r'however:']

        has_partial = any(re.search(pattern, content) for pattern in partial_patterns)
        has_warning = any(re.search(pattern, content) for pattern in warning_patterns)

        # Information mentioned but not fully shared
        if has_warning and has_partial:
            return 0.5

        return 0.0

    def _detect_reasoning_mismatch(self, turn: Dict, context: Dict) -> float:
        """FM-2.6: Reasoning says X but action does Y (13.98% - very common!)"""
        content = turn.get('content', '')

        # Try to separate reasoning from action
        # Common pattern: "I will do X because Y, [action that does Z]"
        reasoning_indicators = ['because', 'since', 'therefore', 'so', 'thus', 'reason']
        action_indicators = ['execute', 'run', 'implement', 'write', 'create']

        has_reasoning = any(indicator in content.lower() for indicator in reasoning_indicators)
        has_action = any(indicator in content.lower() for indicator in action_indicators)

        if not (has_reasoning and has_action):
            return 0.0

        # Split by common delimiters
        parts = re.split(r'[\n\n]+|\.\s+(?=[A-Z])', content)

        if len(parts) < 2:
            return 0.0

        # Check if first part (reasoning) differs from last part (action)
        reasoning_part = parts[0].lower()
        action_part = parts[-1].lower()

        similarity = self._content_similarity(reasoning_part, action_part)

        # Low similarity between reasoning and action suggests mismatch
        if similarity < 0.3:
            return 0.7

        return 0.0

    # ========== FC3: Verification Failures ==========

    def _detect_premature_termination(self, turn: Dict, context: Dict) -> float:
        """FM-3.1: Task terminates before completion (7.82%)"""
        content = turn.get('content', '').lower()
        task_goal = context.get('task_goal', '').lower()
        turn_idx = context.get('turn_idx', 0)

        # Check if this is a termination turn
        termination_patterns = [r'done', r'complete', r'finished', r'end', r'final']
        is_termination = any(re.search(pattern, content) for pattern in termination_patterns)

        if not is_termination:
            return 0.0

        # Very early termination (< 5 turns) is suspicious
        if turn_idx < 5:
            return 0.8

        # Check if key task requirements are mentioned in this turn
        task_keywords = set(re.findall(r'\b\w{5,}\b', task_goal))
        content_keywords = set(re.findall(r'\b\w{5,}\b', content))

        overlap = len(task_keywords & content_keywords) / len(task_keywords) if task_keywords else 1.0

        # If terminating but task keywords barely mentioned, likely premature
        if overlap < 0.3:
            return 0.7

        return 0.0

    def _detect_incomplete_verification(self, turn: Dict, context: Dict) -> float:
        """FM-3.2: No or incomplete verification (6.82%)"""
        content = turn.get('content', '').lower()
        agent_name = turn.get('name', '').lower()

        # Check if this is a verification/review turn
        is_verifier = any(role in agent_name for role in ['review', 'qa', 'test', 'verify'])

        if not is_verifier:
            return 0.0

        # Check depth of verification
        verification_depth_indicators = [
            'test', 'check', 'verify', 'validate', 'review', 'inspect',
            'compare', 'analyze', 'examine', 'evaluate'
        ]

        depth_count = sum(1 for indicator in verification_depth_indicators if indicator in content)

        # Low verification depth
        if depth_count < 2:
            return 0.6
        elif depth_count < 4:
            return 0.3

        return 0.0

    def _detect_incorrect_verification(self, turn: Dict, context: Dict) -> float:
        """FM-3.3: Verification itself is incorrect (6.66%)"""
        content = turn.get('content', '').lower()
        previous_turns = context.get('previous_turns', [])

        # Check if this is approving/accepting something
        approval_patterns = [r'approved', r'correct', r'looks good', r'passed', r'accepted']
        is_approval = any(re.search(pattern, content) for pattern in approval_patterns)

        if not is_approval:
            return 0.0

        # Check if there are error indicators in recent context
        has_errors_in_context = False
        for prev_turn in previous_turns[-3:]:
            prev_content = prev_turn.get('content', '').lower()
            if any(re.search(pattern, prev_content) for pattern in self.error_patterns):
                has_errors_in_context = True
                break

        # Approving despite errors suggests incorrect verification
        if has_errors_in_context:
            return 0.7

        return 0.0

    # ========== Helper Methods ==========

    def _content_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate simple content similarity using word overlap

        Returns: 0.0 to 1.0 (Jaccard similarity)
        """
        if not text1 or not text2:
            return 0.0

        # Extract words (4+ chars)
        words1 = set(re.findall(r'\b\w{4,}\b', text1.lower()))
        words2 = set(re.findall(r'\b\w{4,}\b', text2.lower()))

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def get_mode_description(self, mode_name: str) -> str:
        """Get human-readable description for a failure mode"""
        descriptions = {
            'fm1_1_task_spec_failure': 'FC1.1: Fails to follow task requirements (10.98%)',
            'fm1_2_role_spec_failure': 'FC1.2: Fails to follow agent role (0.5%)',
            'fm1_3_step_repetition': 'FC1.3: Repeats same actions in loop (17.14%)',
            'fm1_4_context_loss': 'FC1.4: Loses critical context (3.33%)',
            'fm1_5_completion_failure': 'FC1.5: Doesn\'t recognize completion (9.82%)',
            'fm2_1_conversation_reset': 'FC2.1: Conversation resets unexpectedly (2.33%)',
            'fm2_2_wrong_assumptions': 'FC2.2: Wrong assumptions without clarification (11.65%)',
            'fm2_3_task_derailment': 'FC2.3: Task derails from goal (7.15%)',
            'fm2_4_info_withholding': 'FC2.4: Withholds crucial information (1.66%)',
            'fm2_6_reasoning_mismatch': 'FC2.6: Reasoning-action mismatch (13.98%)',
            'fm3_1_premature_termination': 'FC3.1: Terminates prematurely (7.82%)',
            'fm3_2_incomplete_verification': 'FC3.2: Incomplete verification (6.82%)',
            'fm3_3_incorrect_verification': 'FC3.3: Incorrect verification (6.66%)',
        }
        return descriptions.get(mode_name, mode_name)
