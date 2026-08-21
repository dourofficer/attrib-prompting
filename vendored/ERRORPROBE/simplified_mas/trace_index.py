"""
TraceIndex: Indexed trace storage for fast backward traversal

Stores FULL trace (no truncation!) with fast lookup capabilities.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import json


@dataclass
class Turn:
    """Single turn in the trace"""
    index: int                          # Turn number (0-based)
    role: str                          # Agent role
    agent: str                         # Agent name
    content: str                       # Full content (no truncation!)
    action: Optional[str] = None       # Action type if any
    tool_calls: List[str] = field(default_factory=list)  # Tools used
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata

    def summary(self, max_length: int = 200) -> str:
        """Short summary for indexing"""
        content_preview = self.content[:max_length] if len(self.content) > max_length else self.content
        return f"[{self.index}] {self.agent}: {content_preview}..."

    def has_keyword(self, keyword: str) -> bool:
        """Check if turn contains keyword (case-insensitive)"""
        return keyword.lower() in self.content.lower()


class TraceIndex:
    """
    Indexed trace storage for fast backward traversal

    Stores EVERYTHING but allows selective loading for analysis.
    Provides fast search and lookup by agent, action type, keywords, etc.
    """

    def __init__(self, trace: Dict[str, Any]):
        self.question_id = trace.get('question_ID', 'unknown')
        self.question = trace.get('question', '')
        self.ground_truth_code = trace.get('ground_truth', '')

        # Ground truth error information (if available)
        self.gt_agent = trace.get('mistake_agent', 'unknown')
        self.gt_step = trace.get('mistake_step', -1)
        self.gt_reason = trace.get('mistake_reason', '')

        # Convert string step to int if needed
        if isinstance(self.gt_step, str):
            try:
                self.gt_step = int(self.gt_step)
            except ValueError:
                self.gt_step = -1

        # Parse all turns into Turn objects
        self.turns: List[Turn] = []
        for idx, turn_data in enumerate(trace.get('history', [])):
            turn = Turn(
                index=idx,
                role=turn_data.get('role', 'unknown'),
                agent=turn_data.get('name', 'unknown'),
                content=turn_data.get('content', ''),
                action=self._extract_action(turn_data),
                tool_calls=self._extract_tool_calls(turn_data),
                metadata=turn_data
            )
            self.turns.append(turn)

        # Build indices for fast lookup
        self._build_indices()

    def _extract_action(self, turn_data: Dict) -> Optional[str]:
        """Extract action type from turn"""
        content = turn_data.get('content', '')

        # Look for action patterns like "command_name": "Editor.create_file"
        if '"command_name"' in content or "'command_name'" in content:
            # Classify by tool type
            if 'Editor.' in content or 'editor.' in content:
                return 'editor'
            elif 'Plan.' in content or 'plan.' in content:
                return 'planning'
            elif 'TeamLeader.' in content or 'teamleader.' in content:
                return 'coordination'
            elif 'RoleZero.' in content:
                return 'communication'
            else:
                return 'action'

        # Check for thinking/reasoning turns
        if 'thinking:' in content.lower() or '# past experience' in content.lower():
            return 'thinking'

        return None

    def _extract_tool_calls(self, turn_data: Dict) -> List[str]:
        """Extract tool calls from turn"""
        content = turn_data.get('content', '')
        tools = []

        # Look for common tool patterns
        tool_patterns = [
            'Editor.', 'Plan.', 'TeamLeader.', 'RoleZero.',
            'ProductManager.', 'Architect.', 'Engineer.'
        ]

        for pattern in tool_patterns:
            if pattern in content:
                tools.append(pattern.replace('.', ''))

        return tools

    def _build_indices(self):
        """Build lookup indices for fast access"""
        # Index by agent
        self.by_agent: Dict[str, List[int]] = {}
        for turn in self.turns:
            if turn.agent not in self.by_agent:
                self.by_agent[turn.agent] = []
            self.by_agent[turn.agent].append(turn.index)

        # Index by action type
        self.by_action: Dict[str, List[int]] = {}
        for turn in self.turns:
            if turn.action:
                if turn.action not in self.by_action:
                    self.by_action[turn.action] = []
                self.by_action[turn.action].append(turn.index)

        # Build keyword index for common error indicators
        self.keyword_index: Dict[str, List[int]] = {}
        error_keywords = ['error', 'failed', 'exception', 'bug', 'incorrect', 'wrong']

        for keyword in error_keywords:
            matching_turns = []
            for turn in self.turns:
                if turn.has_keyword(keyword):
                    matching_turns.append(turn.index)
            if matching_turns:
                self.keyword_index[keyword] = matching_turns

    # ===== Query Methods =====

    def search(self, keyword: str) -> List[Turn]:
        """Search for turns containing keyword"""
        results = []
        for turn in self.turns:
            if turn.has_keyword(keyword):
                results.append(turn)
        return results

    def get_agent_turns(self, agent: str) -> List[Turn]:
        """Get all turns by specific agent"""
        indices = self.by_agent.get(agent, [])
        return [self.turns[i] for i in indices]

    def get_action_turns(self, action: str) -> List[Turn]:
        """Get all turns with specific action type"""
        indices = self.by_action.get(action, [])
        return [self.turns[i] for i in indices]

    def get_window(self, center: int, radius: int = 5) -> List[Turn]:
        """Get turns around a specific index"""
        start = max(0, center - radius)
        end = min(len(self.turns), center + radius + 1)
        return self.turns[start:end]

    def get_range(self, start: int, end: int) -> List[Turn]:
        """Get turns in a specific range"""
        start = max(0, start)
        end = min(len(self.turns), end)
        return self.turns[start:end]

    def get_last_n(self, n: int) -> List[Turn]:
        """Get last N turns"""
        return self.turns[-n:] if n > 0 else []

    def get_turn(self, index: int) -> Optional[Turn]:
        """Get specific turn by index"""
        if 0 <= index < len(self.turns):
            return self.turns[index]
        return None

    # ===== Properties =====

    @property
    def length(self) -> int:
        """Total number of turns"""
        return len(self.turns)

    @property
    def final_turn(self) -> Optional[Turn]:
        """Get the last turn"""
        return self.turns[-1] if self.turns else None

    @property
    def agents(self) -> List[str]:
        """Get list of all agents in trace"""
        return list(self.by_agent.keys())

    # ===== Summary Methods =====

    def summary(self) -> str:
        """High-level trace summary"""
        return f"""
Trace: {self.question_id}
Total turns: {self.length}
Agents: {', '.join(self.agents)}
Final outcome: {self.final_turn.summary() if self.final_turn else 'No turns'}
Ground truth: {self.gt_agent} at step {self.gt_step}
        """.strip()

    def format_turns(self, turns: List[Turn], include_full_content: bool = False) -> str:
        """Format turns for display"""
        lines = []
        for turn in turns:
            if include_full_content:
                lines.append(f"Turn {turn.index} [{turn.agent}]:\n{turn.content}\n")
            else:
                lines.append(f"Turn {turn.index} [{turn.agent}]: {turn.content[:300]}...")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'question_id': self.question_id,
            'length': self.length,
            'agents': self.agents,
            'ground_truth': {
                'agent': self.gt_agent,
                'step': self.gt_step,
                'reason': self.gt_reason
            },
            'turns': [
                {
                    'index': t.index,
                    'agent': t.agent,
                    'role': t.role,
                    'action': t.action,
                    'content_length': len(t.content)
                }
                for t in self.turns
            ]
        }


def get_trace_length(trace: Dict[str, Any]) -> int:
    """Helper function to get trace length from raw trace dict"""
    return len(trace.get('history', []))
