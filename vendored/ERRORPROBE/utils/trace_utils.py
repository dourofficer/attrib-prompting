#!/usr/bin/env python3
"""
Utility functions for trace analysis and processing
"""

from typing import Dict, List, Set, Any, Tuple, Optional


def extract_agent_names(trace: Dict[str, Any]) -> List[str]:
    """
    Extract all unique agent names from a trace.

    This enables classification-style agent identification rather than
    open-ended generation, significantly improving accuracy.

    Args:
        trace: The execution trace containing messages

    Returns:
        Sorted list of unique agent names found in the trace
    """
    agent_names = set()

    # Extract from messages
    messages = trace.get('messages', [])
    for msg in messages:
        # Check multiple fields for agent names
        for field in ['role', 'agent', 'name', 'sender']:
            if field in msg and msg[field]:
                agent_names.add(msg[field])

    # Also check history if available
    history = trace.get('history', [])
    for msg in history:
        # Check multiple fields for agent names
        for field in ['role', 'agent', 'name', 'sender']:
            if field in msg and msg[field]:
                agent_names.add(msg[field])

    # Remove generic/system names
    generic_names = {'system', 'user', 'assistant', 'function', 'tool', ''}
    agent_names = agent_names - generic_names

    # Return sorted for consistency
    return sorted(list(agent_names))


def calculate_trace_position(turn_idx: int, total_turns: int) -> Dict[str, Any]:
    """
    Calculate position metadata for a turn in the trace.

    Args:
        turn_idx: The turn index (0-based)
        total_turns: Total number of turns in the trace

    Returns:
        Dictionary with position information:
        - position_pct: Position as percentage (0.0 to 1.0)
        - position_label: "early", "middle", or "late"
        - turn_range: (start, end) as percentages
    """
    if total_turns <= 0:
        return {
            "position_pct": 0.0,
            "position_label": "unknown",
            "turn_range": (0.0, 0.0)
        }

    position_pct = turn_idx / total_turns

    # Determine label
    if position_pct < 0.33:
        label = "early"
    elif position_pct < 0.67:
        label = "middle"
    else:
        label = "late"

    # Calculate turn range (with 10% window)
    window = 0.1
    range_start = max(0.0, position_pct - window)
    range_end = min(1.0, position_pct + window)

    return {
        "position_pct": round(position_pct, 3),
        "position_label": label,
        "turn_range": (round(range_start, 3), round(range_end, 3))
    }


def calculate_span_statistics(span_start: int, span_end: int, total_turns: int) -> Dict[str, Any]:
    """
    Calculate statistics for an error span.

    Args:
        span_start: Start turn of error span
        span_end: End turn of error span
        total_turns: Total turns in trace

    Returns:
        Dictionary with span statistics:
        - start_pct: Start position as percentage
        - end_pct: End position as percentage
        - mid_pct: Midpoint position as percentage
        - duration_pct: Span duration as percentage
        - position_label: "early", "middle", or "late"
    """
    if total_turns <= 0:
        return {
            "start_pct": 0.0,
            "end_pct": 0.0,
            "mid_pct": 0.0,
            "duration_pct": 0.0,
            "position_label": "unknown"
        }

    start_pct = span_start / total_turns
    end_pct = span_end / total_turns
    mid_pct = (start_pct + end_pct) / 2
    duration_pct = (span_end - span_start) / total_turns

    # Determine label based on midpoint
    if mid_pct < 0.33:
        label = "early"
    elif mid_pct < 0.67:
        label = "middle"
    else:
        label = "late"

    return {
        "start_pct": round(start_pct, 3),
        "end_pct": round(end_pct, 3),
        "mid_pct": round(mid_pct, 3),
        "duration_pct": round(duration_pct, 3),
        "position_label": label
    }


def format_agent_options(agent_names: List[str]) -> str:
    """
    Format agent names as a numbered list for LLM classification.

    Args:
        agent_names: List of agent names

    Returns:
        Formatted string for LLM prompt
    """
    if not agent_names:
        return "No agents identified in trace"

    lines = ["Available agents in this trace:"]
    for i, name in enumerate(agent_names, 1):
        lines.append(f"  {i}. {name}")

    return "\n".join(lines)


def format_position_hint(position_stats: Dict[str, Any],
                         similar_patterns: Optional[List[Dict]] = None) -> str:
    """
    Format position information as a hint for the LLM.

    Args:
        position_stats: Position statistics from calculate_span_statistics
        similar_patterns: Optional list of similar patterns with position info

    Returns:
        Formatted string for LLM prompt
    """
    lines = []

    if position_stats:
        label = position_stats.get('position_label', 'unknown')
        mid_pct = position_stats.get('mid_pct', 0.0)
        lines.append(f"Typical error position: {label} in trace ({mid_pct:.0%})")

    if similar_patterns:
        # Calculate average position from similar patterns
        positions = []
        for pattern in similar_patterns:
            if 'position_info' in pattern:
                positions.append(pattern['position_info'].get('mid_pct', 0.5))

        if positions:
            avg_pos = sum(positions) / len(positions)
            if avg_pos < 0.33:
                hint = "early"
            elif avg_pos < 0.67:
                hint = "middle"
            else:
                hint = "late"

            lines.append(f"Similar errors typically occur: {hint} in trace ({avg_pos:.0%})")
            lines.append(f"  Based on {len(positions)} similar pattern(s)")

    return "\n".join(lines) if lines else "No position guidance available"


def get_trace_length(trace: Dict[str, Any]) -> int:
    """
    Get the total number of turns/messages in a trace.

    Args:
        trace: The execution trace

    Returns:
        Number of turns in the trace
    """
    messages = trace.get('messages', [])
    history = trace.get('history', [])
    # Return the maximum of messages and history length
    # (some traces use 'messages', others use 'history')
    return max(len(messages), len(history))


def extract_tool_from_turn(turn: Dict[str, Any]) -> str:
    """
    Extract the tool/agent name from a specific turn.

    Args:
        turn: A single turn/message from the trace

    Returns:
        Tool/agent name, or "unknown" if not found
    """
    # Try different fields
    for field in ['role', 'agent', 'name', 'sender', 'tool']:
        if field in turn and turn[field]:
            return turn[field]

    return "unknown"
