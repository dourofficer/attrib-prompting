"""
Simplified LLM-Based Multi-Agent System

Combines:
- LLM reasoning (Analyzer + Verifier agents)
- Algorithmic memory management (VBW + RFI-Δ)
"""

from .llm_agents import SimplifiedMAS, AnalyzerAgent, VerifierAgent, LLMConfig

__all__ = ['SimplifiedMAS', 'AnalyzerAgent', 'VerifierAgent', 'LLMConfig']
