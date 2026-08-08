"""CORRECT's schema-guided all-at-once detection — batched over trajectories via VLLM.

Prompt strings, schema injection, response trimming and parsing are copied
**verbatim** from the vendored vLLM code-path
(``baselines/CORRECT/src/Lib/local_model.py``:
``analyze_all_at_once_vllm`` / ``_run_vllm_generation_with_schemata``), which is
the path both vendored inference scripts (Who&When and CORRECT-Error) route open
models through. Notably, in that path the *active* prompt excludes the gold
answer and does not number the conversation steps (the ground-truth / numbered
variants are commented out in the vendored source).

Deliberate deviations (see README):
  * the agent-identity field prefers ``history[t]["role"]`` — this is the
    vendored ``_agent_label(entry, "role")`` behaviour, which is what every
    dataset in this repo stores;
  * outputs are passed through :func:`~baselines.correct.engine.strip_think`
    BEFORE the vendored "Agent Name:" trim, so reasoning backbones are handled;
  * parsing applies prompting's markdown strip (a no-op on vendored-style
    outputs) before the verbatim evaluate.py regexes.
"""
from __future__ import annotations

import re

from .engine import strip_think

SYSTEM_PROMPT = "You are a helpful assistant skilled in analyzing conversations."

# Parsing regexes — identical to the vendored evaluate.py.
AGENT_RE = re.compile(r"Agent Name:\s*([\w_]+)", re.IGNORECASE)
STEP_RE = re.compile(r"Step Number:\s*(\d+)", re.IGNORECASE)

# Markdown decoration to drop before applying the vendored regexes (same fix as
# the prompting baseline): reasoning models bold the labels — `**Agent Name:**` —
# which defeats `Agent Name:\s*([\w_]+)`.
_MARKDOWN_RE = re.compile(r"[*`#]")


def _agent_label(entry: dict, preferred_field: str = "role") -> str:
    """Verbatim vendored ``_agent_label`` — prefer one field, fall back to the other."""
    fallback_field = "name" if preferred_field == "role" else "role"
    return entry.get(preferred_field) or entry.get(fallback_field) or "Unknown Agent"


def messages(user_prompt: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_all_at_once_prompt(history: list[dict], problem: str) -> str:
    """The vendored vLLM all-at-once prompt (no gold answer, unnumbered dump)."""
    chat_content = "\n".join(
        f"{_agent_label(entry, 'role')}: {entry.get('content', '')}" for entry in history
    )
    return (
        "You are an AI assistant tasked with analyzing a multi-agent conversation history when solving a real world problem. "
        f"The problem is:  {problem} \n"
        "Identify which agent made an error, at which step, and explain the reason for the error. "
        "Here's the conversation:\n\n" + chat_content +
        "\n\nBased on this conversation, please predict the following:\n"
        "1. The name of the agent who made a mistake that should be directly responsible for the wrong solution to the real world problem. If there are no agents that make obvious mistakes, decide one single agent in your mind. Directly output the name of the Expert.\n"
        "2. In which step the mistake agent first made mistake. For example, in a conversation structured as follows: "
        '{\n"agent a": "xx",\n"agent b": "xxxx",\n"agent c": "xxxxx",\n"agent a": "xxxxxxx"\n},\n'
        "each entry represents a 'step' where an agent provides input. The 'x' symbolizes the speech of each agent. If the mistake is in agent c's speech, the step number is 2. If the second speech by 'agent a' contains the mistake, the step number is 3, and so on. Please determine the step number where the first mistake occurred.\n"
        "3. The reason for your prediction."
        "Please answer in the format: Agent Name: (Your prediction)\n, Step Number: (Your prediction)\n, Reason for Mistake: (Your reason)\n."
    )


def inject_schemata(prompt: str, schema_content: str | list[str]) -> str:
    """Append retrieved error schema(ta) — verbatim vendored wording and layout.

    ``schema_content`` may be a single schema string or a list of schemata,
    exactly as in ``_run_vllm_generation_with_schemata`` (both inference scripts
    pass lists; the string branch is kept for fidelity).
    """
    if isinstance(schema_content, list):
        # Multiple schemata - format them nicely
        if len(schema_content) == 1:
            combined_schema = schema_content[0]
        else:
            schema_parts = []
            for i, content in enumerate(schema_content):
                schema_parts.append(f"Schema {i+1}:\n{content}")
            combined_schema = "\n\n".join(schema_parts)

        schema_text = f"Here are error schemata to help guide your analysis:\n\n{combined_schema}"
    else:
        # Single schema (string)
        schema_text = f"Here's a error schema to help guide your analysis:\n\n{schema_content}"

    return f"{prompt}\n\n{schema_text}.\n\n\nPlease remember the error schema{'s are' if isinstance(schema_content, list) and len(schema_content) > 1 else ' is'} just to guide your analysis. You can neglect it if you find the schema is not helpful.\nPlease remember to answer in the following format: Agent Name: (Your prediction)\n, Step Number: (Your prediction)\n, Reason for Mistake: (Your reason)\n"


def trim_response(response: str) -> str:
    """Verbatim vendored response clean-up: keep the first "Agent Name:" block."""
    if "Agent Name:" in response:
        parts = response.split("Agent Name:")
        response = "Agent Name:" + parts[1].split("\n\n")[0]
    return response.strip()


def parse_prediction(raw: str) -> tuple[str | None, int | None, str]:
    """Extract (predicted_agent, predicted_step, trimmed) from a generation.

    Order matters: strip_think FIRST (reasoning blocks may mention
    "Agent Name:", which would corrupt the vendored trim), then the verbatim
    trim, then the markdown-tolerant verbatim regexes. Pure function of ``raw``
    so stored predictions can be re-parsed.
    """
    trimmed = trim_response(strip_think(raw))
    answer = _MARKDOWN_RE.sub("", trimmed)
    agent_m = AGENT_RE.search(answer)
    step_m = STEP_RE.search(answer)
    return (agent_m.group(1) if agent_m else None,
            int(step_m.group(1)) if step_m else None,
            trimmed)
