"""CORRECT's schema-guided all-at-once detection — the unified cloud-paper variant.

Prompt strings, schema injection, unicode scrubbing and retrieval semantics are
copied **verbatim** from the vendored closed-source code-path
(``vendored/CORRECT/src/Lib/cloud_paper.py``), the module the CORRECT authors
keep byte-identical to their paper runs and the only path the paper ever used
with closed-source detectors. Per user decision this variant is used for ALL
datasets and backends (the vendored local-vLLM variant with its short schema
injection is not adapted).

Two methods:

- ``correct`` — base prompt (``cloud_paper.py:364-375``) + top-k retrieved
  neighbour schemata injected as "THOUGHT TEMPLATE(S) FOR GUIDANCE"
  (``_correct_modify_prompt_paper``, ``:252-344``), then the aggressive ASCII
  scrub ``clean_text`` on user AND system prompt (``:380``, ``:388``).
- ``correct_baseline`` — the k=0 baseline prompt (``cloud_paper.py:173-192``),
  which differs from the base prompt in three byte-level ways (no space before
  ``\\n`` after the problem, a triple-quoted indented JSON example, tail
  ``"Reason for Mistake: \\n"``) and is NOT ``clean_text``-scrubbed — only the
  smart-quote mapping ``_clean_unicode_content`` that ``_make_api_call_cloud``
  applies (``:87-90``).

Both run in the repo's two GT settings (GUIDE.md "GT settings"): the programs
take ``include_gt``, and ``include_gt=True`` inserts the vendored ground-truth
line (``local_model.py:637-638``) after the problem line. **The vendored cloud
path never includes the answer**, so for this baseline the parity-tested
vendored bytes are the *without*-GT setting (``include_gt=False`` default —
the inverse of prompting, where the vendored prompt carries the answer).

Deliberate deviations (see README):
  * the agent-identity field is ``history[t]["role"]`` for every dataset (the
    vendored ``index_agent`` resolves to ``"role"`` for hand-crafted-shaped
    data, which is what every dataset in this repo stores; we do NOT replicate
    the ``is_handcrafted="False"`` truthiness bug that labeled the original
    algorithm-generated turns ``user``/``assistant``);
  * scrubbing is applied at message-build time — our backends send messages
    verbatim, and ``_clean_unicode_content`` after ``clean_text`` is a no-op;
  * parsing reuses prompting's ``parse_all_at_once`` (strip_think + markdown
    tolerance over the verbatim evaluate.py regexes — the vendored
    ``CORRECT/src/evaluate.py`` patterns are the identical family).
"""
from __future__ import annotations

from typing import Generator

from baselines.prompting.methods import agent_vocabulary, parse_all_at_once, strip_think  # noqa: F401 (strip_think re-exported for schemagen)

# The agent identity lives in the "role" field for every dataset in this repo.
AGENT_KEY = "role"

SYSTEM_PROMPT = "You are a helpful assistant skilled in analyzing conversations."


# ─────────────────────────────────────────────────────────────────────────────
# Unicode scrubbing (verbatim from cloud_paper.py)
# ─────────────────────────────────────────────────────────────────────────────

def _clean_unicode_content(text):
    """Verbatim ``cloud_paper.py::_clean_unicode_content`` — smart quotes/dashes
    only; does NOT strip arbitrary non-ASCII characters."""
    if not isinstance(text, str):
        return text
    replacements = {
        '“': '"', '”': '"',
        '‘': "'", '’': "'",
        '–': '-', '—': '--',
        '…': '...',
    }
    for unicode_char, replacement in replacements.items():
        text = text.replace(unicode_char, replacement)
    try:
        text = text.encode('utf-8', errors='replace').decode('utf-8')
    except UnicodeEncodeError:
        text = text.encode('ascii', errors='replace').decode('ascii')
    return text


def clean_text(text):
    """Verbatim ``cloud_paper.py::clean_text`` — replaces every non-ASCII
    character with a space (including the ``•`` bullets inside the
    schema-injection block)."""
    if text is None:
        return ""
    replacements = {
        '“': '"', '”': '"',
        '‘': "'", '’': "'",
        '–': '-', '—': '--',
        '…': '...', ' ': ' ',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    cleaned = []
    for char in text:
        if ord(char) < 128:
            cleaned.append(char)
        elif char.isspace():
            cleaned.append(' ')
        else:
            cleaned.append(' ')
    return ''.join(cleaned)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builders (verbatim from cloud_paper.py; optional GT line)
# ─────────────────────────────────────────────────────────────────────────────

_GT_LINE = "The Answer for the problem is: {ground_truth}\n"  # local_model.py:637-638


def _serialize_history(history: list[dict]) -> str:
    # cloud_paper.py:360-363 / :168-171 with index_agent="role".
    return "\n".join([
        f"{entry.get(AGENT_KEY, 'Unknown Agent')}: {entry.get('content', '')}"
        for entry in history
    ])


def build_correct_base_prompt(history: list[dict], problem: str,
                              ground_truth: str | None = None) -> str:
    """The schema-guided base prompt — verbatim ``cloud_paper.py:364-375``."""
    chat_content = _serialize_history(history)
    return (
        "You are an AI assistant tasked with analyzing a multi-agent conversation history when solving a real world problem. "
        f"The problem is:  {problem} \n"
        + (_GT_LINE.format(ground_truth=ground_truth) if ground_truth is not None else "") +
        "Identify which agent made an error, at which step, and explain the reason for the error. "
        "Here's the conversation:\n\n" + chat_content +
        "\n\nBased on this conversation, please predict the following:\n"
        "1. The name of the agent who made a mistake that should be directly responsible for the wrong solution to the real world problem. If there are no agents that make obvious mistakes, decide one single agent in your mind. Directly output the name of the Expert.\n"
        "2. In which step the mistake agent first made mistake. For example, in a conversation structured as follows: "
        '{\n"agent a": "xx",\n"agent b": "xxxx",\n"agent c": "xxxxx",\n"agent a": "xxxxxxx"\n},\n'
        "each entry represents a 'step' where an agent provides input. The 'x' symbolizes the speech of each agent. If the mistake is in agent c's speech, the step number is 2. If the second speech by 'agent a' contains the mistake, the step number is 3, and so on. Please determine the step number where the first mistake occurred.\n"
        "3. The reason for your prediction."
        "Please answer in the format: Agent Name: (Your prediction)\n Step Number: (Your prediction)\n Reason for Mistake: (Your reason)\n"
    )


def build_baseline_prompt(history: list[dict], problem: str,
                          ground_truth: str | None = None) -> str:
    """The k=0 baseline prompt — verbatim ``cloud_paper.py:173-192``."""
    chat_content = _serialize_history(history)
    return (
        "You are an AI assistant tasked with analyzing a multi-agent conversation history when solving a real world problem. "
        f"The problem is:  {problem}\n"
        + (_GT_LINE.format(ground_truth=ground_truth) if ground_truth is not None else "") +
        "Identify which agent made an error, at which step, and explain the reason for the error. "
        "Here's the conversation:\n\n" + chat_content +
        "\n\nBased on this conversation, please predict the following:\n"
        "1. The name of the agent who made a mistake that should be directly responsible for the wrong solution to the real world problem. If there are no agents that make obvious mistakes, decide one single agent in your mind. Directly output the name of the Expert.\n"
        "2. In which step the mistake agent first made mistake. For example, in a conversation structured as follows: "
        """
        {
            "agent a": "xx",
            "agent b": "xxxx",
            "agent c": "xxxxx",
            "agent a": "xxxxxxx"
        },
        """
        "each entry represents a 'step' where an agent provides input. The 'x' symbolizes the speech of each agent. If the mistake is in agent c's speech, the step number is 2. If the second speech by 'agent a' contains the mistake, the step number is 3, and so on. Please determine the step number where the first mistake occurred.\n"
        "3. The reason for your prediction."
        "Please answer in the format: Agent Name: (Your prediction)\n Step Number: (Your prediction)\n Reason for Mistake: \n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Schema injection (verbatim from cloud_paper.py:252-344; wording='template'
# is the paper-byte-identical default and the only wording this repo uses)
# ─────────────────────────────────────────────────────────────────────────────

def modify_prompt_paper(prompt, schema_keys, schema_contents, wording='template'):
    is_schema = wording == 'schema'
    NOUN_S = "schema" if is_schema else "template"  # singular noun
    NOUN_P = "schemata" if is_schema else "templates"  # plural noun
    HDR_S = "ERROR SCHEMA" if is_schema else "THOUGHT TEMPLATE"  # single header
    HDR_P = "ERROR SCHEMATA" if is_schema else "THOUGHT TEMPLATES"  # multi header

    if not schema_contents:
        return prompt + (
            "\n\n"
            f"Since no reference {NOUN_S} is available, analyze the conversation step by step:\n"
            "1. Read the entire conversation first to understand the flow\n"
            "2. Go through each step (Step 0, Step 1, Step 2, etc.) and evaluate:\n"
            "   • Is the information accurate?\n"
            "   • Is the reasoning sound?\n"
            "   • Does it advance toward the correct answer?\n"
            "3. Identify where the first error occurs\n\n"
            "Format your response as:\n"
            "Agent Name: [your prediction]\n"
            "Step Number: [step where error occurred, counting from Step 0]\n"
            "Reason for Mistake: [your explanation]\n"
        )
    parts = []
    if len(schema_contents) == 1:
        key = schema_keys[0]
        content = schema_contents[0]
        parts.append(
            f"\n\n==== {HDR_S} FOR GUIDANCE ====\n"
            f"Here is how a similar error was identified in Case #{key}:\n\n"
            f"{content}\n"
        )
        parts.append(
            "HOW TO USE THIS REFERENCE EXAMPLE:\n"
            f"This {NOUN_S} demonstrates one type of error pattern for reference. To apply it to your analysis:\n\n"
            "1. Study the ERROR PATTERN shown: What type of mistake does this example identify?\n"
            "2. Use this as reference to analyze YOUR conversation:\n"
            "   • Read through your conversation systematically (Step 0, Step 1, Step 2...)\n"
            "   • At each step, ask: 'Is there an error here, and does it match this pattern or a different one?'\n"
            "   • The error in your case may follow the same pattern or be completely different\n"
            "3. Remember this is just a reference example:\n"
            "   • Your error may occur at any step number\n"
            "   • Your error may be a different type entirely\n"
            f"   • Use this {NOUN_S} to help you recognize what errors look like, not to assume your error matches\n"
        )
    else:
        parts.append(
            f"\n\n==== {HDR_P} FOR GUIDANCE ====\n"
            f"Here are {len(schema_contents)} examples of how similar errors were identified:\n"
            f"When applying these {NOUN_P}:\n"
            "• Look for common error patterns across these examples\n"
            "• Each example shows different step numbers - focus on the ERROR TYPE, not the step position\n"
            "• Systematically check each step in your conversation (starting from Step 0)\n"
        )
        for i, (key, content) in enumerate(zip(schema_keys, schema_contents), 1):
            parts.append(
                f"\n--- Example {i} (Case #{key}) ---\n"
                f"{content}\n"
                f"--- End Example {i} ---\n"
            )
        parts.append(
            "HOW TO USE THESE REFERENCE EXAMPLES:\n"
            f"These {len(schema_contents)} examples show different error patterns for reference. For your analysis:\n\n"
            "1. Study the various error patterns demonstrated above\n"
            "2. Read through your conversation step by step (Step 0, Step 1, Step 2...)\n"
            "3. At each step, check for errors - they may match one of these patterns or be different types\n"
            "4. When you identify an error, determine if it follows a similar pattern or is a new type\n\n"
            "Important: These are reference examples only. Your conversation may contain:\n"
            "• The same type of error as shown in the examples\n"
            "• A completely different type of error not shown here\n"
            "• An error at any step number, regardless of the examples\n"
        )
    section = '\n'.join(parts)
    return (
        f"{prompt}"
        f"{section}\n\n"
        "Now analyze your conversation using these reference examples as guidance:\n"
        "1. Examine your conversation step by step (starting from Step 0)\n"
        "2. Look for errors at each step - they may match the example patterns or be different types\n"
        "3. Identify where an error occurs and what type it is\n\n"
        "Format your response as:\n"
        "Agent Name: [agent who made the error]\n"
        "Step Number: [step where error occurred, counting from Step 0]\n"
        "Reason for Mistake: [explain the error - may match example patterns or be a different type]\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Message assembly (the scrub is where the two methods diverge)
# ─────────────────────────────────────────────────────────────────────────────

def correct_messages(record: dict, schema_keys: list[int], schema_contents: list[str],
                     include_gt: bool = False) -> list[dict]:
    """Schema-guided messages: full ``clean_text`` scrub on user AND system
    prompt (``cloud_paper.py:380``, ``:388``)."""
    prompt = modify_prompt_paper(
        build_correct_base_prompt(record["history"], record["question"],
                                  record["ground_truth"] if include_gt else None),
        schema_keys, schema_contents,
    )
    return [
        {"role": "system", "content": clean_text(SYSTEM_PROMPT)},
        {"role": "user", "content": clean_text(prompt)},
    ]


def baseline_messages(record: dict, include_gt: bool = False) -> list[dict]:
    """k=0 baseline messages: only the ``_clean_unicode_content`` mapping the
    vendored ``_make_api_call_cloud`` applies (``cloud_paper.py:87-90``)."""
    prompt = build_baseline_prompt(record["history"], record["question"],
                                   record["ground_truth"] if include_gt else None)
    return [
        {"role": "system", "content": _clean_unicode_content(SYSTEM_PROMPT)},
        {"role": "user", "content": _clean_unicode_content(prompt)},
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Method programs
# ─────────────────────────────────────────────────────────────────────────────

Program = Generator[list[list[dict]], list[str], dict]


def _finish(raw: str, schema_cases: list[int], num_schemata: int,
            agents: list[str] | None = None) -> dict:
    agent, step = parse_all_at_once(raw, agents)
    return {
        "predicted_agent": agent,
        "predicted_step": step,
        "raw": raw,
        "calls": [{"response": raw}],
        "schema_cases": list(schema_cases),
        "num_schemata": num_schemata,
    }


def correct_program(record: dict, *, analyzer, num_schemata: int,
                    include_gt: bool = False) -> Program:
    keys, contents = analyzer.get_similarity_based_schema(int(record["id"]), num_schemata)
    raw = (yield [correct_messages(record, keys, contents, include_gt)])[0]
    return _finish(raw, keys, num_schemata, agent_vocabulary(record["history"]))


def correct_baseline_program(record: dict, *, analyzer=None, num_schemata: int = 0,
                             include_gt: bool = False) -> Program:
    raw = (yield [baseline_messages(record, include_gt)])[0]
    return _finish(raw, [], 0, agent_vocabulary(record["history"]))


METHODS = {
    "correct": correct_program,              # needs schemata + similarities
    "correct_baseline": correct_baseline_program,
}
