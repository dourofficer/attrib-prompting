"""One-off capture of golden RAFFLES prompt strings from the prompt builders.

Run from the repo root:  python tests/fixtures/capture_raffles_goldens.py

RAFFLES has no vendored code, and this repo runs deliberately simplified
prompts (see ``baselines/raffles/prompts.py``). These goldens pin the current
wording: the test suite asserts the builders keep reproducing these bytes
character-for-character, and ``tests/test_raffles_pipeline.py`` separately
asserts the prompts' core-contract sentences, so a fixture regeneration cannot
silently change what the prompts ask for.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from baselines.raffles.prompts import (  # noqa: E402
    build_evaluator_prompt,
    build_judge_prompt,
)

HISTORY = [
    {"role": "Orchestrator", "content": "Plan: search the web, then compute."},
    {"role": "WebSurfer", "content": "I found the population is 8,336,000."},
    {"role": "Assistant", "content": "Final answer: 8,336,000 * 2 = 16,672,000."},
]
RECORD = {
    "history": HISTORY,
    "question": "What is twice the population of the city?",
    "ground_truth": "16,672,000",
}
CANDIDATE = {
    "agent_name": "WebSurfer",
    "step_number": 1,
    "mistake_reason": "The population figure is outdated.",
    "first_mistake": "Every later step builds on the wrong figure.",
    "mistake_not_corrected": "No later agent re-checked the figure.",
}
FEEDBACK = [{
    "iteration": 0,
    "answer": '{"agent_name": "Orchestrator", "step_number": 0}',
    "evals": [
        {"criterion": 1, "confidence": 20, "reason": "The plan itself is fine."},
        {"criterion": 2, "confidence": 10, "reason": "The mistake happens later."},
        {"criterion": 3, "confidence": 90, "reason": "Nothing was corrected."},
        {"criterion": 4, "confidence": 100, "reason": "Step 0 exists in the log."},
    ],
}]

goldens = {
    "judge": build_judge_prompt(RECORD),
    "judge_with_gt": build_judge_prompt(RECORD, include_gt=True),
    "judge_with_feedback": build_judge_prompt(RECORD, FEEDBACK),
    "evaluator_1": build_evaluator_prompt(1, RECORD, CANDIDATE),
    "evaluator_2": build_evaluator_prompt(2, RECORD, CANDIDATE),
    "evaluator_3": build_evaluator_prompt(3, RECORD, CANDIDATE),
}

out = Path(__file__).parent / "raffles_golden_prompts.json"
out.write_text(json.dumps(goldens, indent=2, ensure_ascii=False))
print(f"wrote {out} ({len(goldens)} prompts)")
