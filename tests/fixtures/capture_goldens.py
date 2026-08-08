"""One-off capture of golden prompt strings from the prompt builders.

Run from the repo root:  python tests/fixtures/capture_goldens.py

Captured BEFORE the standalone refactor so the goldens pin the exact prompt
bytes of the original (vendored-faithful) builders; the test suite asserts the
refactored builders still reproduce them character-for-character.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# The original engine.py imports src.models at module level; stub it so
# methods.py can be imported without the (removed) attribscope src package.
if "src" not in sys.modules:  # pragma: no branch
    src_pkg = types.ModuleType("src")
    src_models = types.ModuleType("src.models")
    src_models.get_adapter = lambda *_a, **_k: None
    src_pkg.models = src_models
    sys.modules["src"] = src_pkg
    sys.modules["src.models"] = src_models

from baselines.prompting.methods import (  # noqa: E402
    build_all_at_once_prompt,
    build_binary_search_prompt,
    build_step_by_step_prompt,
)

HISTORY = [
    {"role": "Orchestrator", "content": "Plan: search the web, then compute."},
    {"role": "WebSurfer", "content": "I found the population is 8,336,000."},
    {"role": "Assistant", "content": "Final answer: 8,336,000 * 2 = 16,672,000."},
]
PROBLEM = "What is twice the population of the city?"
GROUND_TRUTH = "16,672,000"

goldens = {
    "all_at_once": build_all_at_once_prompt(HISTORY, PROBLEM, GROUND_TRUTH),
    "step_by_step": build_step_by_step_prompt(
        PROBLEM,
        GROUND_TRUTH,
        "Step 0 - Orchestrator: Plan: search the web, then compute.\n",
        0,
        "Orchestrator",
    ),
    "binary_search": build_binary_search_prompt(
        PROBLEM,
        GROUND_TRUTH,
        "Orchestrator: Plan: search the web, then compute.\n"
        "WebSurfer: I found the population is 8,336,000.",
        range_description="from step 0 to step 1",
        upper_half_desc="from step 0 to step 0",
        lower_half_desc="from step 1 to step 1",
    ),
}

out = Path(__file__).parent / "golden_prompts.json"
out.write_text(json.dumps(goldens, indent=2, ensure_ascii=False))
print(f"wrote {out}")
for name, text in goldens.items():
    print(f"  {name}: {len(text)} chars")
