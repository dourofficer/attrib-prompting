"""Program-level behavior of the CORRECT methods (scripted responses, no models)."""
from __future__ import annotations

import pytest

from baselines.correct.methods import (
    METHODS,
    baseline_messages,
    correct_baseline_program,
    correct_messages,
    correct_program,
)

RECORD = {
    "id": "7",
    "filename": "7.json",
    "question_id": "qid-7",
    "history": [
        {"role": "Orchestrator", "content": "Plan the work."},
        {"role": "WebSurfer", "content": "Found it."},
    ],
    "question": "What is the answer?",
    "ground_truth": "42",
    "gold_agent": "WebSurfer",
    "gold_step": 1,
}


class OneSchemaAnalyzer:
    def get_similarity_based_schema(self, file_num, num_schemata=1):
        assert file_num == 7
        return [3], ["A schema about WebSurfer errors."]


def _drive(program, responses):
    issued, script = [], list(responses)
    try:
        prompts = next(program)
        while True:
            issued.extend(prompts)
            outs = [script.pop(0) for _ in prompts]
            prompts = program.send(outs)
    except StopIteration as si:
        return issued, si.value


def test_correct_program_doc_shape():
    gen = correct_program(RECORD, analyzer=OneSchemaAnalyzer(), num_schemata=1)
    issued, doc = _drive(gen, ["Agent Name: WebSurfer\nStep Number: 1\nReason for Mistake: x"])
    assert len(issued) == 1
    assert issued[0] == correct_messages(RECORD, [3], ["A schema about WebSurfer errors."], with_gt=False)
    assert doc["predicted_agent"] == "WebSurfer" and doc["predicted_step"] == 1
    assert doc["schema_cases"] == [3] and doc["num_schemata"] == 1
    assert doc["calls"] == [{"response": "Agent Name: WebSurfer\nStep Number: 1\nReason for Mistake: x"}]
    assert doc["raw"].startswith("Agent Name:")


def test_baseline_program_doc_shape():
    issued, doc = _drive(correct_baseline_program(RECORD), ["Agent Name: Orchestrator\nStep Number: 0"])
    assert issued == [baseline_messages(RECORD, with_gt=False)]
    assert doc["predicted_agent"] == "Orchestrator" and doc["predicted_step"] == 0
    assert doc["schema_cases"] == [] and doc["num_schemata"] == 0


def test_unparseable_response_gives_none():
    issued, doc = _drive(correct_baseline_program(RECORD), ["I have no idea."])
    assert doc["predicted_agent"] is None and doc["predicted_step"] is None
    assert doc["raw"] == "I have no idea."


@pytest.mark.parametrize("with_gt", [False, True])
def test_gt_flag_controls_answer_line(with_gt):
    for messages in (correct_messages(RECORD, [], [], with_gt=with_gt),
                     baseline_messages(RECORD, with_gt=with_gt)):
        user = messages[1]["content"]
        assert ("The Answer for the problem is: 42" in user) == with_gt


def test_methods_registry():
    assert set(METHODS) == {"correct", "correct_gt", "correct_baseline", "correct_baseline_gt"}
    for name, (program, with_gt, needs_artifacts) in METHODS.items():
        assert with_gt == name.endswith("_gt")
        assert needs_artifacts == (not name.startswith("correct_baseline"))
        assert callable(program)


def test_reasoning_output_parses():
    raw = "<think>Maybe WebSurfer at step 9?</think>**Agent Name:** (WebSurfer)\n**Step Number:** 1"
    _, doc = _drive(correct_baseline_program(RECORD), [raw])
    assert doc["predicted_agent"] == "WebSurfer" and doc["predicted_step"] == 1
