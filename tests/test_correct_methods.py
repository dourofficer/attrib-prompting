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
    assert issued[0] == correct_messages(RECORD, [3], ["A schema about WebSurfer errors."], include_gt=False)
    assert doc["predicted_agent"] == "WebSurfer" and doc["predicted_step"] == 1
    assert doc["schema_cases"] == [3] and doc["num_schemata"] == 1
    assert doc["calls"] == [{"response": "Agent Name: WebSurfer\nStep Number: 1\nReason for Mistake: x"}]
    assert doc["raw"].startswith("Agent Name:")


def test_baseline_program_doc_shape():
    issued, doc = _drive(correct_baseline_program(RECORD), ["Agent Name: Orchestrator\nStep Number: 0"])
    assert issued == [baseline_messages(RECORD, include_gt=False)]
    assert doc["predicted_agent"] == "Orchestrator" and doc["predicted_step"] == 0
    assert doc["schema_cases"] == [] and doc["num_schemata"] == 0


def test_unparseable_response_gives_none():
    issued, doc = _drive(correct_baseline_program(RECORD), ["I have no idea."])
    assert doc["predicted_agent"] is None and doc["predicted_step"] is None
    assert doc["raw"] == "I have no idea."


@pytest.mark.parametrize("include_gt", [False, True])
def test_gt_flag_controls_answer_line(include_gt):
    """GUIDE GT axis: include_gt inserts the answer line; the without-GT bytes
    are the vendored (paper) default for this baseline."""
    for messages in (correct_messages(RECORD, [], [], include_gt=include_gt),
                     baseline_messages(RECORD, include_gt=include_gt)):
        user = messages[1]["content"]
        assert ("The Answer for the problem is: 42" in user) == include_gt


def test_methods_registry():
    assert set(METHODS) == {"correct", "correct_baseline"}
    for program in METHODS.values():
        assert callable(program)


def test_reasoning_output_parses():
    raw = "<think>Maybe WebSurfer at step 9?</think>**Agent Name:** (WebSurfer)\n**Step Number:** 1"
    _, doc = _drive(correct_baseline_program(RECORD), [raw])
    assert doc["predicted_agent"] == "WebSurfer" and doc["predicted_step"] == 1
