"""Execution-level parity between the CORRECT methods and the vendored cloud path.

Loads ``vendored/CORRECT/src/Lib/cloud_paper.py`` (tqdm stubbed if absent),
drives the vendored analyze functions with a **fake OpenAI client** — so the
vendored unicode scrubbing inside ``_make_api_call_cloud`` runs for real — and
asserts our message builders produce byte-identical requests, for both the
schema-guided path and the k=0 baseline, including the GT-line variants and
every schema-injection branch.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

import baselines.prompting.methods as prompting_methods
from baselines.correct.methods import (
    baseline_messages,
    build_baseline_prompt,
    build_correct_base_prompt,
    correct_messages,
    modify_prompt_paper,
)
from baselines.prompting.predict import load_records

REPO_ROOT = Path(__file__).resolve().parents[1]
VENDORED_CLOUD = REPO_ROOT / "vendored/CORRECT/src/Lib/cloud_paper.py"

# Content deliberately exercises both scrub functions: smart quotes/dashes
# (mapped by both), bullets/CJK/NBSP (spaced by clean_text, kept by
# _clean_unicode_content).
HISTORIES = {
    "1": [
        {"role": "Orchestrator", "content": "Plan: “search” the web — then compute…"},
        {"role": "WebSurfer", "content": "I found • the population is 8,336,000\xa0people 中文"},
        {"role": "Assistant", "content": "Final answer: ‘16,672,000’"},
    ],
    "2": [
        {"role": "Planner", "content": "Break the task into steps."},
        {"role": "Computer_terminal", "content": "exit code 0"},
    ],
    "3": [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "MathAgent", "content": "The answer is 5."},
    ],
}
QUESTIONS = {"1": "What is twice the “population” of the city?",
             "2": "Do the task.", "3": "What is 2+2?"}
SCHEMAS = {
    1: "Agent Name: WebSurfer\nStep Number: 1\nReason for Mistake: mis-read — the page…",
    2: "### Error Schema\n1. Error Signatures:\n   • placeholder tokens",
    3: "Schema three with ‘quotes’",
}


def _stub(name: str, **attrs) -> None:
    if name in sys.modules:
        return
    try:
        __import__(name)
        return
    except ImportError:
        pass
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod


@pytest.fixture(scope="module")
def vendored():
    _stub("tqdm", tqdm=lambda it, **kw: it)
    spec = importlib.util.spec_from_file_location("vendored_cloud_paper", VENDORED_CLOUD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def data_dir(tmp_path):
    for stem, history in HISTORIES.items():
        doc = {
            "history": history,
            "question": QUESTIONS[stem],
            "ground_truth": f"answer-{stem}",
            "question_ID": f"qid-{stem}",
            "mistake_agent": "WebSurfer",
            "mistake_step": "1",
        }
        (tmp_path / f"{stem}.json").write_text(json.dumps(doc), encoding="utf-8")
    return tmp_path


class FakeClient:
    """Captures chat.completions.create kwargs — the exact bytes 'sent'."""

    def __init__(self, response: str = "Agent Name: X\nStep Number: 0"):
        self.captured: list[dict] = []

        def create(**kwargs):
            self.captured.append(kwargs)
            msg = SimpleNamespace(content=response)
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))


class ScriptedAnalyzer:
    """Returns fixed (keys, contents) per file number, like the vendored analyzer."""

    def __init__(self, per_file: dict[int, tuple[list[int], list[str]]]):
        self.per_file = per_file

    def get_similarity_based_schema(self, file_num, num_schemata=1):
        return self.per_file.get(file_num, ([], []))


def _by_user_content(requests: list[list[dict]]) -> list[list[dict]]:
    return sorted(requests, key=lambda msgs: msgs[-1]["content"])


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    import time

    monkeypatch.setattr(time, "sleep", lambda s: None)


@pytest.mark.parametrize("k", [1, 2])
def test_correct_prompt_parity(vendored, data_dir, k):
    per_file = {
        1: ([2], [SCHEMAS[2]]) if k == 1 else ([2, 3], [SCHEMAS[2], SCHEMAS[3]]),
        2: ([1], [SCHEMAS[1]]) if k == 1 else ([1, 3], [SCHEMAS[1], SCHEMAS[3]]),
        3: ([], []),  # empty retrieval → fallback block
    }
    analyzer = ScriptedAnalyzer(per_file)
    client = FakeClient()
    vendored.analyze_all_at_once_cloud_with_schemata_parallel(
        client, str(data_dir), is_handcrafted=True, model="gpt-4o",
        max_tokens=1024, model_type="gpt", schema_analyzer=analyzer,
        num_schemata=k, batch_size=10, max_workers=1,
    )
    assert len(client.captured) == 3
    for kwargs in client.captured:
        assert kwargs["model"] == "gpt-4o" and kwargs["max_tokens"] == 1024

    ours = []
    for record in load_records(str(data_dir)):
        keys, contents = analyzer.get_similarity_based_schema(int(record["id"]), k)
        ours.append(correct_messages(record, keys, contents, include_gt=False))
    assert _by_user_content([c["messages"] for c in client.captured]) == _by_user_content(ours)


def test_baseline_prompt_parity(vendored, data_dir):
    client = FakeClient()
    vendored.all_at_once_baseline_parallel(
        client, str(data_dir), is_handcrafted=True, model="gpt-4o",
        max_tokens=1024, model_type="gpt", batch_size=10, max_workers=1,
    )
    assert len(client.captured) == 3
    ours = [baseline_messages(r, include_gt=False) for r in load_records(str(data_dir))]
    assert _by_user_content([c["messages"] for c in client.captured]) == _by_user_content(ours)


def test_gpt5_max_completion_tokens(vendored, data_dir, monkeypatch):
    monkeypatch.delenv("OPENAI_REASONING_EFFORT", raising=False)
    client = FakeClient()
    vendored.all_at_once_baseline_parallel(
        client, str(data_dir), is_handcrafted=True, model="gpt-5",
        max_tokens=4096, model_type="gpt", batch_size=10, max_workers=1,
    )
    for kwargs in client.captured:
        assert kwargs["max_completion_tokens"] == 4096 and "max_tokens" not in kwargs


@pytest.mark.parametrize("nkeys", [0, 1, 3])
def test_injection_branches(vendored, nkeys):
    base = "BASE PROMPT\n"
    keys = [7, 42, 13][:nkeys]
    contents = [SCHEMAS[1], SCHEMAS[2], SCHEMAS[3]][:nkeys]
    assert modify_prompt_paper(base, keys, contents) == \
        vendored._correct_modify_prompt_paper(base, keys, contents, wording="template")


def test_gt_line_bytes():
    history = HISTORIES["3"]
    with_gt = build_correct_base_prompt(history, "Q", "A")
    assert "The problem is:  Q \nThe Answer for the problem is: A\nIdentify which agent" in with_gt
    without = build_correct_base_prompt(history, "Q", None)
    assert "The Answer for the problem is" not in without
    # Baseline: no space before the newline after the problem.
    b_with = build_baseline_prompt(history, "Q", "A")
    assert "The problem is:  Q\nThe Answer for the problem is: A\nIdentify which agent" in b_with
    assert "The Answer for the problem is" not in build_baseline_prompt(history, "Q", None)
    # The GT line is the ONLY difference within each builder.
    assert with_gt.replace("The Answer for the problem is: A\n", "") == without
    assert b_with.replace("The Answer for the problem is: A\n", "") == build_baseline_prompt(history, "Q", None)


def test_baseline_vs_correct_prompt_divergences(vendored, data_dir):
    """The three byte-level differences between the two base prompts."""
    record = load_records(str(data_dir))[2]  # "3.json", pure-ASCII content
    base = build_correct_base_prompt(record["history"], record["question"])
    baseline = build_baseline_prompt(record["history"], record["question"])
    assert f"The problem is:  {record['question']} \n" in base
    assert f"The problem is:  {record['question']}\nIdentify" in baseline
    assert '{\n"agent a": "xx",' in base
    assert '\n        {\n            "agent a": "xx",' in baseline
    assert base.endswith("Reason for Mistake: (Your reason)\n")
    assert baseline.endswith("Reason for Mistake: \n")


def test_parse_reuses_prompting(vendored):
    """Parsing is shared with the prompting baseline, and the vendored CORRECT
    evaluate.py uses the same regex family that parser extends."""
    import baselines.correct.methods as m

    assert m.parse_all_at_once is prompting_methods.parse_all_at_once
    evaluate_src = (REPO_ROOT / "vendored/CORRECT/src/evaluate.py").read_text(encoding="utf-8")
    assert r"Agent Name:\s*([\w_]+)" in evaluate_src
    assert r"Step Number:\s*(\d+)" in evaluate_src


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval parity (inference_whoandwhen.py::SimilarityBasedSchemaAnalyzer)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def vendored_whoandwhen():
    _stub("tqdm", tqdm=lambda it, **kw: it)
    _stub("dotenv", load_dotenv=lambda *a, **kw: None)
    _stub("torch")
    _stub("transformers", pipeline=None, AutoTokenizer=None,
          AutoModelForCausalLM=None, Pipeline=type("Pipeline", (), {}))
    _stub("openai", OpenAI=object, AzureOpenAI=object)
    _stub("vllm", LLM=None, SamplingParams=None)
    src_dir = str(REPO_ROOT / "vendored/CORRECT/src")
    sys.path.insert(0, src_dir)
    try:
        spec = importlib.util.spec_from_file_location(
            "vendored_inference_whoandwhen",
            REPO_ROOT / "vendored/CORRECT/src/inference_whoandwhen.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(src_dir)
    return mod


def test_retrieval_parity(vendored_whoandwhen):
    from baselines.correct.retrieval import SchemaAnalyzer

    schemata = {1: "S1", 2: "S2", 4: "S4"}  # 3 has no schema
    similarities = {1: [3, 2, 4], 2: [1, 4, 3], 5: []}
    theirs = vendored_whoandwhen.SimilarityBasedSchemaAnalyzer(schemata, similarities)
    ours = SchemaAnalyzer(schemata, similarities)

    cases = [
        (1, 1),   # top-1 neighbour lacks a schema → silently empty
        (1, 2),   # slice [3, 2] → only 2 has a schema
        (2, 2),   # full cache in slice
        (2, 10),  # k > list length
        (5, 3),   # empty neighbour list
        (9, 1),   # unknown file_num
    ]
    for file_num, k in cases:
        assert ours.get_similarity_based_schema(file_num, k) == \
            theirs.get_similarity_based_schema(file_num, k), (file_num, k)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: schema-generation prompt parity (error_schema_generator_cloud.py)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def vendored_schemagen():
    _stub("tqdm", tqdm=lambda it, **kw: it)
    _stub("openai", OpenAI=object)
    _stub("google")
    _stub("google.generativeai")
    spec = importlib.util.spec_from_file_location(
        "vendored_schemagen_cloud",
        REPO_ROOT / "vendored/CORRECT/src/error_schema_generator_cloud.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _schemagen_record(tmp_path, doc):
    from baselines.correct.schemagen import load_schemagen_records

    (tmp_path / "1.json").write_text(json.dumps(doc), encoding="utf-8")
    return load_schemagen_records(str(tmp_path))[0]


WW_DOC = {
    "history": HISTORIES["1"],
    "question": "What is twice the “population” of the city?",
    "ground_truth": "16,672,000",
    "question_ID": "qid-1",
    "mistake_agent": "WebSurfer",
    "mistake_step": "1",
    "mistake_reason": "mis-read the page",
}


def test_schemagen_prompt_parity(vendored_schemagen, tmp_path):
    from baselines.correct.schemagen import schemagen_messages

    client = FakeClient(response="### Error Schema ...")
    for name, doc in {
        "ww": WW_DOC,
        # CE-shaped: both ground-truth spellings present and equal (as released
        # upstream — empty; and as restored in data/correct-error-gt — equal).
        "ce": {**WW_DOC, "ground_truth": "", "groundtruth": "", "mistake_step": 1},
        "ce-gt": {**WW_DOC, "ground_truth": "B", "groundtruth": "B"},
    }.items():
        d = tmp_path / name
        d.mkdir()
        record = _schemagen_record(d, doc)
        vendored_schemagen.generate_schema_gpt(doc, client, "gpt-4o", 1024)
        assert client.captured[-1]["messages"] == schemagen_messages(record), name
        assert client.captured[-1]["max_tokens"] == 1024
        assert "temperature" not in client.captured[-1]  # commented out upstream


def test_schemagen_gpt5_params(vendored_schemagen):
    client = FakeClient(response="x")
    vendored_schemagen.generate_schema_gpt(WW_DOC, client, "gpt-5", 4096)
    assert client.captured[-1]["max_completion_tokens"] == 4096
    assert "max_tokens" not in client.captured[-1]
    assert "temperature" not in client.captured[-1]


def test_schemagen_role_key_deviation(vendored_schemagen, tmp_path):
    """On this repo's algorithm-generated layout (``role`` = agent name,
    ``name`` = user/assistant) the vendored autodetect would pick ``name`` and
    label turns user/assistant. We fix ``role``, which equals what the vendored
    code produces on the *original* Who&When layout — pinned by comparing
    against a name-stripped copy."""
    from baselines.correct.schemagen import build_schemagen_prompt

    doc = {**WW_DOC,
           "history": [{"role": "Excel_Expert", "name": "assistant", "content": "sum it"},
                       {"role": "user", "name": "user", "content": "thanks"}]}
    d = tmp_path / "alggen"
    d.mkdir()
    record = _schemagen_record(d, doc)
    stripped = {**doc, "history": [{k: v for k, v in e.items() if k != "name"}
                                   for e in doc["history"]]}
    assert build_schemagen_prompt(record) == vendored_schemagen.create_prompt(stripped)
    assert build_schemagen_prompt(record) != vendored_schemagen.create_prompt(doc)
    assert "Step 0: Excel_Expert: sum it" in build_schemagen_prompt(record)


def test_clean_text_pins(vendored):
    from baselines.correct.methods import _clean_unicode_content, clean_text

    samples = [
        "“smart” ‘quotes’ – and — dashes … done",
        "bullets • and\xa0nbsp and 中文 mixed",
        "plain ascii stays untouched\n\ttabs too",
        None,
    ]
    for s in samples:
        if s is None:
            assert clean_text(s) == vendored.clean_text(s) == ""
            continue
        assert clean_text(s) == vendored.clean_text(s)
        assert _clean_unicode_content(s) == vendored._clean_unicode_content(s)
    # Behavior pins: clean_text spaces out what _clean_unicode_content keeps.
    assert clean_text("a•b 中") == "a b  "
    assert _clean_unicode_content("a•b 中") == "a•b 中"
    assert clean_text("“x”") == _clean_unicode_content("“x”") == '"x"'
    # clean_text output is ASCII, so the vendored API-call-time
    # _clean_unicode_content is a no-op on it (the documented claim).
    for s in samples[:3]:
        assert _clean_unicode_content(clean_text(s)) == clean_text(s)
