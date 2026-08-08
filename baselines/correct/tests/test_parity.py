"""Byte-level parity tests: baselines/correct vs the vendored baselines/CORRECT.

Feeds identical dummy trajectories / schemata / canned LLM outputs through both
codebases and asserts byte-identical prompts, deep-equal retrievals/parses, and
identical similarity rankings. The vendored inference functions are exercised
end-to-end with ``vllm`` stubbed out and fake capture classes monkeypatched in,
so no GPU (or vllm install) is needed.

Run from the repo root:

    python -m pytest baselines/correct/tests/test_parity.py -q
    # or, without pytest:
    python -m baselines.correct.tests.test_parity
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import re
import sys
import tempfile
import types
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[3]
VENDORED_SRC = REPO_ROOT / "baselines" / "CORRECT" / "src"

sys.path.insert(0, str(REPO_ROOT))

from baselines.correct import methods, retrieval, similarity  # noqa: E402
from baselines.correct.schemagen import build_schema_prompt  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Vendored-module loading (with vllm & co. stubbed)
# ─────────────────────────────────────────────────────────────────────────────

class _FakeSamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeOutput:
    def __init__(self, text):
        self.text = text


class _FakeRequestOutput:
    def __init__(self, text):
        self.outputs = [_FakeOutput(text)]


class FakeLLM:
    """Captures the formatted prompts the vendored code hands to vllm."""

    captured_prompts: list[str] = []
    canned_response = ("Agent Name: WebSurfer\n, Step Number: 3\n, "
                       "Reason for Mistake: it clicked the wrong link\n")

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def generate(self, prompts, sampling_params):
        FakeLLM.captured_prompts.extend(prompts)
        return [_FakeRequestOutput(FakeLLM.canned_response) for _ in prompts]


class FakeTokenizer:
    """Deterministic stand-in for AutoTokenizer.apply_chat_template."""

    padding_side = "right"

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return json.dumps({"messages": messages, "gen": add_generation_prompt})


def _stub_module(**attrs) -> types.ModuleType:
    mod = types.ModuleType("stub")
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


_VLLM_STUB = _stub_module(LLM=FakeLLM, SamplingParams=_FakeSamplingParams)
_DOTENV_STUB = _stub_module(load_dotenv=lambda *a, **k: None)

_vendored_cache: dict[str, types.ModuleType] = {}


def load_vendored(relpath: str) -> types.ModuleType:
    """Exec a vendored source file with heavy/absent deps stubbed in sys.modules."""
    if relpath in _vendored_cache:
        return _vendored_cache[relpath]

    stubs = {"vllm": _VLLM_STUB, "dotenv": _DOTENV_STUB}
    saved = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    sys.path.insert(0, str(VENDORED_SRC))
    try:
        name = "vendored_" + relpath.replace("/", "_").removesuffix(".py")
        spec = importlib.util.spec_from_file_location(name, VENDORED_SRC / relpath)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        with contextlib.redirect_stdout(io.StringIO()):
            spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(VENDORED_SRC))
        for mod_name, orig in saved.items():
            if orig is None:
                sys.modules.pop(mod_name, None)
            else:
                sys.modules[mod_name] = orig

    _vendored_cache[relpath] = module
    return module


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def make_records() -> dict[int, dict]:
    """Dummy trajectories keyed by file number (role-keyed, like all repo data)."""
    return {
        1: {
            "question": "What is the capital of France?",
            "ground_truth": "Paris",
            "history": [
                {"role": "Orchestrator", "content": "Let us plan.\nFirst search."},
                {"role": "WebSurfer", "content": "I clicked the wrong link."},
                {"role": "Orchestrator (thought)", "content": "Summarize: Lyon."},
            ],
            "mistake_agent": "WebSurfer",
            "mistake_step": "1",
            "mistake_reason": "Clicked an unrelated result.",
        },
        2: {
            "question": "Sum 2+2?",
            "ground_truth": "4",
            "history": [
                {"role": "Planner", "content": "Compute the sum."},
                {"role": "CodeExecutor", "content": "print(5)"},
            ],
            "mistake_agent": "CodeExecutor",
            "mistake_step": "1",
            "mistake_reason": "Printed the wrong constant.",
        },
        3: {
            "question": "Name the largest planet.",
            "ground_truth": "Jupiter",
            "history": [
                {"role": "Assistant", "content": "It is Saturn."},
            ],
            "mistake_agent": "Assistant",
            "mistake_step": "0",
            "mistake_reason": "Wrong planet.",
        },
    }


def make_schemata() -> dict[int, str]:
    return {
        1: ("Agent Name: WebSurfer\nStep Number: 1\nReason for Mistake:\n"
            "1. Error Signatures:\n   - clicks unrelated results\n"
            "2. Error Context Analysis:\n   - search tasks\n"
            "3. Detection Heuristics:\n   - is the link on-topic?"),
        2: "Agent Name: CodeExecutor\nStep Number: 1\nReason: constant mismatch.",
        3: "Agent Name: Assistant\nStep Number: 0\nReason: factual slip.",
    }


def write_trajectory_dir(tmp: Path, records: dict[int, dict]) -> Path:
    d = tmp / "trajs"
    d.mkdir()
    for num, rec in records.items():
        # The vendored schema generator reads the raw CORRECT-Error key
        # ``groundtruth``; this repo's data stores ``ground_truth``. Write both
        # (with the same value) so each codepath reads its own key.
        raw = dict(rec)
        raw["groundtruth"] = rec["ground_truth"]
        (d / f"{num}.json").write_text(json.dumps(raw), encoding="utf-8")
    return d


def our_formatted(prompt: str) -> str:
    """Our messages, rendered with the same fake template as the vendored side."""
    return FakeTokenizer().apply_chat_template(methods.messages(prompt), tokenize=False,
                                               add_generation_prompt=True)


@contextlib.contextmanager
def patched_local_model():
    lm = load_vendored("Lib/local_model.py")
    FakeLLM.captured_prompts = []
    with mock.patch.object(lm, "LLM", FakeLLM), \
         mock.patch.object(lm, "SamplingParams", _FakeSamplingParams), \
         mock.patch.object(lm, "AutoTokenizer", FakeTokenizer), \
         contextlib.redirect_stdout(io.StringIO()):
        yield lm


# ─────────────────────────────────────────────────────────────────────────────
# Prompt parity — vendored vLLM path, end to end
# ─────────────────────────────────────────────────────────────────────────────

def test_base_prompt_parity():
    """analyze_all_at_once_vllm (k=0 baseline) builds our exact prompts."""
    records = make_records()
    with tempfile.TemporaryDirectory() as tmp:
        traj_dir = write_trajectory_dir(Path(tmp), records)
        for is_handcrafted in (True, False):  # role- vs name-preferred: equal on role-keyed data
            with patched_local_model() as lm:
                lm.analyze_all_at_once_vllm("fake-model", str(traj_dir),
                                            is_handcrafted=is_handcrafted)
                vendored = list(FakeLLM.captured_prompts)

            ours = [
                our_formatted(methods.build_all_at_once_prompt(rec["history"], rec["question"]))
                for _, rec in sorted(records.items())
            ]
            assert vendored == ours, f"base prompt mismatch (is_handcrafted={is_handcrafted})"


def test_schema_injected_prompt_parity():
    """_run_vllm_generation_with_schemata injects schemata exactly like inject_schemata."""
    records = make_records()
    schemata = make_schemata()
    cases = {  # file_num -> schema payload, covering all vendored branches
        1: [schemata[2]],                            # single-element list
        2: [schemata[1], schemata[3]],               # multi-schema list
        3: schemata[1],                              # bare string branch
    }
    with tempfile.TemporaryDirectory() as tmp:
        traj_dir = write_trajectory_dir(Path(tmp), records)
        with patched_local_model() as lm:
            lm.analyze_all_at_once_vllm_with_schemata(
                "fake-model", str(traj_dir), is_handcrafted="True", schemata=dict(cases))
            vendored = list(FakeLLM.captured_prompts)

    ours = [
        our_formatted(methods.inject_schemata(
            methods.build_all_at_once_prompt(records[num]["history"], records[num]["question"]),
            cases[num]))
        for num in sorted(records)
    ]
    assert vendored == ours


def test_no_schema_fallback_prompt_parity():
    """A file with no retrieved schema gets the plain base prompt in both codebases."""
    records = make_records()
    with tempfile.TemporaryDirectory() as tmp:
        traj_dir = write_trajectory_dir(Path(tmp), records)
        with patched_local_model() as lm:
            lm.analyze_all_at_once_vllm_with_schemata(
                "fake-model", str(traj_dir), is_handcrafted="True", schemata={})
            vendored = list(FakeLLM.captured_prompts)
    ours = [
        our_formatted(methods.build_all_at_once_prompt(rec["history"], rec["question"]))
        for _, rec in sorted(records.items())
    ]
    assert vendored == ours


def test_schema_generation_prompt_parity():
    """schemagen.build_schema_prompt matches the vendored create_prompt byte-for-byte."""
    gen = load_vendored("error_schema_generator.py")
    for _, rec in sorted(make_records().items()):
        raw = dict(rec)
        raw["groundtruth"] = rec["ground_truth"]  # vendored key
        assert gen.create_prompt(raw, tokenizer=None) == build_schema_prompt(rec)


def test_response_trim_parity():
    """The vendored post-generation trim equals trim_response(strip_think(.))."""
    samples = [
        "Agent Name: WebSurfer\n, Step Number: 3\n, Reason for Mistake: bad link\n",
        "Some preamble.\n\nAgent Name: Planner\n, Step Number: 0\n, Reason for Mistake: x\n\nAgent Name: Foo\n, Step Number: 9\n",
        "no structured answer here",
        "Agent Name: A\n\nStep Number: 2",  # vendored trim cuts at the blank line
    ]
    records = {1: make_records()[1]}
    orig_response = FakeLLM.canned_response
    try:
        for text in samples:
            with tempfile.TemporaryDirectory() as tmp:
                traj_dir = write_trajectory_dir(Path(tmp), records)
                with patched_local_model() as lm:
                    FakeLLM.canned_response = text
                    buf = io.StringIO()
                    with contextlib.redirect_stdout(buf):
                        lm.analyze_all_at_once_vllm("fake-model", str(traj_dir),
                                                    is_handcrafted=True)
                m = re.search(r"Prediction for 1\.json:\n(.*?)\n\n=+\n", buf.getvalue(), re.DOTALL)
                vendored_trimmed = m.group(1)
            assert vendored_trimmed == methods.trim_response(methods.strip_think(text)), repr(text)
    finally:
        FakeLLM.canned_response = orig_response


# ─────────────────────────────────────────────────────────────────────────────
# Schemata file round-trip & retrieval parity
# ─────────────────────────────────────────────────────────────────────────────

def test_schemata_file_roundtrip_parity():
    """write_schemata_file output parses identically in ours and both vendored loaders."""
    schemata = make_schemata()
    ww = load_vendored("inference_whoandwhen.py")
    ce = load_vendored("inference_correct_error.py")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "error_schemata.txt"
        retrieval.write_schemata_file(schemata, path)

        ours = retrieval.load_error_schemata(path)
        with contextlib.redirect_stdout(io.StringIO()):
            vendored_ww = ww.load_error_schemata(str(path))
            # correct_error's loader takes (dataset, dir) with a fixed layout.
            ds_dir = Path(tmp) / "schemata" / "dummy"
            ds_dir.mkdir(parents=True)
            (ds_dir / "error_schemata.txt").write_text(path.read_text(), encoding="utf-8")
            vendored_ce = ce.load_error_schemata_for_dataset("dummy", str(Path(tmp) / "schemata"))

    expected = {k: v.strip() for k, v in schemata.items()}
    assert ours == expected
    assert vendored_ww == expected
    assert vendored_ce == expected


def test_retrieval_parity():
    """SchemaAnalyzer reproduces both vendored analyzers, on complete and holey caches."""
    ww = load_vendored("inference_whoandwhen.py")
    ce = load_vendored("inference_correct_error.py")

    similarities = {
        1: [3, 2, 5, 4, 7, 6],
        2: [1, 3, 4, 5, 6, 7],
        3: [7, 6, 5, 4, 2, 1],
    }
    complete = {n: f"schema-{n}" for n in range(1, 8)}
    holey = {n: f"schema-{n}" for n in (2, 4, 6)}  # gaps exercise the scan variants

    for cache in (complete, holey):
        with contextlib.redirect_stdout(io.StringIO()):
            vendored_ww = ww.SimilarityBasedSchemaAnalyzer(cache, similarities)
            vendored_ce = ce.DatasetSimilaritySchemaAnalyzer.__new__(ce.DatasetSimilaritySchemaAnalyzer)
            vendored_ce.schemata = cache
            vendored_ce.similarities = similarities
            vendored_ce.use_random_fallback = False
            vendored_ce.schema_list = list(cache.values())
            vendored_ce.schema_keys = list(cache.keys())

        ours_topk = retrieval.SchemaAnalyzer(cache, similarities, scan_until_filled=False)
        ours_scan = retrieval.SchemaAnalyzer(cache, similarities, scan_until_filled=True)

        for file_num in (1, 2, 3, 99):  # 99: no similarity entry
            for k in (1, 2, 3, 5, 10):
                with contextlib.redirect_stdout(io.StringIO()):
                    expected_ww = vendored_ww.get_similarity_based_schema(file_num, k)
                    expected_ce = vendored_ce.get_similarity_based_schema(file_num, k)
                assert ours_topk.get_similarity_based_schema(file_num, k) == expected_ww, \
                    (cache is holey, file_num, k, "ww")
                assert ours_scan.get_similarity_based_schema(file_num, k) == expected_ce, \
                    (cache is holey, file_num, k, "ce")


# ─────────────────────────────────────────────────────────────────────────────
# Prediction parsing parity (vendored evaluate.py)
# ─────────────────────────────────────────────────────────────────────────────

def test_parse_parity_with_vendored_evaluate():
    """parse_prediction agrees with evaluate.py's regexes on trimmed outputs."""
    ev = load_vendored("evaluate.py")
    outputs = {
        "1.json": "Agent Name: WebSurfer\n, Step Number: 3\n, Reason for Mistake: bad link",
        "2.json": "Agent Name: Code_Executor\n, Step Number: 12\n, Reason for Mistake: x",
        "3.json": "agent name: planner\n, step number: 0\n, reason: lower-case labels",
        "4.json": "nothing parseable",
    }
    # Build a prediction log exactly like the vendored stdout format.
    log = ""
    for fn, text in outputs.items():
        log += f"Prediction for {fn}:\n{text}\n\n{'=' * 50}\n\n"
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "preds.txt"
        log_path.write_text(log, encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            vendored = ev.read_predictions(str(log_path))

    for fn, text in outputs.items():
        agent, step, _ = methods.parse_prediction(text)
        if fn in vendored:
            assert agent == vendored[fn]["predicted_agent"]
            assert step == int(vendored[fn]["predicted_step"])
        else:
            assert agent is None or step is None  # unparseable in both


def test_reasoning_output_robustness():
    """<think> blocks and markdown never corrupt the trim/parse (our deviation)."""
    cases = [
        # think block that mentions the label — would corrupt a naive trim
        ("<think>Maybe Agent Name: Foo\n\nno wait.</think>\n"
         "Agent Name: WebSurfer\n, Step Number: 3\n, Reason for Mistake: x",
         ("WebSurfer", 3)),
        # dangling closer
        ("...trailing reasoning</think>Agent Name: Planner\n, Step Number: 1\n, Reason for Mistake: y",
         ("Planner", 1)),
        # bolded markdown labels (deepseek style)
        ("**Agent Name:** CodeExecutor\n**Step Number:** 7\n**Reason for Mistake:** z",
         ("CodeExecutor", 7)),
        # unterminated opener → nothing left to parse
        ("<think>never closes, Agent Name: Foo, Step Number: 9", (None, None)),
    ]
    for raw, expected in cases:
        agent, step, _ = methods.parse_prediction(raw)
        assert (agent, step) == expected, repr(raw)

    # No-op check: on a plain vendored-style output, strip_think + markdown strip
    # change nothing — the parse equals a raw vendored-regex parse.
    plain = "Agent Name: Orchestrator\n, Step Number: 5\n, Reason for Mistake: r"
    agent, step, trimmed = methods.parse_prediction(plain)
    assert trimmed == methods.trim_response(plain)
    assert (agent, step) == ("Orchestrator", 5)


# ─────────────────────────────────────────────────────────────────────────────
# Similarity-ranking parity (fake deterministic encoder through both codepaths)
# ─────────────────────────────────────────────────────────────────────────────

class FakeEncTokenizer:
    @classmethod
    def from_pretrained(cls, *a, **k):
        return cls()

    def __call__(self, texts, padding=True, truncation=True, max_length=8192,
                 return_tensors="pt"):
        import torch
        L = min(max(len(t.encode("utf-8")[:64]) for t in texts), 64)
        ids = torch.zeros(len(texts), L, dtype=torch.long)
        mask = torch.zeros(len(texts), L, dtype=torch.long)
        for i, t in enumerate(texts):
            bs = t.encode("utf-8")[:L]
            for j, b in enumerate(bs):
                ids[i, j] = b + 1
                mask[i, j] = 1
        return {"input_ids": ids, "attention_mask": mask}


class FakeEncModel:
    @classmethod
    def from_pretrained(cls, *a, **k):
        return cls()

    def eval(self):
        return self

    def to(self, device):
        return self

    def __call__(self, input_ids=None, attention_mask=None, **k):
        import torch
        D = 16
        freqs = torch.arange(1, D + 1, dtype=torch.float32,
                             device=input_ids.device) * 0.05
        emb = torch.sin(input_ids.unsqueeze(-1).float() * freqs)
        return (emb,)


def test_similarity_ranking_parity():
    """Identical neighbour rankings from our similarity.py and the vendored script."""
    gts = load_vendored("generate_trajectory_similarities.py")
    records = make_records()
    with tempfile.TemporaryDirectory() as tmp:
        traj_dir = write_trajectory_dir(Path(tmp), records)

        import transformers
        with mock.patch.object(gts, "AutoTokenizer", FakeEncTokenizer), \
             mock.patch.object(gts, "AutoModel", FakeEncModel), \
             mock.patch.object(transformers, "AutoTokenizer", FakeEncTokenizer), \
             mock.patch.object(transformers, "AutoModel", FakeEncModel), \
             contextlib.redirect_stdout(io.StringIO()):
            vendored = gts.compute_trajectory_similarities(str(traj_dir), "fake-encoder")
            ours = similarity.compute_trajectory_similarities(str(traj_dir), "fake-encoder")

    assert vendored == ours
    assert set(ours) == set(records)
    for num, neighbours in ours.items():
        assert num not in neighbours          # leave-one-out self-exclusion
        assert set(neighbours) == set(records) - {num}


# ─────────────────────────────────────────────────────────────────────────────

ALL_TESTS = [
    test_base_prompt_parity,
    test_schema_injected_prompt_parity,
    test_no_schema_fallback_prompt_parity,
    test_schema_generation_prompt_parity,
    test_response_trim_parity,
    test_schemata_file_roundtrip_parity,
    test_retrieval_parity,
    test_parse_parity_with_vendored_evaluate,
    test_reasoning_output_robustness,
    test_similarity_ranking_parity,
]

if __name__ == "__main__":
    for t in ALL_TESTS:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(ALL_TESTS)}/{len(ALL_TESTS)} parity tests passed")
