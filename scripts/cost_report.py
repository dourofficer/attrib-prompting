#!/usr/bin/env python
"""Inference cost of every prompt-based baseline on Who&When, replayed offline.

The stored predictions keep every judge response in ``calls`` but never the prompt,
the token counts, or the latency. This script rebuilds each prompt exactly, without
a model or an API key, by driving the method's own generator program with the stored
responses: the program yields the same prompts it yielded at run time, because a
prompt depends only on the trajectory, the artifacts, and the earlier responses. The
prompts are then tokenized with the judge's tokenizer (the Qwen checkpoint's chat
template for the open judge, ``tiktoken``'s o200k encoding for GPT-4o).

Per (setting, judge, subset, method) it reports, averaged over trajectories:

    calls_per_traj        judge requests
    prompt_tok_per_traj   prompt tokens the judge read (after the run's truncation cap)
    out_tok_per_traj      tokens the judge generated
    n_traj / n_costed     trajectories in the cell / trajectories that could be replayed

Cells imported from a legacy run carry no ``calls``; they are reported with
``n_costed = 0`` (All-at-Once is the exception: its single prompt needs no response,
so its prompt tokens are still counted). Every replay is checked against the stored
prediction, and a mismatch is counted in ``n_mismatch`` — a non-zero value means the
reconstruction is not byte-faithful for that cell and its numbers should be read as
approximate.

    python scripts/cost_report.py                      # reports/cost_ww.tsv
    python scripts/cost_report.py --judge qwen3.5-9b --method chief --limit 5 -v
"""
from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from baselines.prompting.predict import load_records                      # noqa: E402
from baselines.prompting.methods import METHODS as PROMPT_METHODS         # noqa: E402
from baselines.correct.methods import correct_program                     # noqa: E402
from baselines.correct.retrieval import SchemaAnalyzer                    # noqa: E402
from baselines.chief.methods import chief_program                         # noqa: E402
from baselines.chief.predict import _load_rag_texts                       # noqa: E402
from baselines.errorprobe.paper.program import errorprobe_paper_program   # noqa: E402

ROOTS = [("outputs-nogt", False), ("outputs", True)]
JUDGES = {"qwen3.5-9b": "qwen", "gpt-4o": "openai"}
SUBSETS = ["algorithm-generated", "hand-crafted"]
METHODS = ["all_at_once", "step_by_step", "binary_search", "correct", "chief",
           "errorprobe_paper"]
QWEN_PATH = REPO.parent / "hub" / "Qwen" / "Qwen3.5-9B"

COLUMNS = ["with_gt", "judge", "subset", "method", "n_traj", "n_costed", "n_mismatch",
           "calls_per_traj", "prompt_tok_per_traj", "out_tok_per_traj",
           "prompt_tok_per_call", "truncate_prompt_tokens", "step_mode"]


# ── tokenizers ───────────────────────────────────────────────────────────────
class QwenCounter:
    """Token count of a chat as vLLM sees it: the checkpoint's template, thinking off."""

    def __init__(self, path: Path) -> None:
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(str(path), trust_remote_code=True)

    def prompt(self, messages: list[dict]) -> int:
        text = self.tok.apply_chat_template(messages, tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False)
        return len(self.tok(text, add_special_tokens=False)["input_ids"])

    def output(self, text: str) -> int:
        return len(self.tok(text or "", add_special_tokens=False)["input_ids"])


class OpenAICounter:
    """The OpenAI cookbook's chat-completions estimate: 3 tokens per message plus 3."""

    def __init__(self, model: str = "gpt-4o") -> None:
        import tiktoken
        self.enc = tiktoken.encoding_for_model(model)

    def prompt(self, messages: list[dict]) -> int:
        n = 3
        for m in messages:
            n += 3 + len(self.enc.encode(m.get("role", ""))) + len(self.enc.encode(m.get("content", "") or ""))
        return n

    def output(self, text: str) -> int:
        return len(self.enc.encode(text or ""))


# ── replay ───────────────────────────────────────────────────────────────────
class ReplayMismatch(Exception):
    pass


def replay(gen, responses: list[str]) -> tuple[list[list[dict]], dict]:
    """Drive a program with the stored responses; return every prompt it yielded."""
    prompts_all: list[list[dict]] = []
    pos = 0
    try:
        prompts = next(gen)
        while True:
            n = len(prompts)
            chunk = responses[pos:pos + n]
            if len(chunk) < n:
                raise ReplayMismatch(f"program wants {n} responses at call {pos}, "
                                     f"only {len(responses) - pos} stored")
            prompts_all.extend(prompts)
            pos += n
            prompts = gen.send(chunk)
    except StopIteration as si:
        result = si.value
    if pos != len(responses):
        raise ReplayMismatch(f"program consumed {pos} of {len(responses)} stored responses")
    return prompts_all, result


@lru_cache(maxsize=None)
def analyzer_for(schemata_dir: str, similarities: str, subset: str) -> SchemaAnalyzer:
    """The CORRECT retrieval artifacts; legacy run snapshots may name a path that
    has since moved into artifacts/, so fall back to the committed location."""
    sd, sp = Path(schemata_dir), Path(similarities)
    if not sd.is_dir():
        sd = REPO / "artifacts" / "ww" / subset / "schemagen" / "gpt-4o"
    if not sp.is_file():
        sp = REPO / "artifacts" / "ww" / subset / "similarities" / "bge-m3.json"
    return SchemaAnalyzer.from_paths(sd, sp)


@lru_cache(maxsize=None)
def rag_for(path: str) -> dict[str, str]:
    return _load_rag_texts(path)


def make_program(method: str, record: dict, doc: dict, run: dict, subset: str, with_gt: bool):
    gt = bool(doc.get("gt_in_prompt", run.get("gt_in_prompt", with_gt)))
    if method in PROMPT_METHODS:
        return PROMPT_METHODS[method](record, step_mode=run.get("step_mode") or "batch",
                                      include_gt=gt)
    if method == "correct":
        an = analyzer_for(run["schemata_dir"], run["similarities"], subset)
        return correct_program(record, analyzer=an, num_schemata=int(doc["num_schemata"]),
                               include_gt=gt)
    if method == "chief":
        rag = rag_for(run["rag_texts"]) if run.get("rag_texts") else {}
        return chief_program(record, rag_text=rag.get(record["id"]) if doc.get("rag_in_prompt") else None,
                             include_gt=gt, include_step_hint=bool(run.get("step_hint_in_prompt", True)))
    if method == "errorprobe_paper":
        return errorprobe_paper_program(record, include_gt=gt, **doc["paper_params"])
    raise ValueError(method)


def same_prediction(method: str, doc: dict, result: dict) -> bool:
    if method == "correct" and list(result.get("schema_cases", [])) != list(doc.get("schema_cases", [])):
        return False
    return (result.get("predicted_step") == doc.get("predicted_step")
            and result.get("predicted_agent") == doc.get("predicted_agent"))


# ── one cell ─────────────────────────────────────────────────────────────────
def cost_cell(root: str, with_gt: bool, judge: str, subset: str, method: str,
              counter, records: dict[str, dict], limit: int | None, verbose: bool) -> dict | None:
    cell = REPO / root / "ww" / subset / judge / method
    if not cell.is_dir():
        return None
    files = sorted((p for p in cell.glob("*.json") if p.stem.isdigit()), key=lambda p: int(p.stem))
    if limit:
        files = files[:limit]
    run = json.loads((cell / "_run.json").read_text()) if (cell / "_run.json").exists() else {}
    cap = (run.get("request_params") or {}).get("truncate_prompt_tokens")

    n_traj = n_costed = n_out = n_mismatch = 0
    calls = ptok = otok = 0.0
    for p in files:
        doc = json.loads(p.read_text())
        n_traj += 1
        responses = [c.get("response") or "" for c in doc.get("calls") or []]
        record = records[doc["id"]]
        out_known = bool(responses)
        if not responses:
            if method != "all_at_once":
                continue                          # legacy import: the path is unknown
            responses = [""]                      # one prompt, no response needed
        try:
            prompts, result = replay(make_program(method, record, doc, run, subset, with_gt), responses)
        except ReplayMismatch as exc:
            n_mismatch += 1
            if verbose:
                print(f"    [mismatch] {cell.name}/{p.name}: {exc}")
            continue
        if out_known and not same_prediction(method, doc, result):
            n_mismatch += 1
            if verbose:
                print(f"    [mismatch] {cell.name}/{p.name}: replay predicts "
                      f"{result.get('predicted_step')} vs stored {doc.get('predicted_step')}")
        n_costed += 1
        calls += len(prompts)
        pt = [counter.prompt(m) for m in prompts]
        ptok += sum(min(t, cap) if cap else t for t in pt)
        if out_known:
            n_out += 1
            otok += sum(counter.output(r) for r in responses)
    if n_traj == 0:
        return None
    step_mode = (run.get("step_mode") or "") if method == "step_by_step" else ""
    row = {
        "with_gt": with_gt, "judge": judge, "subset": subset, "method": method,
        "n_traj": n_traj, "n_costed": n_costed, "n_mismatch": n_mismatch,
        "calls_per_traj": round(calls / n_costed, 3) if n_costed else "",
        "prompt_tok_per_traj": round(ptok / n_costed, 1) if n_costed else "",
        "out_tok_per_traj": round(otok / n_out, 1) if n_out else "",
        "prompt_tok_per_call": round(ptok / calls, 1) if calls else "",
        "truncate_prompt_tokens": cap if cap is not None else "",
        "step_mode": step_mode,
    }
    print(f"  {root:12s} {judge:11s} {subset:19s} {method:16s} n={n_traj:3d} costed={n_costed:3d}"
          f" mismatch={n_mismatch:2d} calls={row['calls_per_traj']!s:>6} prompt={row['prompt_tok_per_traj']!s:>9}"
          f" out={row['out_tok_per_traj']!s:>7}")
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judge", action="append", choices=list(JUDGES), help="default: both")
    ap.add_argument("--method", action="append", choices=METHODS, help="default: all six")
    ap.add_argument("--subset", action="append", choices=SUBSETS)
    ap.add_argument("--limit", type=int, default=None, help="first N trajectories per cell (smoke test)")
    ap.add_argument("--out", default=str(REPO / "reports" / "cost_ww.tsv"))
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    judges = args.judge or list(JUDGES)
    methods = args.method or METHODS
    subsets = args.subset or SUBSETS
    counters = {}
    for j in judges:
        counters[j] = QwenCounter(QWEN_PATH) if JUDGES[j] == "qwen" else OpenAICounter(j)
    records = {s: {r["id"]: r for r in load_records(str(REPO / "data" / "ww" / s))} for s in subsets}

    rows = []
    for root, with_gt in ROOTS:
        for j in judges:
            for s in subsets:
                for m in methods:
                    row = cost_cell(root, with_gt, j, s, m, counters[j], records[s], args.limit, args.verbose)
                    if row:
                        rows.append(row)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write("\t".join(COLUMNS) + "\n")
        for r in rows:
            f.write("\t".join(str(r[c]) for c in COLUMNS) + "\n")
    print(f"  wrote {out}  ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
