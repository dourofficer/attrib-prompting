"""Build ``data/correct-error-gt/`` — CORRECT-Error with task ground truth restored.

The released CORRECT-Error dataset (``yifanyu/CORRECT-Error``, and therefore
``data/correct-error/``) ships ``groundtruth``/``ground_truth`` empty on all 2,226
records, so this repo's with-GT setting — realised purely by a record carrying a
non-empty ``ground_truth`` (see ``baselines/prompting/methods.py``) — cannot be
exercised on it. The authors clearly had the answers (the paper's schema-generation
prompt, Fig. 9, interpolates ``Ground Truth: {ground truth}``); they were not released.

They are recoverable exactly. Each record's ``question_id`` has the form
``task<N>_<K>``, where **N is a 0-based row index into the source benchmark split**
and K is a repetition counter. This script re-joins on that index and copies the
answer in. No fuzzy matching, no model inference.

The join is more reliable than matching on the stored ``question`` text, which was
damaged by unquoted shell expansion in the authors' generation pipeline
(``$ABCD$`` -> ``$ ``, ``$10,000`` -> ``,000``, and literally ``$0 \\le`` ->
``/bin/sh \\le``); two ARC records even hold a Planner reasoning fragment instead of
the question, and six records have an empty question. The index recovers all of them.

The output corpus is a sibling of ``data/correct-error/``, never a rewrite of it:
same subsets, same filename stems (so ``split_data`` yields *identical* seeded
partitions and with-GT vs without-GT is an exact paired comparison), same records
byte-for-byte apart from four keys:

    ground_truth     the recovered answer  (read by this repo's loaders)
    groundtruth      the same value        (read by the vendored CORRECT generator)
    question_source  the clean benchmark question text — reference/audit only,
                     never rendered into a prompt
    gt_source        {dataset, config, split, row_index, match}

The stored ``question`` is deliberately left damaged so that the only prompt-visible
difference from ``data/correct-error/`` is the answer.

Usage:
    python scripts/build_correct_error_gt.py --dry-run   # stats only, writes nothing
    python scripts/build_correct_error_gt.py

Requires the ``data`` extra:  pip install -e ".[data]"
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "data" / "correct-error"
DST_ROOT = ROOT / "data" / "correct-error-gt"

# GAIA needs no network: the validation set is vendored, and its row order is the
# GAIA validation order that ``task<N>`` indexes (asserted at load time against
# gaia_kb.json, whose list position is likewise the index).
GAIA_PARQUET = ROOT / "vendored" / "CHIEF" / "rag" / "data" / "GAIA_dataset.parquet"
GAIA_KB = ROOT / "vendored" / "CHIEF" / "rag" / "kb" / "gaia_kb.json"

# subset -> (hf repo, config, split). Each verified by re-joining every record's
# stored question against the source row at ``task<N>``; see MIN_MATCH_RATE below.
#
# Two near-miss mirrors are deliberately NOT used, because their row order differs:
# ``dgslibisey/MuSiQue`` (puts musique task58's question at row 1176) and
# ``xanhho``/``voidful`` 2WikiMultihopQA.
HF_SOURCES = {
    "hotpot": ("hotpotqa/hotpot_qa", "distractor", "validation"),
    "wikimqa": ("scholarly-shadows-syndicate/2wikimultihopqa_with_q_gpt35", "default", "validation"),
    "musique": ("bdsaglam/musique", "default", "validation"),
    "arc": ("allenai/ai2_arc", "ARC-Challenge", "test"),
    "mmlu_pro": ("TIGER-Lab/MMLU-Pro", "default", "test"),
    "math500": ("HuggingFaceH4/MATH-500", "default", "test"),
}
SUBSETS = [*HF_SOURCES, "gaia"]

# Fraction of a subset's records whose stored question must still agree with the
# source row at ``task<N>``. Below this the index mapping is not what we think it
# is and the build aborts rather than emit plausible-looking wrong answers.
# Floors sit under the rates measured when the mapping was established (see the
# per-subset numbers printed by --dry-run); math500 and mmlu_pro are lower purely
# because shell corruption ate their `$...$` spans.
MIN_MATCH_RATE = {
    "hotpot": 0.99, "wikimqa": 0.99, "musique": 0.98,
    "arc": 0.98, "mmlu_pro": 0.85, "math500": 0.80, "gaia": 0.95,
}


# --------------------------------------------------------------------------- #
# source loading
# --------------------------------------------------------------------------- #

def load_gaia() -> list[dict]:
    """GAIA validation rows, ordered so that index == ``task<N>``."""
    import pyarrow.parquet as pq

    rows = pq.read_table(GAIA_PARQUET).to_pylist()
    kb = json.loads(GAIA_KB.read_text(encoding="utf-8"))
    if len(rows) != len(kb):
        raise SystemExit(f"gaia: {GAIA_PARQUET.name} has {len(rows)} rows, kb has {len(kb)}")
    by_question = {r["Question"]: r for r in rows}
    if len(by_question) != len(rows):
        raise SystemExit("gaia: duplicate questions in the vendored parquet; cannot order by text")
    # kb list position is the GAIA validation index; re-key the parquet by it.
    ordered = []
    for i, entry in enumerate(kb):
        row = by_question.get(entry["question"])
        if row is None:
            raise SystemExit(f"gaia: kb[{i}] question absent from {GAIA_PARQUET.name}")
        ordered.append(row)
    return ordered


def load_source(subset: str) -> list[dict]:
    if subset == "gaia":
        return load_gaia()
    from datasets import load_dataset

    repo, config, split = HF_SOURCES[subset]
    return load_dataset(repo, config, split=split).to_list()


# --------------------------------------------------------------------------- #
# answer + question extraction
# --------------------------------------------------------------------------- #

def _choice(options: list[str], index: int) -> str:
    """One multiple-choice option, rendered the way the question renders it.

    CORRECT-Error writes choices as ``A) first, B) second, ...``, so the answer is
    given in the same shape (``"C) guanine"``) rather than as a bare letter.
    """
    return f"{chr(ord('A') + index)}) {options[index]}"


def _render_mc(stem: str, labels: list[str], options: list[str]) -> str:
    """Reproduce CORRECT-Error's multiple-choice question rendering, verbatim."""
    return stem + " " + ", ".join(f"{l}) {t}" for l, t in zip(labels, options))


def _mc_parts(subset: str, row: dict) -> tuple[list[str], list[str], int]:
    """-> (option labels, option texts, index of the correct one)."""
    if subset == "arc":
        # ARC labels are "A".."E" on most rows but "1".."4" on others, and the
        # corpus renders whichever the row carries — so use them, don't assume A-D.
        labels, texts = list(row["choices"]["label"]), list(row["choices"]["text"])
        if row["answerKey"] not in labels:
            raise KeyError(f"arc: answerKey {row['answerKey']!r} not in {labels}")
        return labels, texts, labels.index(row["answerKey"])
    options = list(row["options"])
    return [chr(ord("A") + i) for i in range(len(options))], options, int(row["answer_index"])


def extract(subset: str, row: dict) -> tuple[str, str]:
    """-> (ground-truth answer, clean question text) for one source row.

    The question text is rendered in the same shape the corpus stores it (options
    appended for the multiple-choice subsets) so it is directly comparable.
    """
    if subset in ("hotpot", "wikimqa", "musique"):
        return row["answer"], row["question"]
    if subset == "math500":
        return row["answer"], row["problem"]
    if subset == "gaia":
        return row["Final answer"], row["Question"]
    if subset in ("arc", "mmlu_pro"):
        labels, texts, i = _mc_parts(subset, row)
        return f"{labels[i]}) {texts[i]}", _render_mc(row["question"], labels, texts)
    raise SystemExit(f"no extractor for subset {subset!r}")


# --------------------------------------------------------------------------- #
# corruption-tolerant question agreement
# --------------------------------------------------------------------------- #

_KEEP = re.compile(r"[^a-z0-9]")


def _fingerprint(text: str) -> str:
    """Letters+digits of the text, with ``$`` treated as a separator.

    ``$`` is dropped rather than the whole ``$...$`` span: the stored questions have
    unbalanced delimiters (that is the damage), so span-matching on them mangles
    arbitrary amounts of good prose.
    """
    return _KEEP.sub("", (text or "").replace("$", " ").lower())


def _subsequence_coverage(a: str, b: str) -> float:
    """Fraction of ``a`` consumable, in order, from ``b`` (greedy).

    Shell expansion only ever *deleted* characters from the stored question, so a
    correctly joined record leaves the stored fingerprint an exact subsequence of
    the source one and this returns 1.0 no matter how heavy the damage. Greedy is
    exact for deletion-only edits, which is the case that matters here.
    """
    it = iter(b)
    return sum(1 for ch in a if any(c == ch for c in it)) / len(a)


def agreement(stored: str, source: str) -> tuple[str, float]:
    """Classify how well a stored question matches its source row.

    Scored by the better of subsequence coverage and SequenceMatcher's block
    coverage — the former nails pure shell damage, the latter tolerates the
    occasional substitution. A genuinely different task scores far below both (the
    two ARC records holding Planner prose instead of their question land at ~0.2).
    """
    if not (stored or "").strip():
        return "question-field-empty", 1.0
    a, b = _fingerprint(stored)[:1200], _fingerprint(source)[:1200]
    if not a or not b:
        return "unscorable", 1.0
    if a == b or b.startswith(a) or a.startswith(b):
        return "exact", 1.0
    blocks = sum(bl.size for bl in SequenceMatcher(None, a, b).get_matching_blocks())
    coverage = max(_subsequence_coverage(a, b), blocks / len(a))
    if coverage >= 0.95:
        return "shell-corrupted", coverage
    if coverage >= 0.70:
        return "partial", coverage
    return "unmatched", coverage


AGREEING = {"exact", "shell-corrupted", "partial", "question-field-empty", "unscorable"}


def _is_subsequence(a: str, b: str) -> bool:
    it = iter(b)
    return all(any(c == ch for c in it) for ch in a)


_OPTION_HEAD = r"(?:^|[\s,]){label}\)\s"


def answer_option_status(stored: str, labels: list[str], texts: list[str], i: int) -> str:
    """How the correct option survives in the stored question (audit only).

    ``verbatim``       the benchmark's correct option appears as-is
    ``shell-damaged``  present but with characters deleted (``$1,400.00`` -> ``,400.00``)
    ``perturbed``      the corpus carries *different* text under the answer's label —
                       real in a handful of records (e.g. ARC 117's correct option reads
                       "made of one or more crystals" where the benchmark says "minerals")
    ``absent``         no option list found under that label
    """
    if not (stored or "").strip():
        return "question-field-empty"
    if f"{labels[i]}) {texts[i]}" in stored:
        return "verbatim"
    spans = []
    for label in labels:
        m = re.search(_OPTION_HEAD.format(label=re.escape(label)), stored)
        if m:
            spans.append((m.start(), m.end(), label))
    spans.sort()
    for n, (_, end, label) in enumerate(spans):
        if label != labels[i]:
            continue
        stop = spans[n + 1][0] if n + 1 < len(spans) else len(stored)
        got = _fingerprint(stored[end:stop].rstrip(", "))
        want = _fingerprint(texts[i])
        # Shell expansion only deleted characters, so genuine damage leaves the
        # stored text a strictly shorter subsequence of the benchmark's.
        return "shell-damaged" if len(got) < len(want) and _is_subsequence(got, want) else "perturbed"
    return "absent"


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #

def task_index(question_id: str) -> int:
    m = re.fullmatch(r"task(\d+)_(\d+)", question_id or "")
    if not m:
        raise SystemExit(f"unparseable question_id {question_id!r}")
    return int(m.group(1))


def build_subset(subset: str, dry_run: bool) -> dict:
    src_files = sorted(
        (SRC_ROOT / subset).glob("*.json"),
        key=lambda p: int("".join(filter(str.isdigit, p.name)) or 0),
    )
    if not src_files:
        raise SystemExit(f"{subset}: no records under {SRC_ROOT / subset}")
    source = load_source(subset)
    repo, config, split = ("vendored/CHIEF GAIA_dataset.parquet", "-", "validation") \
        if subset == "gaia" else HF_SOURCES[subset]

    verdicts: dict[str, int] = {}
    options: dict[str, int] = {}
    notable: list[dict] = []
    records: list[tuple[Path, dict]] = []

    for path in src_files:
        rec = json.loads(path.read_text(encoding="utf-8"))
        idx = task_index(rec.get("question_id"))
        if not 0 <= idx < len(source):
            raise SystemExit(
                f"{subset}/{path.name}: task index {idx} out of range for "
                f"{repo}[{config}/{split}] (n={len(source)})"
            )
        answer, clean_q = extract(subset, source[idx])
        verdict, ratio = agreement(rec.get("question", ""), clean_q)
        verdicts[verdict] = verdicts.get(verdict, 0) + 1
        option_status = None
        if subset in ("arc", "mmlu_pro"):
            labels, texts, i = _mc_parts(subset, source[idx])
            option_status = answer_option_status(rec.get("question", ""), labels, texts, i)
            options[option_status] = options.get(option_status, 0) + 1
        if verdict not in ("exact", "shell-corrupted") or option_status == "perturbed":
            notable.append({
                "file": f"{subset}/{path.name}", "question_id": rec.get("question_id"),
                "row_index": idx, "match": verdict, "ratio": round(ratio, 3),
                "answer_option": option_status,
                "stored_question": (rec.get("question") or "")[:100],
                "source_question": clean_q[:100], "ground_truth": answer,
            })
        if not str(answer or "").strip():
            raise SystemExit(f"{subset}/{path.name}: empty answer at row {idx} of {repo}")

        out = dict(rec)  # preserves key order; history/labels copied verbatim
        out["groundtruth"] = answer
        out["ground_truth"] = answer
        out["question_source"] = clean_q
        out["gt_source"] = {
            "dataset": repo, "config": config, "split": split,
            "row_index": idx, "match": verdict,
        }
        if option_status is not None:
            out["gt_source"]["answer_option"] = option_status
        records.append((DST_ROOT / subset / path.name, out))

    n = len(records)
    agreed = sum(c for v, c in verdicts.items() if v in AGREEING)
    rate = agreed / n
    floor = MIN_MATCH_RATE[subset]
    print(f"  {subset:9s} n={n:<4d} match={rate:6.1%} (floor {floor:.0%})  {verdicts}"
          + (f"\n{'':13s}answer option in stored question: {options}" if options else ""))
    if rate < floor:
        raise SystemExit(
            f"{subset}: only {rate:.1%} of records agree with {repo}[{config}/{split}] "
            f"at task<N> — index mapping is wrong, refusing to write"
        )

    if not dry_run:
        (DST_ROOT / subset).mkdir(parents=True, exist_ok=True)
        for dst, doc in records:
            tmp = dst.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp.replace(dst)

    return {
        "subset": subset, "n": n, "source": {"dataset": repo, "config": config, "split": split,
                                             "rows": len(source)},
        "match_rate": round(rate, 4), "verdicts": verdicts,
        "answer_option": options or None, "notable": notable,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true", help="report stats, write nothing")
    ap.add_argument("--subset", action="append", choices=SUBSETS,
                    help="restrict to one subset (repeatable); default: all")
    args = ap.parse_args()
    subsets = args.subset or SUBSETS

    print(f"{'dry run: ' if args.dry_run else ''}data/correct-error -> data/correct-error-gt")
    stats = [build_subset(s, args.dry_run) for s in subsets]

    total = sum(s["n"] for s in stats)
    notable = [n for s in stats for n in s["notable"]]
    print(f"\n{total} records; {len(notable)} needing a closer look:")
    for item in notable:
        print(f"  {item['file']:22s} {item['question_id']:12s} row={item['row_index']:<4d} "
              f"{item['match']} ({item['ratio']})"
              + (f" answer-option={item['answer_option']}" if item["answer_option"] else ""))
        print(f"      stored: {item['stored_question']!r}")
        print(f"      source: {item['source_question']!r}")
        print(f"      gt    : {item['ground_truth']!r}")

    if args.dry_run:
        print("\n(dry run — nothing written)")
        return 0

    DST_ROOT.mkdir(parents=True, exist_ok=True)
    (DST_ROOT / "_provenance.json").write_text(
        json.dumps({
            "built_by": "scripts/build_correct_error_gt.py",
            "source_corpus": "data/correct-error",
            "join": "question_id 'task<N>_<K>' -> 0-based row index N of the source split",
            "added_keys": ["ground_truth", "groundtruth", "question_source", "gt_source"],
            "note": "question/history/labels are copied verbatim; the damaged question text "
                    "is left as-is so the only prompt-visible delta is the answer",
            "subsets": stats,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nwrote {total} records + _provenance.json under {DST_ROOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
