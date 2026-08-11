"""CHIEF stage 1 (offline) — precompute the retrieved exemplar block per trajectory.

CHIEF's first prompt injects a worked example retrieved from a knowledge base of
GAIA and AssistantBench tasks: embed the trajectory's *question*, search the two
committed FAISS indices under ``vendored/CHIEF/rag/``, and render the hits as the
``[RAG Example i]`` block. That lookup depends only on the question, the KB and
the encoder — not on the detector, and not on whether the answer appears in the
prompt — so it runs **once per subset**, offline, and its result is an artifact::

    artifacts/<ds>/<subset>/rag/all-MiniLM-L6-v2.json    {"<id>": "<rag_text>"}
    artifacts/<ds>/<subset>/rag/all-MiniLM-L6-v2.meta.json

Detection then reads that JSON. No FAISS search, no sentence-transformers, no
torch during the six LLM calls — the machine holding the API key needs none of
them installed, and every rerun injects byte-identical exemplars.

The rendered text is exactly what the vendored code builds, quirks included: the
search keeps ``combined_sorted[1:top_k]`` (``vendored/CHIEF/rag/rag_search.py:65``),
which drops the best hit and, at the vendored ``top_k=2``, injects exactly one
exemplar. See IMPLEMENTATION.md — it is not "fixed" here.

Usage
-----
python -m baselines.chief.ragprep \\
    --input  data/ww/hand-crafted \\
    --output artifacts/ww/hand-crafted/rag/all-MiniLM-L6-v2.json \\
    --rag-root vendored/CHIEF/rag
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from baselines.prompting.predict import load_records

from .rag import DEFAULT_RAG_ROOT, EMBED_MODEL, build_retriever
from .stages import format_rag_blocks


def rag_texts_for(retriever, records, top_k: int = 2) -> dict[str, str]:
    """Render the stage-1 exemplar block for every record, keyed by trajectory id."""
    return {r["id"]: format_rag_blocks(retriever.search(r.get("question", ""), top_k=top_k))
            for r in records}


def _kb_list(x: str) -> list[str]:
    return [k.strip() for k in str(x).split(",") if k.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description="Precompute CHIEF's stage-1 RAG exemplars.")
    p.add_argument("--input", required=True, help="Subset directory of trajectory JSONs.")
    p.add_argument("--output", required=True,
                   help="Output JSON path, e.g. "
                        "artifacts/<ds>/<subset>/rag/all-MiniLM-L6-v2.json.")
    p.add_argument("--rag-root", default=DEFAULT_RAG_ROOT,
                   help=f"Directory holding index/ and kb/ (default: {DEFAULT_RAG_ROOT}).")
    p.add_argument("--rag-kb", default="gaia,assistantbench",
                   help="Comma-separated knowledge bases to search.")
    p.add_argument("--rag-top-k", type=int, default=2,
                   help="Vendored top_k; the [1:top_k] slice makes 2 mean one exemplar.")
    p.add_argument("--embed-model", default=EMBED_MODEL,
                   help="Sentence-transformers model (HF name or local path).")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    out_path = Path(args.output)
    if out_path.exists() and not args.overwrite:
        print(f"skip (exists): {out_path}")
        return

    records = load_records(args.input)
    if not records:
        print(f"  no trajectories under {args.input}")
        return

    kbs = _kb_list(args.rag_kb)
    retriever = build_retriever(args.rag_root, kbs, embed_model=args.embed_model)
    if retriever is None:
        raise SystemExit("--rag-kb selected no knowledge base; nothing to retrieve")

    texts = rag_texts_for(retriever, records, top_k=args.rag_top_k)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(texts, indent=2, ensure_ascii=False), encoding="utf-8")
    meta_path = out_path.with_name(out_path.stem + ".meta.json")
    meta_path.write_text(json.dumps({
        "input": args.input,
        "rag_root": args.rag_root,
        "kbs": kbs,
        "top_k": args.rag_top_k,
        "embed_model": args.embed_model,
        "n_trajectories": len(texts),
        "n_empty": sum(1 for t in texts.values() if not t),
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2), encoding="utf-8")
    print(f"  wrote {out_path}  ({len(texts)} trajectories)")


if __name__ == "__main__":
    main()
