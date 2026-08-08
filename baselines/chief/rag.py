"""Config-driven, Linux-safe RAG retriever for CHIEF's stage 1.

The vendored ``baselines/CHIEF/rag/rag_search.RAGRetriever`` (a) hardcodes Windows
backslash paths (``index\\gaia.index``), (b) is instantiated at *import* time, and
(c) always searches *both* the GAIA and AssistantBench indices. Here we wrap the
same FAISS + sentence-transformers logic but:

  * resolve index/kb paths under a configurable ``rag_root`` with forward slashes,
  * build lazily and only when RAG is enabled,
  * let the caller pick which KB(s) to load (``gaia`` / ``assistantbench``).

When both KBs are selected the search reproduces the vendored behaviour exactly
(including the ``combined_sorted[1:top_k]`` slice), so CHIEF on Who&When stays
faithful. Selecting a single KB (e.g. reuse GAIA off-domain for CORRECT-Error /
TraceElephant) searches just that index.
"""
from __future__ import annotations

from pathlib import Path

_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Map KB shorthand -> (index filename, kb filename, source tag used in step-1 blocks).
_KB_FILES = {
    "gaia": ("gaia.index", "gaia_kb.json", "GAIA"),
    "assistantbench": ("assistantbench.index", "assistantbench_kb.json", "AssistantBench"),
}


class ChiefRetriever:
    def __init__(self, rag_root, kbs=("gaia", "assistantbench"), embed_model=_EMBED_MODEL):
        import faiss
        import json
        from sentence_transformers import SentenceTransformer

        self._faiss = faiss
        root = Path(rag_root)
        self.kbs = [k for k in kbs if k in _KB_FILES]
        if not self.kbs:
            raise ValueError(f"No valid RAG KBs in {kbs!r}; choose from {list(_KB_FILES)}")

        self.model = SentenceTransformer(embed_model)
        self.indices = {}
        self.records = {}
        self.sources = {}
        for kb in self.kbs:
            idx_name, kb_name, source = _KB_FILES[kb]
            self.indices[kb] = faiss.read_index(str(root / "index" / idx_name))
            with open(root / "kb" / kb_name, "r", encoding="utf-8") as f:
                self.records[kb] = json.load(f)
            self.sources[kb] = source

    def _encode(self, text: str):
        vec = self.model.encode([text], convert_to_numpy=True)
        self._faiss.normalize_L2(vec)
        return vec

    def search(self, query: str, top_k: int = 2):
        """Return retrieved records, mirroring the vendored ``RAGRetriever.search``.

        Each hit carries the ``source``/``question``/``steps``/``text`` fields that
        ``stages.format_rag_blocks`` expects. When both KBs are loaded the result is
        identical to the vendored combine-sort-``[1:top_k]`` behaviour.
        """
        query_vec = self._encode(query)
        combined = []
        for kb in self.kbs:
            source = self.sources[kb]
            D, I = self.indices[kb].search(query_vec, top_k)
            recs = self.records[kb]
            for i in range(len(I[0])):
                rec = recs[I[0][i]]
                hit = {"source": source, "score": float(D[0][i])}
                if source == "GAIA":
                    hit["question"] = rec["question"]
                    hit["steps"] = rec["steps"]
                else:
                    hit["text"] = rec["text"]
                combined.append(hit)

        combined_sorted = sorted(combined, key=lambda x: x["score"], reverse=True)
        return combined_sorted[1:top_k]


def build_retriever(rag_root, kbs):
    """Construct a retriever, or ``None`` if RAG is disabled (empty ``kbs``)."""
    if not kbs:
        return None
    return ChiefRetriever(rag_root, kbs=list(kbs))


def rag_texts_for(retriever, records, top_k: int = 2):
    """Precompute the stage-1 ``rag_text`` block for every record.

    Returns a list aligned to ``records``: a formatted string when the retriever is
    present, else ``None`` (stage-1 omits the retrieved-example section).
    """
    from .stages import format_rag_blocks

    if retriever is None:
        return [None] * len(records)
    return [format_rag_blocks(retriever.search(r.get("question", ""), top_k=top_k))
            for r in records]
