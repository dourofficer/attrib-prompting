"""Config-driven, Linux-safe RAG retriever for CHIEF's stage 1.

The vendored ``vendored/CHIEF/rag/rag_search.RAGRetriever`` (a) hardcodes Windows
backslash paths (``index\\gaia.index``), (b) is instantiated at *import* time, and
(c) always searches *both* the GAIA and AssistantBench indices. Here we wrap the
same FAISS + sentence-transformers logic but:

  * resolve index/kb paths under a configurable ``rag_root`` with forward slashes,
  * build lazily and only when retrieval actually runs,
  * let the caller pick which KB(s) to load (``gaia`` / ``assistantbench``).

Ranking is otherwise untouched — including the ``combined_sorted[1:top_k]`` slice
that drops the top hit — so the injected exemplars stay byte-identical to the
vendored ones.

Only :mod:`baselines.chief.ragprep` imports this module: retrieval is an offline
stage whose output is committed under ``artifacts/``. ``faiss`` and
``sentence_transformers`` are therefore optional dependencies (``pip install
-e ".[rag]"``), imported inside the constructor so that detection — and the whole
test suite — runs without them.
"""
from __future__ import annotations

from pathlib import Path

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_RAG_ROOT = "vendored/CHIEF/rag"

# Map KB shorthand -> (index filename, kb filename, source tag used in step-1 blocks).
_KB_FILES = {
    "gaia": ("gaia.index", "gaia_kb.json", "GAIA"),
    "assistantbench": ("assistantbench.index", "assistantbench_kb.json", "AssistantBench"),
}


class ChiefRetriever:
    def __init__(self, rag_root=DEFAULT_RAG_ROOT, kbs=("gaia", "assistantbench"),
                 embed_model=EMBED_MODEL):
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
            index_path = root / "index" / idx_name
            kb_path = root / "kb" / kb_name
            for p in (index_path, kb_path):
                if not p.is_file():
                    raise SystemExit(
                        f"missing RAG artifact {p} — point --rag-root at the vendored "
                        f"knowledge base (default: {DEFAULT_RAG_ROOT})")
            self.indices[kb] = faiss.read_index(str(index_path))
            with open(kb_path, "r", encoding="utf-8") as f:
                self.records[kb] = json.load(f)
            self.sources[kb] = source

    def _encode(self, text: str):
        vec = self.model.encode([text], convert_to_numpy=True)
        self._faiss.normalize_L2(vec)
        return vec

    def search(self, query: str, top_k: int = 2):
        """Return retrieved records, mirroring the vendored ``RAGRetriever.search``.

        Each hit carries the ``source``/``question``/``steps``/``text`` fields that
        ``stages.format_rag_blocks`` expects. With both KBs loaded the result is
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


def build_retriever(rag_root=DEFAULT_RAG_ROOT, kbs=("gaia", "assistantbench"),
                    embed_model=EMBED_MODEL):
    """Construct a retriever, or ``None`` if no knowledge base is selected."""
    if not kbs:
        return None
    return ChiefRetriever(rag_root, kbs=list(kbs), embed_model=embed_model)
