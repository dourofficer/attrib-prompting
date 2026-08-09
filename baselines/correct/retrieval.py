"""Schema retrieval — load stage-1/2 artifacts and pick top-k neighbour schemata.

Retrieval semantics are the vendored Who&When analyzer's
(``vendored/CORRECT/src/inference_whoandwhen.py::SimilarityBasedSchemaAnalyzer
.get_similarity_based_schema``): take the top-k slice of the precomputed
neighbour ranking and keep only neighbours that have a schema — **silently
fewer** than k if some are missing, ``([], [])`` for an unknown id. Self never
appears (stage 2 excludes it from every ranking). The vendored random fallback
is off by default upstream and not ported.

Deviation (storage only): schemata are read from stage-1 per-trajectory JSONs
(``.../schemagen/<id>.json``) keyed by trajectory id, instead of parsing an
``error_schemata.txt`` whose 1-based enumeration must coincide with file
numbering. The schema text itself is byte-identical.
"""
from __future__ import annotations

import json
from pathlib import Path

from .methods import strip_think


def load_schemata(schemagen_dir: str | Path) -> dict[int, str]:
    """``{trajectory id: schema text}`` from a stage-1 output directory."""
    schemata: dict[int, str] = {}
    for path in Path(schemagen_dir).glob("*.json"):
        if not path.stem.isdigit():
            continue  # _run.json etc.
        doc = json.loads(path.read_text(encoding="utf-8"))
        schema = doc.get("schema") or strip_think(doc.get("raw") or "").strip()
        if schema:
            schemata[int(path.stem)] = schema
    return schemata


def load_trajectory_similarities(path: str | Path) -> dict[int, list[int]]:
    """Vendored loader semantics: JSON string keys cast back to int."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


class SchemaAnalyzer:
    """Top-k neighbour-schema lookup over precomputed artifacts."""

    def __init__(self, schemata: dict[int, str], similarities: dict[int, list[int]]):
        self.schemata = schemata
        self.similarities = similarities

    @classmethod
    def from_paths(cls, schemagen_dir: str | Path, similarities_path: str | Path) -> "SchemaAnalyzer":
        return cls(load_schemata(schemagen_dir),
                   load_trajectory_similarities(similarities_path))

    def get_similarity_based_schema(self, file_num: int,
                                    num_schemata: int = 1) -> tuple[list[int], list[str]]:
        schema_keys: list[int] = []
        schema_contents: list[str] = []
        similar_indices = self.similarities.get(file_num, [])
        for similar_idx in similar_indices[:num_schemata]:
            if similar_idx in self.schemata:
                schema_keys.append(similar_idx)
                schema_contents.append(self.schemata[similar_idx])
        return schema_keys, schema_contents
