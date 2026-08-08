"""Schema cache I/O and similarity-based retrieval for the CORRECT baseline.

Ports, verbatim in behaviour:

* the ``error_schemata.txt`` block writer of
  ``baselines/CORRECT/src/error_schema_generator.py`` (``process_single_dataset``),
* the block parser ``load_error_schemata`` of
  ``baselines/CORRECT/src/inference_whoandwhen.py`` (accepts the legacy
  "Template" marker too — a superset of ``inference_correct_error.py``'s parser),
* ``load_trajectory_similarities`` (shared by both vendored scripts), and
* the two retrieval behaviours as one :class:`SchemaAnalyzer`:
  ``scan_until_filled=False`` inspects only the top-k neighbours
  (``SimilarityBasedSchemaAnalyzer``, Who&When script);
  ``scan_until_filled=True`` keeps scanning until k schemata are found, capped
  after checking ``5*k`` neighbours (``DatasetSimilaritySchemaAnalyzer``,
  CORRECT-Error script).
  With a complete schema cache — our case, one schema per trajectory — the two
  are identical.

One deliberate fix over the vendored writer: schema blocks are keyed by the
trajectory's numeric filename instead of a 1-based enumeration position. On this
repo's data (contiguous ``1.json…N.json``, all valid) the two are byte-identical;
keying by filename simply removes the latent misalignment the enumeration would
cause if a file were skipped, since retrieval looks schemata up by file number.
"""
from __future__ import annotations

import json
import random


def write_schemata_file(schemata: dict[int, str], path) -> None:
    """Write schema blocks in the vendored ``error_schemata.txt`` format."""
    with open(path, "w", encoding="utf-8") as f:
        for file_num in sorted(schemata):
            f.write(f"=== Schema for Error Log {file_num} ===\n")
            f.write("Generated Schema:\n")
            f.write(schemata[file_num])
            f.write("\n\n" + "=" * 50 + "\n\n")


def load_error_schemata(schemata_file) -> dict[int, str]:
    """Parse a vendored-format ``error_schemata.txt`` into ``{log_num: schema}``."""
    schemata: dict[int, str] = {}
    current_log_num: int | None = None
    current_schema = ""

    with open(schemata_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i]
        # Start of a new schema section. Accept the legacy "Template" marker so
        # vendored-generated caches can be reproduced without conversion.
        is_schema_marker = line.startswith("=== Schema for Error Log ")
        is_template_marker = line.startswith("=== Template for Error Log ")
        if is_schema_marker or is_template_marker:
            # Save previous schema if there was one (first occurrence wins).
            if current_log_num is not None and current_schema:
                if current_log_num not in schemata:
                    schemata[current_log_num] = current_schema.strip()

            marker_text = line.strip().split("===")[1].strip()
            marker_text = marker_text.replace("Schema for Error Log ", "")
            marker_text = marker_text.replace("Template for Error Log ", "")
            current_log_num = int(marker_text)

            # Start collecting schema content from the next line.
            current_schema = ""
            i += 1

            # Collect all lines until the separator.
            while i < len(lines) and not lines[i].startswith("=" * 50):
                # Skip generated-content header lines if present.
                if lines[i].strip() not in {"Generated Schema:", "Generated Template:"}:
                    current_schema += lines[i]
                i += 1

            # When we reach the separator, save this schema.
            if i < len(lines) and lines[i].startswith("=" * 50):
                if current_log_num not in schemata:
                    schemata[current_log_num] = current_schema.strip()
        else:
            i += 1

    # Save the last schema if we ended without a separator.
    if current_log_num is not None and current_schema and current_log_num not in schemata:
        schemata[current_log_num] = current_schema.strip()

    return schemata


def load_trajectory_similarities(similarities_file) -> dict[int, list[int]]:
    """Load the precomputed ranked-neighbour map, with int keys."""
    with open(similarities_file, "r", encoding="utf-8") as f:
        similarities = json.load(f)
    return {int(key): value for key, value in similarities.items()}


class SchemaAnalyzer:
    """Similarity-based schema retrieval — both vendored behaviours in one class."""

    def __init__(
        self,
        schemata: dict[int, str],
        similarities: dict[int, list[int]],
        *,
        scan_until_filled: bool = False,
        use_random_fallback: bool = False,
    ) -> None:
        self.schemata = schemata
        self.similarities = similarities
        self.scan_until_filled = scan_until_filled
        self.use_random_fallback = use_random_fallback

        self.schema_list = list(schemata.values()) if schemata else []
        self.schema_keys = list(schemata.keys()) if schemata else []

    def get_similarity_based_schema(
        self, file_num: int, num_schemata: int = 1
    ) -> tuple[list[int], list[str]]:
        """Return ``(schema_keys, schema_contents)`` for the query trajectory.

        The similarity lists exclude the trajectory itself by construction
        (leave-one-out), so a trajectory never receives its own schema.
        """
        schema_keys: list[int] = []
        schema_contents: list[str] = []

        if file_num in self.similarities:
            similar_indices = self.similarities[file_num]
            if similar_indices:
                if self.scan_until_filled:
                    # CORRECT-Error script: keep scanning until k schemata are
                    # found; stop after checking many indices to avoid long search.
                    checked_count = 0
                    for similar_idx in similar_indices:
                        if similar_idx in self.schemata:
                            schema_keys.append(similar_idx)
                            schema_contents.append(self.schemata[similar_idx])
                            if len(schema_contents) >= num_schemata:
                                break
                        checked_count += 1
                        if checked_count > num_schemata * 5:
                            break
                else:
                    # Who&When script: only inspect the top-k neighbours.
                    for similar_idx in similar_indices[:num_schemata]:
                        if similar_idx in self.schemata:
                            schema_keys.append(similar_idx)
                            schema_contents.append(self.schemata[similar_idx])

        # Fallback to random if enabled and we don't have enough schemata
        # (vendored option; off by default and unused in our configs).
        if self.use_random_fallback and len(schema_contents) < num_schemata and self.schema_list:
            num_random_needed = num_schemata - len(schema_contents)
            available_indices = [i for i in range(len(self.schema_list))
                                 if self.schema_keys[i] not in schema_keys]
            for _ in range(min(num_random_needed, len(available_indices))):
                if available_indices:
                    idx = random.choice(available_indices)
                    available_indices.remove(idx)
                    schema_keys.append(self.schema_keys[idx])
                    schema_contents.append(self.schema_list[idx])

        return schema_keys, schema_contents
