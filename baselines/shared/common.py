"""Shared helpers for the baseline sub-packages.

Every function here is vendored verbatim from its upstream source so behaviour
stays bit-identical; do not "improve" any of them:

- ``_get_sorted_json_files`` / ``_load_json_data`` — from the vendored Who&When
  baseline (``vendored/Agents_Failure_Attribution/Automated_FA/Lib/local_model.py``).
  Note the digit-extraction sort key: files sort by the integer formed from all
  digits in the name (identical to ``int(stem)`` for this repo's ``<N>.json``
  corpus files, but kept verbatim).
- ``split_data`` — the attribscope project's seeded shuffle-and-cut. The
  seed→partition mapping is the experiment's identity: the per-seed val/test
  splits every reported number is computed on are exactly the partitions this
  function produces over the corpus file list. Changing the copy/seed/shuffle
  /slice sequence would silently change every split.
- ``standardize_role`` — the attribscope agent-name normalization used by the
  agent@1 metric (all "…orchestrator…" variants collapse to "Orchestrator").
"""
from __future__ import annotations

import json
import os
import random


def _get_sorted_json_files(directory_path):
    try:
        files = [f for f in os.listdir(directory_path) if f.endswith('.json')]
        return sorted(files, key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))
    except FileNotFoundError:
        print(f"Error: Directory not found at {directory_path}")
        return []
    except Exception as e:
        print(f"Error reading or sorting files in {directory_path}: {e}")
        return []


def _load_json_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}")
        return None
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None


def split_data(data: list, ratio: float, seed: int) -> tuple[list, list]:
    """Shuffle a copy under ``seed``, cut at ``int(len*ratio)``."""
    data = data.copy()
    random.seed(seed)
    random.shuffle(data)
    i = int(len(data) * ratio)
    return data[:i], data[i:]


def standardize_role(role: str) -> str:
    if "orchestrator" in role.lower():
        return "Orchestrator"
    return role


def nogt_root(path: str) -> str:
    """Map a with-GT output root to its without-GT sibling.

    Two layouts are recognized, both keeping the inner path identical so the
    two GT settings never collide:

    - the prompting-based tree — ``outputs/<...>`` ↔ ``outputs-nogt/<...>``;
    - a family tree that spells the setting out, which the
      representation-based baselines use — ``outputs-rb-gt/<...>`` ↔
      ``outputs-rb-nogt/<...>``.

    Any other root raises. A root this function cannot map is a layout it does
    not know, and guessing would file results under the wrong tree.
    """
    path = str(path)
    head, sep, rest = path.partition("/")
    if head == "outputs":
        return "outputs-nogt" + sep + rest
    if head.startswith("outputs-") and head.endswith("-gt"):
        return head[: -len("-gt")] + "-nogt" + sep + rest
    raise ValueError(
        f"cannot derive the without-GT root for {path!r}: expected a path under "
        "'outputs/', or a root named 'outputs-<family>-gt' (set the root "
        "explicitly if you use a custom layout)"
    )


def artifacts_root(path: str) -> str:
    """Map an output root to the sibling root for offline stage artifacts.

    Convention: ``outputs/<...>`` (and its ``outputs-nogt/<...>`` mirror) →
    ``artifacts/<...>``. ``outputs/`` holds predictions only; precomputed
    inputs to a run — CORRECT's schemata and trajectory similarities — live in
    ``artifacts/`` instead. Both GT settings map to the same artifacts root:
    these artifacts are GT-independent, so there is no ``-nogt`` mirror.
    """
    path = str(path)
    for prefix in ("outputs-nogt", "outputs"):
        if path == prefix or path.startswith(prefix + "/"):
            return "artifacts" + path[len(prefix):]
    raise ValueError(
        f"cannot derive the artifacts root for {path!r}: expected a path under "
        "'outputs/' (set `artifacts_root` explicitly if you use a custom layout)"
    )
