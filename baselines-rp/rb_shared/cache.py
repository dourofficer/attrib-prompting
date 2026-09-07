"""An on-disk cache for the tensors an extraction stage produces.

Extraction is the expensive part of a representation-based baseline — one
forward pass of a multi-billion-parameter model per trajectory — and nothing
about it depends on the model that consumes the result. So it is cached, and
the cache is the resume ledger: a file that exists is work already done.

The rules match ``baselines/shared/runner.py``'s ``OutputWriter`` — write to a
temporary name, rename into place, so a crash mid-write can never leave a
half-file that a later run would trust. Two things differ, both because this
cache holds tensors rather than predictions: ids may be hexadecimal (the
training corpus names its files that way, and ``OutputWriter.done_ids`` only
counts digits), and entries are ``torch.save`` payloads rather than JSON.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Iterable, Optional

import torch


def safe_id(trajectory_id: str) -> str:
    """The vendored filename rule (``extract_states.py:143``)."""
    return str(trajectory_id).replace("/", "_").replace(".json", "")


class StateCache:
    """One directory of cached ``.pt`` payloads, keyed by trajectory id."""

    def __init__(self, directory: str | Path, overwrite: bool = False):
        self.dir = Path(directory)
        if overwrite and self.dir.exists():
            shutil.rmtree(self.dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        for stale in self.dir.glob("*.pt.tmp"):
            stale.unlink()

    def path(self, trajectory_id: str) -> Path:
        return self.dir / f"{safe_id(trajectory_id)}.pt"

    def has(self, trajectory_id: str) -> bool:
        return self.path(trajectory_id).exists()

    def missing(self, trajectory_ids: Iterable[str]) -> list[str]:
        return [tid for tid in trajectory_ids if not self.has(tid)]

    def save(self, trajectory_id: str, payload: dict) -> Path:
        target = self.path(trajectory_id)
        tmp = target.with_suffix(".pt.tmp")
        torch.save(payload, tmp)
        os.replace(tmp, target)
        return target

    def load(self, trajectory_id: str) -> Optional[dict]:
        target = self.path(trajectory_id)
        if not target.exists():
            return None
        return torch.load(target, map_location="cpu", weights_only=False)

    def write_manifest(self, manifest: dict) -> None:
        """Record what produced these tensors, so a stale cache is detectable."""
        target = self.dir / "_manifest.json"
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str))
        os.replace(tmp, target)

    def read_manifest(self) -> Optional[dict]:
        target = self.dir / "_manifest.json"
        if not target.exists():
            return None
        return json.loads(target.read_text())

    def __len__(self) -> int:
        return len(list(self.dir.glob("*.pt")))
