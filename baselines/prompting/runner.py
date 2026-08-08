"""Drivers that execute method programs, plus the per-trajectory output writer.

Two drivers over the same generator programs (:mod:`.methods`):

- :func:`run_batched` — lockstep rounds for local engines (vLLM). Each round
  concatenates every live program's prompts *in record order* into ONE
  ``backend.generate`` call. This reproduces the legacy columnar batching
  exactly: all_at_once and step_by_step("batch") produce a single mega-batch
  with records in order (and steps in order within a record); binary_search
  produces one batch per recursion depth over the still-active records.
- :func:`run_streaming` — independent per-trajectory execution for API
  backends: a thread pool drives each program to completion, and ``on_done``
  fires (and the output file is written) the moment a trajectory finishes, so
  a crash mid-run only loses in-flight trajectories. A trajectory whose calls
  exhaust their retries is logged and *skipped without writing*, so a rerun
  resumes exactly the missing ids.

:class:`OutputWriter` owns the per-trajectory files: the method directory
``.../<model>/<method>/`` holds one ``<id>.json`` per trajectory (atomic
tmp-then-rename writes) — file existence IS the resume ledger.
"""
from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

OnDone = Callable[[object, dict], None]


def _start(key, gen, on_done: OnDone):
    """Advance a program to its first yield; returns None if it finished at once."""
    try:
        prompts = next(gen)
    except StopIteration as si:
        on_done(key, si.value)
        return None
    return prompts


def run_batched(programs: list[tuple[object, object]], backend, on_done: OnDone) -> None:
    """Lockstep rounds, one flat ``backend.generate`` per round."""
    live: list[list] = []  # [key, gen, pending_prompts]
    for key, gen in programs:
        prompts = _start(key, gen, on_done)
        if prompts is not None:
            live.append([key, gen, prompts])

    while live:
        flat: list[list[dict]] = []
        counts: list[int] = []
        for _key, _gen, prompts in live:
            flat.extend(prompts)
            counts.append(len(prompts))
        outputs = backend.generate(flat)

        pos = 0
        next_live: list[list] = []
        for (key, gen, _prompts), n in zip(live, counts):
            chunk = outputs[pos:pos + n]
            pos += n
            try:
                nxt = gen.send(chunk)
            except StopIteration as si:
                on_done(key, si.value)
            else:
                next_live.append([key, gen, nxt])
        live = next_live


def run_streaming(
    programs: list[tuple[object, object]],
    backend,
    on_done: OnDone,
    max_workers: int = 8,
) -> None:
    """Drive each program to completion independently on a thread pool."""
    lock = threading.Lock()

    def _done(key, pred):
        with lock:
            on_done(key, pred)

    def drive(key, gen):
        try:
            prompts = _start(key, gen, _done)
            while prompts is not None:
                outputs = backend.generate(prompts)
                try:
                    prompts = gen.send(outputs)
                except StopIteration as si:
                    _done(key, si.value)
                    return
        except Exception as exc:  # noqa: BLE001 — per-trajectory isolation
            print(f"  [stream] trajectory {key!r} failed: {exc!r} — skipped (rerun to resume)")

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(drive, key, gen) for key, gen in programs]
        for f in futures:
            f.result()  # surface driver bugs; per-trajectory errors are caught in drive()


class OutputWriter:
    """Per-trajectory JSON files under one method directory; resume by existence."""

    def __init__(self, method_dir: str | Path, overwrite: bool = False) -> None:
        self.dir = Path(method_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        if overwrite:
            for p in self.dir.glob("*.json"):
                p.unlink()
        # Stale tmp files from a crashed run are dead weight — never resumed from.
        for p in self.dir.glob("*.json.tmp"):
            p.unlink()

    def done_ids(self) -> set[str]:
        return {p.stem for p in self.dir.glob("*.json") if p.stem.isdigit()}

    def write(self, traj_id: str, doc: dict) -> None:
        """Atomic write: a crash can leave a *.tmp file, never a torn <id>.json."""
        tmp = self.dir / f"{traj_id}.json.tmp"
        tmp.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.dir / f"{traj_id}.json")

    def write_run_config(self, config: dict) -> None:
        (self.dir / "_run.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
