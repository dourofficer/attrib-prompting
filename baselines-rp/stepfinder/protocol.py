"""Where the training data comes from, and who gets scored with it.

StepFinder is supervised, so before anything can run something has to decide
which labelled failures the model learns from. This repo answers that twice,
and the two answers are the only thing that separates the package's two output
families:

**The vendored protocol** (``regen``, the default) is the paper's. It trains on
the regenerated failure trajectories that ship with the code — 1,564 for
Algorithm-Generated, 2,604 for Hand-Crafted — which are disjoint from the
official test sets at the task level. Verified here: zero questions are shared
with either ``data/ww`` subset. One model per seed, scoring everything.

**The in-corpus protocol** (``in-corpus``) trains on the corpus under test,
using the 30% partition the evaluation protocol already sets aside and no
other baseline touches. It answers a different question — what StepFinder does
with in-domain supervision — and it can only answer it for the 70% it did not
train on, so its output covers part of each corpus by construction.

Everything downstream of :func:`jobs_for` takes a :class:`Job` and never asks
which family produced it. That is deliberate: extraction, caching, scoring,
resume and the output schema are shared code, and the string ``"in-corpus"``
appears nowhere outside this module and the CLI parser.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from baselines.shared.common import _get_sorted_json_files, _load_json_data, split_data
from stepfinder.features import AGENT_FIELD_REPO, AGENT_FIELD_VENDORED
from stepfinder.train import PRESETS, HParams

# The two corpora that ship with the paper's code.
TRAIN_SETS = ("algorithm-generated", "hand-crafted")
VENDORED_DIRS = {
    "algorithm-generated": "Algorithm-Generated/train",
    "hand-crafted": "Hand-Crafted/train",
}
DEFAULT_TRAIN_DATA_ROOT = "vendored/StepFinder/data"

# Which training set and hyperparameters a subset gets when nothing overrides.
# Pair each subset with the corpus built from the same agent system. Who&When's
# Algorithm-Generated half was produced by CaptainAgent and its Hand-Crafted
# half by Magentic-One (`vendored/Agents_Failure_Attribution/README.md:39`), so
# `ww/algorithm-generated` and `traceelephant/captain` — both CaptainAgent, both
# `*_Expert` vocabularies — train on Algorithm-Generated, while
# `ww/hand-crafted` and `traceelephant/magentic` train on Hand-Crafted.
# `correct-error` matches neither system and falls back to Hand-Crafted, and so
# does `tracertraj` (MetaGPT roles) — its configs run both corpora explicitly.
DEFAULT_TRAIN_SET = "hand-crafted"
TRAIN_SET_BY_SUBSET = {
    "algorithm-generated": "algorithm-generated",
    "captain": "algorithm-generated",
}
PRESET_BY_TRAIN_SET = {"algorithm-generated": "alg", "hand-crafted": "hc"}


def resolve_train_set(subset: str, override: str = "auto") -> str:
    if override and override != "auto":
        if override not in TRAIN_SETS:
            raise SystemExit(f"--train-set must be one of {TRAIN_SETS} or 'auto', got {override!r}")
        return override
    return TRAIN_SET_BY_SUBSET.get(subset, DEFAULT_TRAIN_SET)


def resolve_preset(train_set: str, override: str = "auto") -> str:
    if override and override != "auto":
        if override not in PRESETS:
            raise SystemExit(f"--preset must be one of {sorted(PRESETS)} or 'auto', got {override!r}")
        return override
    return PRESET_BY_TRAIN_SET.get(train_set, "hc")


# --------------------------------------------------------------------------
# Loading the vendored training corpus
# --------------------------------------------------------------------------

def load_vendored_records(train_dir: str | Path) -> list[dict]:
    """The regenerated failures, in the record shape the rest of the repo uses.

    Shaped like ``baselines.prompting.predict.load_records`` so a training
    trajectory and a corpus trajectory are indistinguishable downstream —
    except for ``agent_field``, which these files keep in ``name`` per the
    upstream convention.
    """
    directory = Path(train_dir)
    records = []
    for filename in _get_sorted_json_files(directory):
        data = _load_json_data(directory / filename)
        if not data:
            continue
        history = data.get("history") or []
        if not history:
            continue
        records.append({
            "id": Path(filename).stem,
            "filename": filename,
            "question_id": None,
            "history": history,
            "question": data.get("question", ""),
            "ground_truth": data.get("ground_truth", ""),
            "gold_agent": data.get("mistake_agent"),
            "gold_step": data.get("mistake_step"),
        })
    return records


# --------------------------------------------------------------------------
# A training set, and a unit of work
# --------------------------------------------------------------------------

@dataclass
class TrainingSet:
    """What a model learns from, and everything needed to name its checkpoint."""

    name: str                    # "hand-crafted" | "in-corpus:s7"
    protocol: str                # "regen" | "in-corpus"
    source: str                  # directory or corpus the records came from
    records: list
    agent_field: str
    preset: str
    hparams: HParams
    feats_root: Optional[Path] = None   # None: features live with the scored subset
    questions: frozenset = field(default_factory=frozenset)

    def __post_init__(self):
        if not self.questions:
            self.questions = frozenset(
                str(r.get("question", "")).strip() for r in self.records
                if str(r.get("question", "")).strip()
            )

    @property
    def ids(self) -> list[str]:
        return [str(r["id"]) for r in self.records]


@dataclass
class Job:
    """One trained model and the trajectories it is responsible for."""

    method_dir: str              # "stepfinder.s42" | "stepfinder.e7"
    seed: int
    training_set: TrainingSet
    scored_ids: Optional[set] = None   # None: every trajectory in the slice


def regen_training_set(
    train_data_root: str,
    train_set: str,
    preset: str,
    hparams: HParams,
    feats_root: Path,
) -> TrainingSet:
    source = str(Path(train_data_root) / VENDORED_DIRS[train_set])
    return TrainingSet(
        name=train_set,
        protocol="regen",
        source=source,
        records=load_vendored_records(source),
        agent_field=AGENT_FIELD_VENDORED,
        preset=preset,
        hparams=hparams,
        feats_root=feats_root,
    )


def train_partition_ids(data_dir: str | Path, splits: dict, seed: int) -> list[str]:
    """The 30% no other baseline uses, reproduced exactly as the report cuts it.

    ``baselines.prompting.report.val_test_ids`` throws this slice away
    (``report.py:126`` names it ``_train``); the in-corpus protocol is what
    finally spends it. The cut is over the *file list*, not over loaded
    records, because that is the universe the report's splits are defined on —
    ``load_records`` silently drops empty histories, and a partition built
    after that drop would not line up with the val and test ids being scored.
    """
    train, val = float(splits["train"]), float(splits["val"])
    files = _get_sorted_json_files(Path(data_dir))
    trval, _test = split_data(files, train + val, seed)
    tr, _va = split_data(trval, train / (train + val), seed)
    return [Path(f).stem for f in tr]


def scored_partition_ids(data_dir: str | Path, splits: dict, seed: int) -> set:
    """Val + test for one seed — everything the in-corpus family may predict."""
    train, val = float(splits["train"]), float(splits["val"])
    files = _get_sorted_json_files(Path(data_dir))
    trval, test = split_data(files, train + val, seed)
    _tr, va = split_data(trval, train / (train + val), seed)
    return {Path(f).stem for f in va} | {Path(f).stem for f in test}


def in_corpus_training_set(
    data_dir: str,
    records: list,
    splits: dict,
    seed: int,
    preset: str,
    hparams: HParams,
) -> TrainingSet:
    keep = set(train_partition_ids(data_dir, splits, seed))
    return TrainingSet(
        name=f"in-corpus:s{seed}",
        protocol="in-corpus",
        source=str(data_dir),
        records=[r for r in records if str(r["id"]) in keep],
        agent_field=AGENT_FIELD_REPO,
        preset=preset,
        hparams=hparams,
        feats_root=None,
    )


def jobs_for(
    protocol: str,
    *,
    method_prefix: str,
    seeds: list[int],
    training_set_factory,
    data_dir: Optional[str] = None,
    splits: Optional[dict] = None,
) -> list[Job]:
    """Family A: one training set, many seeds. Family B: one of each, per seed."""
    if protocol == "regen":
        shared = training_set_factory(None)
        return [Job(f"{method_prefix}.s{s}", s, shared, None) for s in seeds]
    if protocol == "in-corpus":
        return [
            Job(f"{method_prefix}.e{s}", s, training_set_factory(s),
                scored_partition_ids(data_dir, splits, s))
            for s in seeds
        ]
    raise SystemExit(f"protocol must be 'regen' or 'in-corpus', got {protocol!r}")


def train_task_overlap(record: dict, training_set: TrainingSet) -> bool:
    """Did the model see a labelled failure on this exact task?

    Always answered against the training set actually used, never a constant.
    The default subset mapping is leak-free on ``ww`` — but the Hand-Crafted
    set shares 69 questions with ``ww/algorithm-generated`` and the
    Algorithm-Generated set shares 11 with ``ww/hand-crafted``, so a config
    that swaps the mapping would create leakage where the table says there is
    none. Under the defaults the flag fires on 22 of 85 ``traceelephant/captain``
    trajectories, 51 of 91 ``magentic``, and 30 ``correct-error/gaia``: those
    benchmarks reuse GAIA questions, and the same task appears in training with
    a different agent system and a different decisive step.
    """
    question = str(record.get("question", "")).strip()
    return bool(question) and question in training_set.questions
