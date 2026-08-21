"""
Error Patch Memory (EPM) Schema and Data Structures

Core data structures for the Probe-and-Verify evaluator MAS.
"""

import hashlib
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
from datetime import datetime


# ============================================================================
# Multi-Agent System Failure Taxonomy (MAST)
# Based on: "Why Do Multi-Agent LLM Systems Fail?"
#
# 14 fine-grained failure modes organized into 3 main categories:
# - FC1: Design and Specification Failures
# - FC2: Coordination and Communication Failures
# - FC3: Verification and Termination Failures
# ============================================================================

ERROR_FAMILIES = {
    # FC1: Design and Specification Failures (41.77%)
    # Failures from system design decisions and poor/ambiguous prompt specifications
    "fail_task_spec": "Fails to follow task requirements/instructions",
    "fail_role_spec": "Fails to follow assigned agent role specifications",
    "step_repetition": "Repeats same steps due to rigid turn configurations",
    "context_loss": "Loses conversation history or critical context information",
    "unaware_stopping": "Fails to recognize task completion conditions",

    # FC2: Coordination and Communication Failures (36.94%)
    # Failures from breakdowns in inter-agent interaction and coordination
    "conversation_reset": "Unexpected conversation restart losing prior context",
    "fail_clarification": "Proceeds with wrong assumptions instead of asking clarification",
    "task_derailment": "Deviates from original task objective",
    "info_withholding": "Withholds crucial information from other agents",
    "ignored_input": "Ignores inputs or feedback from other agents",
    "reasoning_action_mismatch": "Mismatch between stated reasoning and actual actions taken",

    # FC3: Verification and Termination Failures (21.30%)
    # Failures from inadequate verification or premature termination
    "premature_termination": "Terminates task before completion",
    "incomplete_verification": "No verification or only superficial checks performed",
    "incorrect_verification": "Verification process produces incorrect assessment",

    # Legacy/Additional Categories (for backward compatibility)
    # These can be mapped to MAST categories or deprecated over time
    "schema_mismatch": "Tool argument types/fields incorrect",
    "retry_storm": "Looping on same failing tool call",
    "hallucinated_tool": "Attempts to use non-existent tool",
    "incorrect_reasoning": "Logical errors in multi-step reasoning",
    "incomplete_execution": "Partial execution without completion",
    "posthoc_rationalization": "Confident but unsupported final step"
}

# Failure Category Mappings
FAILURE_CATEGORIES = {
    "Design_Specification": [
        "fail_task_spec",
        "fail_role_spec",
        "step_repetition",
        "context_loss",
        "unaware_stopping"
    ],
    "Coordination_Communication": [
        "conversation_reset",
        "fail_clarification",
        "task_derailment",
        "info_withholding",
        "ignored_input",
        "reasoning_action_mismatch"
    ],
    "Verification_Termination": [
        "premature_termination",
        "incomplete_verification",
        "incorrect_verification"
    ]
}

# Legacy to MAST mapping for migration
LEGACY_TO_MAST_MAPPING = {
    "incomplete_execution": "premature_termination",
    "context_overflow": "context_loss",
    "bridge_failure": "reasoning_action_mismatch",
    "insufficient_evidence": "fail_clarification",
    "planner_overreach": "task_derailment",
    "result_misparse": "incorrect_verification",
    "order_dependency": "reasoning_action_mismatch",
    "constraint_violation": "fail_task_spec",
    "memory_staleness": "context_loss"
}


# ============================================================================
# Helper Functions for MAST Taxonomy
# ============================================================================

def get_failure_category(error_family: str) -> str:
    """
    Get the main failure category (FC1/FC2/FC3) for a given error family.

    Args:
        error_family: The specific error family/mode

    Returns:
        The failure category name (e.g., "FC1_Design_Specification")
    """
    for category, modes in FAILURE_CATEGORIES.items():
        if error_family in modes:
            return category

    # Check if it's a legacy category that can be mapped
    if error_family in LEGACY_TO_MAST_MAPPING:
        mapped_family = LEGACY_TO_MAST_MAPPING[error_family]
        for category, modes in FAILURE_CATEGORIES.items():
            if mapped_family in modes:
                return category

    return "Unknown"


def migrate_legacy_family(legacy_family: str) -> str:
    """
    Migrate a legacy error family to the MAST taxonomy.

    Args:
        legacy_family: The legacy error family name

    Returns:
        The corresponding MAST error family name
    """
    return LEGACY_TO_MAST_MAPPING.get(legacy_family, legacy_family)


def is_mast_family(error_family: str) -> bool:
    """
    Check if an error family is part of the core MAST taxonomy.

    Args:
        error_family: The error family to check

    Returns:
        True if it's a MAST category, False if legacy
    """
    for modes in FAILURE_CATEGORIES.values():
        if error_family in modes:
            return True
    return False


def get_all_mast_families() -> List[str]:
    """
    Get all error families in the MAST taxonomy (excluding legacy).

    Returns:
        List of all MAST error family names
    """
    all_families = []
    for modes in FAILURE_CATEGORIES.values():
        all_families.extend(modes)
    return all_families


def get_family_statistics(families: List[str]) -> Dict[str, int]:
    """
    Get statistics on error families grouped by category.

    Args:
        families: List of error family names

    Returns:
        Dictionary with category counts and percentages
    """
    category_counts = {
        "FC1_Design_Specification": 0,
        "FC2_Coordination_Communication": 0,
        "FC3_Verification_Termination": 0,
        "Legacy": 0,
        "Unknown": 0
    }

    for family in families:
        category = get_failure_category(family)
        if category in category_counts:
            category_counts[category] += 1
        elif family in ERROR_FAMILIES and "(legacy)" in ERROR_FAMILIES[family]:
            category_counts["Legacy"] += 1
        else:
            category_counts["Unknown"] += 1

    total = len(families)
    stats = {
        "total": total,
        "counts": category_counts,
        "percentages": {k: (v / total * 100 if total > 0 else 0)
                       for k, v in category_counts.items()}
    }

    return stats


# ============================================================================
# EPM Entry Data Structures
# ============================================================================

@dataclass
class ErrorSignature:
    """Signature for error identification and retrieval"""
    tool: str
    api: str
    arg_schema: List[str]
    context_slots: List[str]
    err_family: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def compute_hash(self) -> str:
        """Compute stable hash for this signature"""
        sig_str = f"{self.tool}|{self.api}|{','.join(sorted(self.arg_schema))}|{','.join(sorted(self.context_slots))}|{self.err_family}"
        return hashlib.sha256(sig_str.encode()).hexdigest()[:16]


@dataclass
class ErrorSpan:
    """Location of error in trace"""
    start_turn: int
    end_turn: int
    role_path: List[str]
    position_info: Optional[Dict[str, Any]] = None  # Position metadata for guidance

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def add_position_info(self, total_turns: int):
        """
        Calculate and add position information based on trace length.

        This helps the LLM understand WHERE in the trace errors typically occur.
        """
        if total_turns <= 0:
            return

        start_pct = self.start_turn / total_turns
        end_pct = self.end_turn / total_turns
        mid_pct = (start_pct + end_pct) / 2
        duration_pct = (self.end_turn - self.start_turn) / total_turns

        # Determine position label
        if mid_pct < 0.33:
            label = "early"
        elif mid_pct < 0.67:
            label = "middle"
        else:
            label = "late"

        self.position_info = {
            "start_pct": round(start_pct, 3),
            "end_pct": round(end_pct, 3),
            "mid_pct": round(mid_pct, 3),
            "duration_pct": round(duration_pct, 3),
            "position_label": label,
            "total_turns": total_turns
        }


@dataclass
class ErrorPatch:
    """Minimal actionable fix"""
    guard: Optional[str] = None
    rewrite: Optional[str] = None
    prompt_hint: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class Evidence:
    """Verification evidence for the error hypothesis"""
    ablation_gain: float = 0.0
    static_hits: int = 0
    replays: int = 0
    verified: bool = False
    probe_details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RFIScores:
    """Recency-Frequency-Impact-DeltaPerf scores for memory management"""
    recency: float = 1.0  # 0-1, decays over time
    freq: int = 1  # usage count
    imp: float = 0.0  # normalized ablation_gain × patch success rate
    dperf: float = 0.0  # change in trace-level success

    def compute_composite(self, alpha=0.4, beta=0.2, gamma=0.3, delta=0.1) -> float:
        """Compute composite RFI-Δ score"""
        # Normalize frequency (log scale, cap at 100)
        norm_freq = min(self.freq, 100) / 100.0
        return (alpha * self.recency +
                beta * norm_freq +
                gamma * self.imp +
                delta * self.dperf)

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['S'] = self.compute_composite()
        return result


@dataclass
class EPMMetadata:
    """Metadata for EPM entry"""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    ttl: int = 172800  # 48h default
    source_run: str = ""
    dataset: str = ""
    last_accessed: str = field(default_factory=lambda: datetime.now().isoformat())
    access_count: int = 0

    def is_expired(self) -> bool:
        """Check if entry has expired"""
        created = datetime.fromisoformat(self.created_at)
        elapsed = (datetime.now() - created).total_seconds()
        return elapsed > self.ttl

    def touch(self):
        """Update last access time"""
        self.last_accessed = datetime.now().isoformat()
        self.access_count += 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EPMEntry:
    """Complete EPM entry structure"""
    key: str
    signature: ErrorSignature
    span: ErrorSpan
    patch: ErrorPatch
    evidence: Evidence
    scores: RFIScores
    meta: EPMMetadata

    @classmethod
    def create(cls,
               signature: ErrorSignature,
               span: ErrorSpan,
               patch: ErrorPatch,
               evidence: Evidence,
               source_run: str = "",
               dataset: str = "") -> "EPMEntry":
        """Factory method to create EPM entry"""
        key = signature.compute_hash()
        scores = RFIScores(
            recency=1.0,
            freq=1,
            imp=evidence.ablation_gain if evidence.ablation_gain > 0 else 0.0,
            dperf=0.0
        )
        meta = EPMMetadata(
            source_run=source_run,
            dataset=dataset,
            ttl=604800 if evidence.verified and evidence.ablation_gain > 0.5 else 172800  # 7d if high confidence, 48h otherwise
        )

        return cls(
            key=key,
            signature=signature,
            span=span,
            patch=patch,
            evidence=evidence,
            scores=scores,
            meta=meta
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "signature": self.signature.to_dict(),
            "span": self.span.to_dict(),
            "patch": self.patch.to_dict(),
            "evidence": self.evidence.to_dict(),
            "scores": self.scores.to_dict(),
            "meta": self.meta.to_dict()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EPMEntry":
        """Load from dictionary"""
        # Extract scores data and remove computed field
        scores_data = data["scores"].copy()
        scores_data.pop('S', None)  # Remove computed composite score

        return cls(
            key=data["key"],
            signature=ErrorSignature(**data["signature"]),
            span=ErrorSpan(**data["span"]),
            patch=ErrorPatch(**data["patch"]),
            evidence=Evidence(**data["evidence"]),
            scores=RFIScores(**scores_data),
            meta=EPMMetadata(**data["meta"])
        )

    def decay_recency(self, decay_rate: float = 0.99):
        """Apply time-based recency decay"""
        self.scores.recency *= decay_rate

    def increment_usage(self):
        """Increment frequency and update metadata"""
        self.scores.freq += 1
        self.meta.touch()

    def update_performance(self, success_delta: float):
        """Update delta performance based on usage"""
        # Running average of performance impact
        self.scores.dperf = (self.scores.dperf * (self.scores.freq - 1) + success_delta) / self.scores.freq


# ============================================================================
# Hypothesis Structure (for analysis pipeline)
# ============================================================================

@dataclass
class ErrorHypothesis:
    """Hypothesis about where/why a trace failed"""
    span: ErrorSpan
    family: str
    rationale: str
    signature: ErrorSignature
    patch: Optional[ErrorPatch] = None
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "span": self.span.to_dict(),
            "family": self.family,
            "rationale": self.rationale,
            "signature": self.signature.to_dict(),
            "confidence": self.confidence
        }
        if self.patch:
            result["patch"] = self.patch.to_dict()
        return result


# ============================================================================
# Memory Configuration
# ============================================================================

@dataclass
class EPMConfig:
    """Configuration for EPM management"""
    # Capacity
    max_entries: int = 500

    # VBW (Verified-Before-Write) thresholds
    vbw_min_ablation_gain: float = 0.2
    vbw_min_static_hits: int = 1

    # RFI-Δ weights
    rfi_alpha: float = 0.4  # recency
    rfi_beta: float = 0.2   # frequency
    rfi_gamma: float = 0.3  # impact
    rfi_delta: float = 0.1  # delta perf

    # Retrieval
    retrieval_k: int = 5
    retrieval_threshold: float = 0.75

    # TTL
    ttl_low_confidence: int = 172800  # 48h
    ttl_high_confidence: int = 604800  # 7d

    # Eviction
    eviction_batch_size: int = 10  # evict in batches when full

    # Decay
    recency_decay_rate: float = 0.99
    decay_interval: int = 3600  # decay every hour

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
