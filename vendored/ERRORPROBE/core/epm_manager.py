"""
Error Patch Memory (EPM) Manager

Manages EPM storage, retrieval, eviction policies (VBW, RFI-Δ, kSE, TTL)
"""

import json
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
from collections import defaultdict

from .epm_schema import (
    EPMEntry, EPMConfig, ErrorSignature, ErrorHypothesis,
    Evidence, ErrorPatch, ErrorSpan
)


class EPMManager:
    """
    Error Patch Memory Manager

    Implements:
    - VBW (Verified-Before-Write) gate
    - RFI-Δ scoring
    - kNN-Submodular Eviction (kSE)
    - TTL management
    """

    def __init__(self, config: Optional[EPMConfig] = None):
        self.config = config or EPMConfig()
        self.entries: Dict[str, EPMEntry] = {}
        self.last_decay_time = datetime.now()

    def __len__(self) -> int:
        return len(self.entries)

    # ========================================================================
    # VBW (Verified-Before-Write) Gate
    # ========================================================================

    def vbw_pass(self, evidence: Evidence) -> bool:
        """
        Check if entry passes Verified-Before-Write gate

        Write only if:
        - verified==True AND ablation_gain >= threshold
        OR
        - static_hits >= threshold AND verified==True
        """
        if not evidence.verified:
            return False

        # High ablation gain
        if evidence.ablation_gain >= self.config.vbw_min_ablation_gain:
            return True

        # Multiple static checks passed
        if evidence.static_hits >= self.config.vbw_min_static_hits:
            return True

        return False

    # ========================================================================
    # Write Operations
    # ========================================================================

    def write_entry(self, entry: EPMEntry, force: bool = False) -> bool:
        """
        Write entry to memory if it passes VBW gate

        Args:
            entry: EPM entry to write
            force: Skip VBW check (for testing)

        Returns:
            True if written, False if rejected
        """
        # Check VBW gate
        if not force and not self.vbw_pass(entry.evidence):
            return False

        # Check capacity
        if len(self.entries) >= self.config.max_entries and entry.key not in self.entries:
            self._evict_entries(1)

        # Write or update
        if entry.key in self.entries:
            # Update existing entry
            existing = self.entries[entry.key]
            existing.increment_usage()
            # Update with new evidence if better
            if entry.evidence.ablation_gain > existing.evidence.ablation_gain:
                existing.evidence = entry.evidence
                existing.patch = entry.patch
        else:
            # New entry
            self.entries[entry.key] = entry

        return True

    def write_from_hypothesis(self,
                            hypothesis: ErrorHypothesis,
                            evidence: Evidence,
                            source_run: str = "",
                            dataset: str = "") -> bool:
        """Create and write EPM entry from hypothesis"""
        if not hypothesis.patch:
            return False

        entry = EPMEntry.create(
            signature=hypothesis.signature,
            span=hypothesis.span,
            patch=hypothesis.patch,
            evidence=evidence,
            source_run=source_run,
            dataset=dataset
        )

        return self.write_entry(entry)

    # ========================================================================
    # Retrieval Operations
    # ========================================================================

    def retrieve(self, signature: ErrorSignature, k: Optional[int] = None) -> List[EPMEntry]:
        """
        Retrieve k most similar entries to the given signature

        Uses simple string similarity for now (can be upgraded to embeddings)
        """
        k = k or self.config.retrieval_k

        # Compute similarity scores
        scores = []
        for key, entry in self.entries.items():
            sim = self._compute_similarity(signature, entry.signature)
            if sim >= self.config.retrieval_threshold:
                scores.append((sim, entry))

        # Sort by similarity * RFI score
        scores.sort(key=lambda x: x[0] * x[1].scores.compute_composite(), reverse=True)

        # Return top-k
        results = [entry for _, entry in scores[:k]]

        # Update access stats
        for entry in results:
            entry.increment_usage()

        return results

    def _compute_similarity(self, sig1: ErrorSignature, sig2: ErrorSignature) -> float:
        """
        Compute similarity between two signatures

        Simple implementation: Jaccard similarity on components
        """
        # Family match is critical
        if sig1.err_family != sig2.err_family:
            return 0.0

        # Tool/API match
        tool_match = 1.0 if sig1.tool == sig2.tool else 0.0
        api_match = 1.0 if sig1.api == sig2.api else 0.0

        # Arg schema overlap
        args1 = set(sig1.arg_schema)
        args2 = set(sig2.arg_schema)
        arg_sim = len(args1 & args2) / max(len(args1 | args2), 1)

        # Context overlap
        ctx1 = set(sig1.context_slots)
        ctx2 = set(sig2.context_slots)
        ctx_sim = len(ctx1 & ctx2) / max(len(ctx1 | ctx2), 1)

        # Weighted average
        return 0.3 * tool_match + 0.3 * api_match + 0.2 * arg_sim + 0.2 * ctx_sim

    # ========================================================================
    # RFI-Δ Scoring
    # ========================================================================

    def compute_rfi_scores(self) -> Dict[str, float]:
        """Compute RFI-Δ scores for all entries"""
        scores = {}
        for key, entry in self.entries.items():
            scores[key] = entry.scores.compute_composite(
                alpha=self.config.rfi_alpha,
                beta=self.config.rfi_beta,
                gamma=self.config.rfi_gamma,
                delta=self.config.rfi_delta
            )
        return scores

    # ========================================================================
    # kNN-Submodular Eviction (kSE)
    # ========================================================================

    def _evict_entries(self, n: int = 1):
        """
        Evict n entries using kNN-Submodular Eviction

        Simplified: Remove entries with lowest RFI-Δ score
        Full implementation would compute marginal recall loss
        """
        if len(self.entries) == 0:
            return

        # Compute RFI scores
        scores = self.compute_rfi_scores()

        # Sort by score (ascending)
        sorted_keys = sorted(scores.keys(), key=lambda k: scores[k])

        # Evict lowest scoring entries
        for key in sorted_keys[:n]:
            del self.entries[key]

    # ========================================================================
    # TTL Management
    # ========================================================================

    def cleanup_expired(self) -> int:
        """Remove expired entries, return count removed"""
        expired = [key for key, entry in self.entries.items() if entry.meta.is_expired()]
        for key in expired:
            del self.entries[key]
        return len(expired)

    def apply_recency_decay(self):
        """Apply time-based recency decay"""
        now = datetime.now()
        elapsed = (now - self.last_decay_time).total_seconds()

        # Decay every hour
        if elapsed >= self.config.decay_interval:
            for entry in self.entries.values():
                entry.decay_recency(self.config.recency_decay_rate)
            self.last_decay_time = now

    # ========================================================================
    # Persistence
    # ========================================================================

    def save(self, filepath: str):
        """Save EPM to file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        data = {
            "config": self.config.to_dict(),
            "entries": [entry.to_dict() for entry in self.entries.values()],
            "metadata": {
                "count": len(self.entries),
                "last_decay": self.last_decay_time.isoformat()
            }
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "EPMManager":
        """Load EPM from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        config = EPMConfig(**data["config"])
        manager = cls(config)

        for entry_data in data["entries"]:
            entry = EPMEntry.from_dict(entry_data)
            manager.entries[entry.key] = entry

        if "metadata" in data and "last_decay" in data["metadata"]:
            manager.last_decay_time = datetime.fromisoformat(data["metadata"]["last_decay"])

        return manager

    # ========================================================================
    # Statistics & Reporting
    # ========================================================================

    def get_statistics(self) -> Dict[str, any]:
        """Get memory statistics"""
        if not self.entries:
            return {
                "total_entries": 0,
                "families": {},
                "avg_confidence": 0.0,
                "avg_freq": 0.0,
                "expired_count": 0
            }

        families = defaultdict(int)
        confidences = []
        frequencies = []

        for entry in self.entries.values():
            families[entry.signature.err_family] += 1
            confidences.append(entry.evidence.ablation_gain)
            frequencies.append(entry.scores.freq)

        expired_count = sum(1 for e in self.entries.values() if e.meta.is_expired())

        return {
            "total_entries": len(self.entries),
            "families": dict(families),
            "avg_confidence": np.mean(confidences) if confidences else 0.0,
            "avg_freq": np.mean(frequencies) if frequencies else 0.0,
            "expired_count": expired_count,
            "capacity_used": len(self.entries) / self.config.max_entries
        }

    def get_top_entries(self, n: int = 10) -> List[EPMEntry]:
        """Get top-n entries by RFI-Δ score"""
        scores = self.compute_rfi_scores()
        sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        return [self.entries[k] for k in sorted_keys[:n]]
