"""
Run Manager - Handles run organization and logging

Implements the runs/<run_id>/ structure from specification.
"""

import os
import json
import hashlib
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path


class RunManager:
    """
    Manages run directory structure:

    runs/<run_id>/
      manifest.json            # seeds, commit hash, config
      traces/failed_*.jsonl    # raw inputs
      probes/*.jsonl           # per-hypothesis probe results
      predictions/*.jsonl      # (span, family, reason, confidence)
      memory/epm.jsonl         # EPM snapshot AFTER the run
      metrics/summary.json     # diagnostics, time, ablations
    """

    def __init__(self,
                 base_dir: str = "runs",
                 run_id: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Args:
            base_dir: Base directory for all runs
            run_id: Run ID (auto-generated if None)
            config: Configuration dict
        """
        self.base_dir = Path(base_dir)
        self.run_id = run_id or self._generate_run_id()
        self.run_dir = self.base_dir / self.run_id
        self.config = config or {}

        # Create directory structure
        self._setup_directories()

        # Initialize manifest
        self.manifest = {
            "run_id": self.run_id,
            "created_at": datetime.now().isoformat(),
            "config": self.config,
            "git_commit": self._get_git_commit(),
            "seed": self._get_seed()
        }

        self._save_manifest()

    def _generate_run_id(self) -> str:
        """Generate unique run ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = hashlib.md5(str(datetime.now().timestamp()).encode()).hexdigest()[:6]
        return f"run_{timestamp}_{random_suffix}"

    def _setup_directories(self):
        """Create run directory structure"""
        dirs = [
            self.run_dir,
            self.run_dir / "traces",
            self.run_dir / "probes",
            self.run_dir / "predictions",
            self.run_dir / "memory",
            self.run_dir / "metrics"
        ]

        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    def _get_git_commit(self) -> str:
        """Get current git commit hash"""
        try:
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.stdout.strip() if result.returncode == 0 else "unknown"
        except:
            return "unknown"

    def _get_seed(self) -> int:
        """Get random seed from config or generate"""
        if "seed" in self.config:
            return self.config["seed"]
        return int(datetime.now().timestamp() * 1000) % (2**32)

    def _save_manifest(self):
        """Save manifest to file"""
        manifest_path = self.run_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(self.manifest, f, indent=2)

    def log_trace(self, trace_id: str, trace: Dict[str, Any]):
        """Log input trace"""
        trace_path = self.run_dir / "traces" / f"failed_{trace_id}.jsonl"
        with open(trace_path, 'a') as f:
            f.write(json.dumps(trace) + "\n")

    def log_probes(self, trace_id: str, hypotheses: List[Dict[str, Any]]):
        """Log probe results for hypotheses"""
        probe_path = self.run_dir / "probes" / f"{trace_id}_probes.jsonl"
        with open(probe_path, 'w') as f:
            for h in hypotheses:
                f.write(json.dumps(h) + "\n")

    def log_prediction(self, trace_id: str, prediction: Dict[str, Any]):
        """Log prediction result"""
        pred_path = self.run_dir / "predictions" / f"{trace_id}_prediction.jsonl"
        with open(pred_path, 'w') as f:
            f.write(json.dumps(prediction) + "\n")

    def save_memory(self, epm_data: Dict[str, Any]):
        """Save EPM snapshot"""
        memory_path = self.run_dir / "memory" / "epm.jsonl"
        with open(memory_path, 'w') as f:
            # Save as JSONL (one entry per line)
            if "entries" in epm_data:
                for entry in epm_data["entries"]:
                    f.write(json.dumps(entry) + "\n")

    def save_metrics(self, metrics: Dict[str, Any]):
        """Save run metrics"""
        metrics_path = self.run_dir / "metrics" / "summary.json"

        # Add timestamp
        metrics["timestamp"] = datetime.now().isoformat()
        metrics["run_id"] = self.run_id

        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

    def save_full_results(self,
                         traces: List[Dict[str, Any]],
                         results: List[Dict[str, Any]],
                         metrics: Dict[str, Any],
                         epm_manager):
        """
        Save complete run results

        Args:
            traces: Input traces
            results: Analysis results
            metrics: Overall metrics
            epm_manager: EPM manager instance
        """
        # Save traces
        for idx, trace in enumerate(traces):
            trace_id = trace.get("id", f"trace_{idx:04d}")
            self.log_trace(trace_id, trace)

        # Save results
        for idx, (result, trace) in enumerate(zip(results, traces)):
            trace_id = trace.get("id", f"trace_{idx:04d}")

            # Log probes
            if "hypotheses" in result:
                self.log_probes(trace_id, result["hypotheses"])

            # Log prediction
            if "prediction" in result:
                self.log_prediction(trace_id, result["prediction"])

        # Save memory
        epm_path = self.run_dir / "memory" / "epm.json"
        epm_manager.save(str(epm_path))

        # Save metrics
        self.save_metrics(metrics)

        # Update manifest
        self.manifest["completed_at"] = datetime.now().isoformat()
        self.manifest["trace_count"] = len(traces)
        self._save_manifest()

    def get_run_summary(self) -> Dict[str, Any]:
        """Get summary of this run"""
        metrics_path = self.run_dir / "metrics" / "summary.json"

        summary = {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "manifest": self.manifest
        }

        # Load metrics if available
        if metrics_path.exists():
            with open(metrics_path) as f:
                summary["metrics"] = json.load(f)

        # Count files
        summary["file_counts"] = {
            "traces": len(list((self.run_dir / "traces").glob("*.jsonl"))),
            "probes": len(list((self.run_dir / "probes").glob("*.jsonl"))),
            "predictions": len(list((self.run_dir / "predictions").glob("*.jsonl")))
        }

        return summary

    @staticmethod
    def list_runs(base_dir: str = "runs") -> List[Dict[str, Any]]:
        """List all runs in base directory"""
        base = Path(base_dir)

        if not base.exists():
            return []

        runs = []
        for run_dir in base.iterdir():
            if run_dir.is_dir() and (run_dir / "manifest.json").exists():
                with open(run_dir / "manifest.json") as f:
                    manifest = json.load(f)
                    runs.append({
                        "run_id": manifest["run_id"],
                        "created_at": manifest["created_at"],
                        "path": str(run_dir)
                    })

        # Sort by creation time (newest first)
        runs.sort(key=lambda x: x["created_at"], reverse=True)

        return runs

    @staticmethod
    def load_run(run_id: str, base_dir: str = "runs") -> "RunManager":
        """Load existing run"""
        run_dir = Path(base_dir) / run_id

        if not run_dir.exists():
            raise ValueError(f"Run {run_id} not found")

        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            raise ValueError(f"Run {run_id} has no manifest")

        with open(manifest_path) as f:
            manifest = json.load(f)

        manager = RunManager.__new__(RunManager)
        manager.base_dir = Path(base_dir)
        manager.run_id = run_id
        manager.run_dir = run_dir
        manager.config = manifest.get("config", {})
        manager.manifest = manifest

        return manager
