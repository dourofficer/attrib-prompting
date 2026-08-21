#!/usr/bin/env python3
"""
Training Script for Simplified LLM-Based MAS

Uses:
- LLM for analysis and verification
- Algorithmic memory management (VBW + RFI-Δ)
"""

import argparse
import json
import sys
from pathlib import Path
from tqdm import tqdm
import time

from simplified_mas import SimplifiedMAS
from utils import RunManager


def load_traces(filepath: str, max_traces: int = None):
    """Load traces from JSONL file"""
    traces = []
    with open(filepath) as f:
        for idx, line in enumerate(f):
            if max_traces and idx >= max_traces:
                break
            trace = json.loads(line.strip())
            if "id" not in trace:
                trace["id"] = f"trace_{idx:04d}"
            traces.append(trace)
    return traces


def train_mas(traces, config_path, run_manager):
    """Train MAS on traces"""
    # Initialize system
    mas = SimplifiedMAS(config_path)

    print(f"\n{'='*60}")
    print(f"Training Simplified LLM-Based MAS")
    print(f"{'='*60}")
    print(f"Traces: {len(traces)}")
    print(f"LLM Model: {mas.llm_config.model_id}")
    print()

    results = []
    start_time = time.time()

    for idx, trace in enumerate(tqdm(traces, desc="Training")):
        print(f"\n[Trace {idx+1}/{len(traces)}] ID: {trace['id']}")

        try:
            result = mas.analyze_trace(trace, source_run=run_manager.run_id)
            results.append({
                "trace_id": trace["id"],
                "analysis": result["analysis"],
                "verified": result["evidence"]["verified"],
                "confidence": result["confidence"],
                "stored": result["memory_stored"]
            })

        except Exception as e:
            print(f"  [Error] {str(e)}")
            results.append({
                "trace_id": trace["id"],
                "error": str(e),
                "verified": False,
                "confidence": 0.0,
                "stored": False
            })

        # Print progress every 5 traces
        if (idx + 1) % 5 == 0:
            stats = mas.get_statistics()
            print(f"\n  Progress: {idx+1}/{len(traces)}")
            print(f"  Verified: {stats['hypotheses_verified']}")
            print(f"  Stored: {stats['patterns_stored']}")
            print(f"  Memory size: {stats['memory']['total_entries']}")

    elapsed = time.time() - start_time

    # Final statistics
    stats = mas.get_statistics()

    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"{'='*60}")
    print(f"Total traces: {len(traces)}")
    print(f"Verified: {stats['hypotheses_verified']} ({stats['verification_rate']:.1%})")
    print(f"Stored in memory: {stats['patterns_stored']}")
    print(f"Memory size: {stats['memory']['total_entries']}")
    print(f"Error families: {stats['memory']['families']}")
    print(f"Time: {elapsed:.1f}s ({elapsed/len(traces):.2f}s/trace)")
    print(f"Results: {run_manager.run_dir}")

    # Save memory
    memory_path = run_manager.run_dir / "memory" / "epm.json"
    mas.save_memory(str(memory_path))

    # Save results
    results_path = run_manager.run_dir / "training_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            "statistics": stats,
            "results": results,
            "elapsed_time": elapsed
        }, f, indent=2)

    return mas, stats


def main():
    parser = argparse.ArgumentParser(description="Train Simplified LLM-Based MAS")

    parser.add_argument("--train", required=True, help="Training traces (JSONL)")
    parser.add_argument("--max-traces", type=int, help="Max traces to use")
    parser.add_argument("--config", default="config.yaml", help="Config file")
    parser.add_argument("--output", default="runs", help="Output directory")
    parser.add_argument("--run-id", help="Custom run ID")

    args = parser.parse_args()

    # Load traces
    print(f"Loading traces from {args.train}...")
    traces = load_traces(args.train, args.max_traces)
    print(f"Loaded {len(traces)} traces")

    # Create run manager
    run_manager = RunManager(
        base_dir=args.output,
        run_id=args.run_id or f"simplified_mas_train",
        config={
            "train_file": args.train,
            "max_traces": args.max_traces,
            "config_file": args.config
        }
    )

    # Train
    mas, stats = train_mas(traces, args.config, run_manager)

    print(f"\nMemory saved: {run_manager.run_dir / 'memory' / 'epm.json'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
