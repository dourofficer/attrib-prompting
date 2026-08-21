#!/usr/bin/env python3
"""
Evaluation Script for Simplified LLM-Based MAS

Tests the learned memory on a test set to measure:
- How well learned patterns help identify errors
- Accuracy of error detection
- Quality of generated fixes
"""

import argparse
import json
import sys
from pathlib import Path
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from simplified_mas import SimplifiedMAS


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


# Global shared MAS instance and lock for thread-safe access
_shared_mas = None
_mas_lock = threading.Lock()


def get_shared_mas(config_path, memory_path):
    """Get or create shared MAS instance (thread-safe)"""
    global _shared_mas
    if _shared_mas is None:
        _shared_mas = SimplifiedMAS(config_path)
        if memory_path:
            _shared_mas.load_memory(memory_path)
    return _shared_mas


def evaluate_single_trace(args):
    """Worker function for parallel trace processing"""
    trace, idx, total, config_path, memory_path, run_id, checkpoint_path = args

    # Get shared MAS instance
    mas = get_shared_mas(config_path, memory_path)

    try:
        # Thread-safe analyze_trace call
        with _mas_lock:
            result = mas.analyze_trace(trace, source_run=run_id)

            # Save memory checkpoint after each trace
            if checkpoint_path:
                mas.save_memory(checkpoint_path)

        # Extract ground truth
        gt_step = trace.get("mistake_step", -1)
        if isinstance(gt_step, str):
            try:
                gt_step = int(gt_step)
            except ValueError:
                gt_step = -1

        ground_truth = {
            "error_agent": trace.get("mistake_agent", "unknown"),
            "error_step": gt_step,
            "error_reason": trace.get("mistake_reason", "unknown")
        }

        # Compare with prediction
        predicted_agent = result["analysis"]["tool_used"]
        predicted_step = result["analysis"]["error_step"]
        predicted_family = result["analysis"]["error_family"]

        # Calculate accuracy metrics
        agent_match = predicted_agent == ground_truth["error_agent"]
        step_match = (predicted_step[0] <= ground_truth["error_step"] <= predicted_step[1]) \
                    if ground_truth["error_step"] >= 0 else False

        trace_result = {
            "trace_id": trace["id"],
            "ground_truth": ground_truth,
            "prediction": {
                "error_family": predicted_family,
                "error_agent": predicted_agent,
                "error_step": predicted_step,
                "error_reason": result["analysis"]["error_reason"],
                "confidence": result["confidence"]
            },
            "metrics": {
                "agent_match": agent_match,
                "step_match": step_match,
                "verified": result["evidence"]["verified"],
                "ablation_gain": result["evidence"]["ablation_gain"]
            },
            "fix_suggested": result["hypothesis"].get("patch", {})
        }

        # Print progress (with thread safety)
        print(f"[Test {idx+1}/{total}] ID: {trace['id']}")
        print(f"  Predicted: {predicted_family} in {predicted_agent} at turns {predicted_step}")
        print(f"  Ground truth: {ground_truth['error_agent']} at step {ground_truth['error_step']}")
        print(f"  Agent match: {'✓' if agent_match else '✗'}, Step match: {'✓' if step_match else '✗'}")

        return trace_result

    except Exception as e:
        print(f"[Test {idx+1}/{total}] ID: {trace['id']}")
        print(f"  [Error] {str(e)}")
        return {
            "trace_id": trace["id"],
            "error": str(e),
            "metrics": {
                "agent_match": False,
                "step_match": False,
                "verified": False,
                "ablation_gain": 0.0
            }
        }


def evaluate_mas(traces, config_path, memory_path, run_manager, max_workers=3):
    """Evaluate MAS on test traces with parallel processing"""
    # Reset global shared MAS instance for this run
    global _shared_mas
    _shared_mas = None

    # Create memory directory for checkpoints
    memory_dir = run_manager.run_dir / "memory"
    memory_dir.mkdir(exist_ok=True)
    checkpoint_path = str(memory_dir / "checkpoint.epm.json")

    # Initialize shared MAS instance
    mas = get_shared_mas(config_path, memory_path)

    if memory_path:
        print(f"Loading learned memory from: {memory_path}")
        initial_stats = mas.memory.get_statistics()
        print(f"  Loaded {initial_stats['total_entries']} patterns")
        print(f"  Families: {initial_stats['families']}")
    else:
        print("No memory loaded - evaluating without prior knowledge")

    print(f"\n{'='*60}")
    print(f"Evaluating Simplified LLM-Based MAS (Parallel)")
    print(f"{'='*60}")
    print(f"Test traces: {len(traces)}")
    print(f"LLM Model: {mas.llm_config.model_id}")
    print(f"Max workers: {max_workers}")
    print(f"Memory checkpoints: {checkpoint_path}")
    print()

    results = []
    start_time = time.time()

    # Prepare arguments for worker function
    args_list = [
        (trace, idx, len(traces), config_path, memory_path, run_manager.run_id, checkpoint_path)
        for idx, trace in enumerate(traces)
    ]

    # Process traces in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(evaluate_single_trace, args): args[0]
            for args in args_list
        }

        # Collect results as they complete
        completed = 0
        for future in tqdm(as_completed(futures), total=len(traces), desc="Evaluating"):
            try:
                result = future.result()
                results.append(result)
                completed += 1

                # Print progress every 10 traces
                if completed % 10 == 0:
                    agent_correct = sum(1 for r in results if r.get("metrics", {}).get("agent_match", False))
                    step_correct = sum(1 for r in results if r.get("metrics", {}).get("step_match", False))

                    print(f"\n  Progress: {completed}/{len(traces)}")
                    print(f"  Agent accuracy: {agent_correct}/{completed} ({agent_correct/completed:.1%})")
                    print(f"  Step accuracy: {step_correct}/{completed} ({step_correct/completed:.1%})")

            except Exception as e:
                print(f"  [Worker Error] {str(e)}")
                results.append({
                    "trace_id": "unknown",
                    "error": str(e),
                    "metrics": {
                        "agent_match": False,
                        "step_match": False,
                        "verified": False,
                        "ablation_gain": 0.0
                    }
                })

    elapsed = time.time() - start_time

    # Final statistics
    total_traces = len(results)
    agent_correct = sum(1 for r in results if r.get("metrics", {}).get("agent_match", False))
    step_correct = sum(1 for r in results if r.get("metrics", {}).get("step_match", False))
    verified = sum(1 for r in results if r.get("metrics", {}).get("verified", False))
    avg_confidence = sum(r.get("prediction", {}).get("confidence", 0) for r in results) / total_traces
    avg_impact = sum(r.get("metrics", {}).get("ablation_gain", 0) for r in results) / total_traces

    print(f"\n{'='*60}")
    print(f"Evaluation Complete!")
    print(f"{'='*60}")
    print(f"Total traces: {total_traces}")
    print(f"Agent accuracy: {agent_correct}/{total_traces} ({agent_correct/total_traces:.1%})")
    print(f"Step accuracy: {step_correct}/{total_traces} ({step_correct/total_traces:.1%})")
    print(f"Verified: {verified} ({verified/total_traces:.1%})")
    print(f"Avg confidence: {avg_confidence:.3f}")
    print(f"Avg impact: {avg_impact:.3f}")
    print(f"Time: {elapsed:.1f}s ({elapsed/total_traces:.2f}s/trace)")
    print(f"Results: {run_manager.run_dir}")

    # Save results
    results_path = run_manager.run_dir / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            "summary": {
                "total_traces": total_traces,
                "agent_accuracy": agent_correct / total_traces,
                "step_accuracy": step_correct / total_traces,
                "verification_rate": verified / total_traces,
                "avg_confidence": avg_confidence,
                "avg_impact": avg_impact,
                "elapsed_time": elapsed
            },
            "results": results
        }, f, indent=2)

    # Save final memory
    final_memory_path = str(memory_dir / "final.epm.json")
    mas.save_memory(final_memory_path)

    # Print memory statistics
    memory_stats = mas.memory.get_statistics()
    print(f"\n{'='*60}")
    print(f"Memory Statistics")
    print(f"{'='*60}")
    print(f"Total patterns stored: {memory_stats['total_entries']}")
    print(f"Error families: {memory_stats['families']}")
    print(f"Avg confidence: {memory_stats['avg_confidence']:.3f}")
    print(f"Capacity used: {memory_stats['capacity_used']:.1%}")
    print(f"Final memory saved: {final_memory_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Simplified LLM-Based MAS")

    parser.add_argument("--test", required=True, help="Test traces (JSONL)")
    parser.add_argument("--memory", help="Learned memory file (epm.json)")
    parser.add_argument("--max-traces", type=int, help="Max test traces to use")
    parser.add_argument("--config", default="config.yaml", help="Config file")
    parser.add_argument("--output", default="runs", help="Output directory")
    parser.add_argument("--run-id", help="Custom run ID")
    parser.add_argument("--max-workers", type=int, default=3, help="Max parallel workers (default: 3)")

    args = parser.parse_args()

    # Load traces
    print(f"Loading test traces from {args.test}...")
    traces = load_traces(args.test, args.max_traces)
    print(f"Loaded {len(traces)} test traces")

    # Create run manager
    from utils import RunManager

    memory_tag = "with_memory" if args.memory else "no_memory"
    run_manager = RunManager(
        base_dir=args.output,
        run_id=args.run_id or f"eval_{memory_tag}",
        config={
            "test_file": args.test,
            "memory_file": args.memory,
            "max_traces": args.max_traces,
            "config_file": args.config,
            "max_workers": args.max_workers
        }
    )

    # Evaluate
    results = evaluate_mas(traces, args.config, args.memory, run_manager, max_workers=args.max_workers)

    print(f"\nResults saved: {run_manager.run_dir / 'evaluation_results.json'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
