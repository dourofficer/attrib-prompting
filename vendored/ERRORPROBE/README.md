# Self-Improving Error Diagnosis in Multi-Agent Systems

An unofficial, minimal reference implementation of the method described in the paper
*"Towards Self-Improving Error Diagnosis in Multi-Agent Systems"*.

> **Note**: Due to institutional policy, the original source code for the paper cannot be
> released. This repository is an independent, anonymized reproduction of the core method,
> provided to help the community reproduce and build on the ideas in the paper. Exact
> numbers may differ slightly from those reported in the paper.

## Overview

Given a **failed execution trace** of an LLM multi-agent system, the goal is to
automatically diagnose the failure:

- **Who** — which agent is responsible for the failure,
- **When** — at which step the decisive error occurred,
- **Why** — the failure mode (following the MAST taxonomy of 14 multi-agent failure modes)
  and a natural-language explanation.

Unlike single-shot LLM-judge baselines, the system **self-improves over time**: verified
diagnoses are distilled into a persistent **Error Pattern Memory (EPM)** that guides future
diagnoses, producing a learning curve rather than flat i.i.d. performance.

## Architecture

```
Failed trace
    │
    ▼
┌─────────────────────┐   retrieves similar patterns
│  Analyzer (LLM)     │◄─────────────────────────────┐
│  locates the error, │                              │
│  classifies failure │                              │
│  mode (MAST)        │                              │
└─────────┬───────────┘                              │
          ▼                                          │
┌─────────────────────┐                    ┌─────────┴─────────┐
│  Verifier (LLM)     │  verified          │ Error Pattern     │
│  checks hypothesis, │  hypotheses        │ Memory (EPM)      │
│  estimates impact,  ├───────────────────►│ VBW write gate +  │
│  suggests a fix     │                    │ RFI-Δ scoring     │
└─────────────────────┘                    └───────────────────┘
```

Three components:

1. **Analyzer agent (LLM)** — reads the trace and produces an error hypothesis
   (responsible agent, error step span, MAST failure mode, rationale). When memory is
   available, the top-scoring similar patterns are injected as guidance.
2. **Verifier agent (LLM)** — independently checks the hypothesis, estimates the impact
   if the error were fixed (counterfactual ablation gain), and proposes a fix
   (guard / rewrite / prompt hint).
3. **Error Pattern Memory (algorithmic, no LLM)** — long-term memory of verified error
   patterns with:
   - **VBW (Verified-Before-Write) gate**: only hypotheses that pass verification with
     sufficient ablation gain / evidence are written to memory;
   - **RFI-Δ scoring**: entries are ranked by Recency × Frequency × Impact for retrieval
     and eviction, with time-based recency decay.

An optional **backward tracing** mode (`backward_tracing.enabled: true` in
`config.yaml`) starts from the failure symptom and walks backwards through the full
trace to the root cause, instead of analyzing a truncated recent window.

## Installation

```bash
pip install -r requirements.txt
```

LLM calls go through [LiteLLM](https://docs.litellm.ai/), so any supported provider
works. Set your provider's credentials as environment variables, e.g.:

```bash
export ANTHROPIC_API_KEY=...   # or OPENAI_API_KEY=..., etc.
```

and set `model.model_id` in `config.yaml` accordingly.

## Data format

Traces are JSONL, one failed trace per line, following the
[Who&When](https://huggingface.co/datasets/Kevin355/Who_and_When) benchmark format:

```json
{
  "id": "trace_0001",
  "question": "the task given to the multi-agent system",
  "history": [
    {"name": "AgentName", "role": "assistant", "content": "agent output ..."}
  ],
  "mistake_agent": "AgentName",
  "mistake_step": "3",
  "mistake_reason": "natural-language explanation of the decisive error"
}
```

The `mistake_*` fields are the ground-truth annotations (used for training-time
verification and evaluation). See `data/sample_traces.jsonl` for two toy examples.
For real experiments, download the Who&When dataset from Hugging Face and convert each
split to JSONL in the format above.

## Usage

### 1. Training (build the memory)

Run the system over training traces; verified error patterns are accumulated into
the EPM:

```bash
python train.py \
  --train data/train_traces.jsonl \
  --config config.yaml \
  --output runs \
  --max-traces 100
```

The learned memory is saved to `runs/<run_id>/memory/epm.json`.

### 2. Evaluation

Evaluate on held-out traces, with or without the learned memory:

```bash
# With memory (self-improving)
python evaluate.py \
  --test data/test_traces.jsonl \
  --memory runs/<run_id>/memory/epm.json \
  --config config.yaml

# Without memory (baseline ablation)
python evaluate.py \
  --test data/test_traces.jsonl \
  --config config.yaml
```

Reported metrics:

- **Agent accuracy** — predicted responsible agent matches the annotation,
- **Step accuracy** — annotated decisive step falls within the predicted span,
- verification rate, average confidence, and average estimated impact.

## Repository layout

```
├── train.py                  # build the error-pattern memory from training traces
├── evaluate.py               # evaluate on test traces (with/without memory)
├── config.yaml               # model + system configuration
├── simplified_mas/           # the multi-agent diagnosis system
│   ├── llm_agents.py         #   Analyzer + Verifier agents, system orchestration
│   ├── backward_tracer.py    #   optional backward tracing from symptom to root cause
│   ├── trace_index.py        #   full-trace indexing
│   ├── error_tracing_memory.py  # working memory used during backward tracing
│   └── failure_mode_detector.py # MAST failure-mode heuristics
├── core/                     # Error Pattern Memory (EPM)
│   ├── epm_schema.py         #   schema + MAST taxonomy definitions
│   └── epm_manager.py        #   VBW gate, RFI-Δ scoring, eviction, persistence
├── utils/                    # metrics, run management, trace utilities
└── data/sample_traces.jsonl  # toy example traces
```

