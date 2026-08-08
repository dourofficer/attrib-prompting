## 1. Project Introduction & Overview

This project is an implementation and engineering realization of the paper **“From Flat Logs to Causal Graphs: Hierarchical Failure Attribution for LLM-based Multi-Agent Systems”**. Its core objective is: when an **LLM-based multi-agent system (MAS)** fails in real-world tasks, to automatically identify **who (which agent) and when (which step)** introduced the decisive error, and to distinguish it from downstream “symptoms” caused by error propagation.

### Background: Why MAS failures are hard to “find the real culprit”

LLM-driven MAS are powerful but fragile: prior studies report failure rates as high as **86.7%** on complex tasks. More importantly, failures often propagate through tool calls, environment feedback, inter-agent communication, and data dependencies, making execution logs appear as a pile of “chronologically ordered fragments” rather than a clear causal chain.
 The paper summarizes the diagnostic challenges into three aspects: **opaque causal flows**, **lack of intermediate supervision (only final success or failure is observable)**, and **ambiguous responsibility boundaries (where an error manifests ≠ where it is introduced)**.

### CHIEF: From “flat logs” to a “diagnosable structure”

To address these challenges, CHIEF no longer treats logs as flat text. Instead, it reformulates failure attribution as a **structured, traceable divide-and-conquer process**:

- First, reconstruct chaotic trajectories into a **Hierarchical Causal Graph (HCG)**;
- Then, perform top-down backtracking using **hierarchical virtual oracles** to rapidly narrow down suspect regions;
- Finally, apply **counterfactual attribution** to rigorously distinguish true root causes from propagated symptoms.

### Evaluation benchmark

The paper evaluates CHIEF on **Who&When** (currently a public benchmark for MAS failure attribution). The dataset contains **184** failure logs, including **126** algorithm-generated instances (from diverse CaptainAgent architectures) and **58** hand-crafted instances (from Magnetic-One), with ground truth ensured via multi-round expert consensus annotation.
 On this benchmark, CHIEF consistently outperforms 8 baseline methods in both agent-level and step-level attribution accuracy.

------

## 2. Feature Overview

### 2.1 Overall Pipeline

<img src="src/overview.png" width="100%">

As illustrated, CHIEF decomposes failure attribution into three stages: (1) hierarchical causal graph construction; (2) hierarchical oracle-guided backtracking; and (3) counterfactual attribution, progressively locating the root cause via a divide-and-conquer strategy.

### 2.2 Core Capabilities

- **Hierarchical Causal Graph Construction (HCG)**: Parses flat trajectories into a “subtask–agent–step” structure and explicitly models inter-step data dependencies and interactions, providing the structural foundation for subsequent diagnosis.
- **Hierarchical Oracle-Guided Backtracking**: Uses synthesized virtual oracles to verify subtasks or stages, replacing linear step-by-step scanning with a top-down search that quickly converges on suspicious error nodes.
- **Counterfactual Attribution**: Applies progressive causal screening (e.g., scope, dependency type, reversibility) to rigorously distinguish true root causes from propagated symptoms.

### 2.3 Core Results

> Table format: each cell is reported as **w/ 𝒢 / w/o 𝒢**, where **𝒢 indicates access to task ground truth (correct outcome)**; FAMAS only reports w/ 𝒢, so missing entries are marked as “–”.

#### (A) Main Results: Who&When Attribution Accuracy (%)

| Type           | Method        | Hand-Crafted Agent ↑ | Hand-Crafted Step ↑ | Alg.-Generated Agent ↑ | Alg.-Generated Step ↑ |
| -------------- | ------------- | -------------------- | ------------------- | ---------------------- | --------------------- |
| Heuristic      | Random        | 12.00 / 12.00        | 4.20 / 4.20         | 29.10 / 29.10          | 19.10 / 19.10         |
| LLM Prompting  | All-at-Once   | 50.00 / 48.28        | 5.17 / 5.17         | 61.11 / 59.52          | 13.49 / 15.87         |
| LLM Prompting  | Step-by-Step  | 36.00 / 34.30        | 6.60 / 6.90         | 39.70 / 28.30          | 27.40 / 17.80         |
| LLM Prompting  | Binary Search | 51.70 / 36.20        | 6.90 / 6.90         | 44.10 / 30.10          | 24.00 / 16.60         |
| LLM Prompting  | ECHO          | 68.40 / 67.90        | 28.10 / 26.80       | 68.80 / 67.20          | 28.80 / 27.20         |
| Spectrum-based | FAMAS         | 62.07 / –            | 41.38 / –           | 55.56 / –              | 23.81 / –             |
| Fine-tuning    | AgenTracer    | 69.10 / 63.82        | 20.70 / 20.68       | 69.62 / 63.73          | 42.90 / 37.30         |
| Fine-tuning    | GraphTracer   | 74.91 / 69.74        | 28.63 / 27.97       | 76.64 / 67.42          | 49.97 / 44.35         |
| **Ours**       | **CHIEF**     | **77.59 / 72.41**    | **29.31 / 29.31**   | **76.80 / 68.80**      | **52.00 / 45.60**     |

#### (B) Cost: Average Token Consumption (w/ 𝒢, lower is better)

| Method        | Hand-Crafted ↓ | Alg.-Generated ↓ |
| ------------- | -------------- | ---------------- |
| All-at-Once   | 21,581         | 5,833            |
| Step-by-Step  | 87,720         | 6,533            |
| Binary Search | 34,659         | 5,226            |
| FAMAS         | 431,620        | 116,660          |
| ECHO          | 53,701         | 25,642           |
| **CHIEF**     | **55,085**     | **19,504**       |

## 3. Environment & Dependencies

### 3.1 Python and Dependency Installation

- Python **3.10+** is recommended.
- Dependencies are pinned in `requirements.txt`:

```
pip install -r requirements.txt
```

> Note: `sentence-transformers` will indirectly install heavy dependencies such as `torch` and `transformers`; a longer installation time is expected on first setup.

### 3.2 API Configuration (.env)

This project uses the **OpenAI Python SDK (compatible with OpenAI-style Chat Completions)**. You need to provide your configuration in `CHIEF/.env`:

```
OPENAI_BASE_URL = "your_url"
OPENAI_API_KEY = "your_key"
```

### 3.3 RAG Resource Preparation (Important)

`CHIEF.py` loads the RAG retriever at startup and expects `./index` and `./kb` in the current directory.
 The repository already provides pre-built resources located at `rag/index` and `rag/kb`.

------

## 4. Quick Start

### 4.1 Running the CHIEF Pipeline

```
cd CHIEF
python CHIEF.py --data_dir data/Hand-Crafted --model deepseek-chat
```

You can also switch to the algorithm-generated subset:

```
python CHIEF.py --data_dir data/Algorithm-Generated --model deepseek-chat
```

### 4.2 Output Location

After execution, results will be generated under `results/CHIEF/`:

- `results_CHIEF_<model>_<dataset>_<time>.jsonl`: per-sample results (useful for inspection and debugging)
- `results_CHIEF_<model>_<dataset>_<time>_summary.json`: aggregated accuracy and detailed lists

### 4.3 Debug Mode Notice

At the top of `CHIEF.py`, the defaults are:

- `DEBUG_MODE = True`
- `DEBUG_SAMPLE_LIMIT = 1`

This means **only one sample is executed by default** for quick sanity checks.
 To run the full dataset, set `DEBUG_MODE` to `False`.

------

## 5. Baseline Mode

All baseline scripts are located under `baseline_method/` and can be run directly for comparison (they also rely on `.env` for API configuration).

> Common parameters:
>
> - `--model`: model name (e.g., `deepseek-chat / gpt-4o-mini / ...`)
> - `--data_dir`: dataset directory (`data/Hand-Crafted` or `data/Algorithm-Generated`)

### 5.1 One-shot Baseline (Direct attribution)

```
cd CHIEF
python baseline_method/baseline.py --model deepseek-chat --data_dir data/Hand-Crafted
```

Outputs are saved under `results/baseline/`.

### 5.2 All-at-Once

```
python baseline_method/allatonce.py --model deepseek-chat --data_dir data/Hand-Crafted
```

Outputs are saved under `results/v1/`.

### 5.3 Step-by-Step

```
python baseline_method/stepbystep.py --model deepseek-chat --data_dir data/Hand-Crafted
```

Outputs are saved under `results/stepbystep/`.

### 5.4 Binary Search

```
python baseline_method/binarysearch.py --model deepseek-chat --data_dir data/Hand-Crafted
```

Outputs are saved under `results/binarysearch/`.

------

## 6. Project Structure

```
CHIEF/
├─ CHIEF.py
├─ requirements.txt
├─ .env
├─ __init__.py
├─ baseline_method/
│  ├─ baseline.py
│  ├─ allatonce.py
│  ├─ stepbystep.py
│  └─ binarysearch.py
├─ rag/
│  ├─ rag_search.py
│  ├─ build_gaia_kb.py
│  ├─ build_assistantbench_kb_faiss.py
│  ├─ index/            # Pre-built FAISS index (to be copied to ./index)
│  ├─ kb/               # Pre-built KB JSON files (to be copied to ./kb)
│  └─ data/             # Raw data for building RAG resources (usually not needed)
├─ data/                # Who&When dataset (not detailed in README)
│  ├─ Algorithm-Generated/
│  └─ Hand-Crafted/
└─ results/             # Execution outputs (reproducible experiment logs)
   └─ CHIEF/
```

- `CHIEF.py`: main entry point (CHIEF mode), reads samples from `--data_dir` and outputs attribution results and summaries.
- `baseline_method/`: baseline method scripts for comparison.
- `rag/`: RAG-related components and resources (index, KB, and build scripts).
- `data/`: dataset directory (intentionally not expanded in the README).
- `results/`: default output directory, organized by method/script.