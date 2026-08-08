# CORRECT-Error Dataset

The CORRECT-Error benchmark is hosted on Hugging Face:

**https://huggingface.co/datasets/yifanyu/CORRECT-Error**

## Download

To run the inference scripts, fetch the per-trajectory JSON layout they read:

```bash
bash scripts/download_data.sh
```

To just inspect the records (not for running inference), load it directly with
the `datasets` library — see [Loading the dataset](../README.md#loading-the-dataset)
in the root README.

## Contents

- **2,226 error-injected multi-agent trajectories** across 7 tasks
  (GAIA, HotpotQA, Musique, WikiMQA, ARC, Math500, MMLU-Pro).
- Two generator-model variants per task: `gpt-4o-mini` and `gpt-5-nano`.
- Each record carries step-level decisive-error labels:
  `mistake_agent`, `mistake_step`, `mistake_reason`.

## Per-record schema

```json
{
  "trajectory_id":   "gpt-4o-mini_hotpot_task12_2",
  "dataset":         "hotpot",
  "generator_model": "gpt-4o-mini",
  "question_id":     "task12_2",
  "question":        "...",
  "groundtruth":     "...",
  "history":         [{"role": "Planner", "content": "..."}, ...],
  "mistake_agent":   "Planner",
  "mistake_step":    5,
  "mistake_reason":  "...",
  "is_corrected":    false,
  "gaia_level":      null,
  "level":           0
}
```

`dataset` is one of `arc`, `hotpot`, `musique`, `wikimqa`, `math500`, `mmlu_pro`,
`gaia`, and `trajectory_id` is `{generator_model}_{dataset}_{question_id}`.
`gaia_level` is `1` for the GAIA gpt-5-nano subset (the GAIA Level-1 split) and
`null` otherwise; `level` is a source-side metadata field (`0` for all records).

## Local layout after download

`scripts/download_data.sh` restores the per-trajectory JSON layout that the
inference scripts read:

```
data/correct_error/{dataset}/individual_trajectories/{N}.json            # gpt-4o-mini
data/correct_error_gpt5nano/{dataset}/individual_trajectories/{N}.json   # gpt-5-nano
```

These directories hold the downloaded benchmark trajectories; they are
fetched on demand and are not part of the repo.

## Pre-extracted error schemata

Pre-extracted schemata for both benchmarks are included.

- **CORRECT-Error**, both Table 2 splits:
  - `data/schemata_correct_error_gpt5nano/{dataset}/error_schemata.txt` —
    GPT-5-Nano split (1,908 schemata). See
    [`schemata_correct_error_gpt5nano/README.md`](schemata_correct_error_gpt5nano/README.md).
  - `data/schemata_correct_error_gpt4omini/{dataset}/error_schemata.txt` —
    GPT-4o-mini split (318 schemata).

- **Who&When** — `data/schemata_whoandwhen/{subset}/error_schemata.txt`
  (184 schemata), covers paper Table 1.
  Generator: GPT-5 (paper §A.3 canonical). See
  [`schemata_whoandwhen/README.md`](schemata_whoandwhen/README.md).

## Pre-computed trajectory similarities

The repo also ships the exact retrieval mappings used in the paper runs, so
similarity-based schema retrieval works out of the box (no recompute needed).
Each file maps a trajectory index to its ranked list of nearest neighbours.

- `data/similarities_gpt5nano/{dataset}_trajectory_similarities.json` —
  CORRECT-Error gpt-5-nano split
  (used by `scripts/run_inference_correct_error.sh`, the default `SPLIT=gpt5nano`)
- `data/similarities/{dataset}_trajectory_similarities.json` —
  CORRECT-Error gpt-4o-mini split
  (used by `scripts/run_inference_correct_error.sh SPLIT=gpt4omini`)
- `data/similarities_whoandwhen/{subset}_trajectory_similarities.json` —
  Who&When (used by `scripts/run_inference_whoandwhen.sh`)

Regenerate any of these with `scripts/generate_similarities*.sh`
(code: `src/generate_trajectory_similarities.py`) only if you change the
underlying trajectory corpus.
