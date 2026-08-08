# Pre-extracted error schemata — gpt-5-nano subset

These are the error schemata used by CORRECT to detect decisive errors in the
gpt-5-nano subset of CORRECT-Error (paper Table 2, lower half).

| Dataset       | # schemata |
|---------------|-----------:|
| arc           |        204 |
| gaia_level1   |         36 |
| hotpot        |        509 |
| math500       |         98 |
| mmlu_pro      |         68 |
| musique       |        294 |
| wikimqa       |        699 |
| **Total**     |  **1,908** |

Each `{dataset}/error_schemata.txt` follows the format produced by
`src/error_schema_generator.py` / `src/error_schema_generator_cloud.py`:

```
=== Schema for Error Log <N> ===
Generated Schema:
### Error Schema for Identifying Similar Errors in Future Conversations

#### 1. Error Signatures
- ...

#### 2. Error Context Analysis
- ...

#### 3. Detection Heuristics
- ...

==================================================
=== Schema for Error Log <N+1> ===
...
```

The three numbered sections correspond to the schema components in
paper §3.1: **Error Signatures (Σ)**, **Error Context Analysis (C)**,
and **Detection Heuristics (H)**.

## How these were generated

Author-supplied schemata, distilled by GPT-5 from the corresponding
training trajectories. See paper §3.1 and Algorithm 1, line 4.

## Regenerating

```bash
OPENAI_API_KEY=sk-... \
  bash scripts/generate_schemata_cloud.sh gpt-5 gpt5nano
```

## GPT-4o-mini subset

Schemata for the **gpt-4o-mini** subset of CORRECT-Error (paper Table 2
upper half) are in `data/schemata_correct_error_gpt4omini/`. Regenerate them
with `scripts/generate_schemata.sh` (local) or
`scripts/generate_schemata_cloud.sh gpt-4o gpt4omini` (cloud).
