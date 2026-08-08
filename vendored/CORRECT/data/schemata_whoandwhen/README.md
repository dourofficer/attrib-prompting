# Pre-extracted error schemata — Who&When

These are the error schemata used by CORRECT to detect decisive errors on
the Who&When benchmark (Zhang et al., ICML 2025; paper Table 1).

```
Algorithm-Generated/error_schemata.txt   (126 schemata)
Hand-Crafted/error_schemata.txt          ( 58 schemata)
```

Generator: **GPT-5** (cloud), per paper §A.3 — *"we first generate all the
error schemata using GPT-5 model"*.

Each `error_schemata.txt` follows the same format as
`data/schemata_correct_error_gpt5nano/` (see that directory's README for details):

```
=== Schema for Error Log <N> ===
Generated Schema:
... three sections: Error Signatures (Σ), Error Context Analysis (C),
    Detection Heuristics (H) ...
==================================================
```

## Regenerating

These schemata are included. They were generated with GPT-5 via
`src/error_schema_generator_cloud.py` (the `generate_schemata_cloud.sh`
wrapper only covers the CORRECT-Error splits). The exact prompt is
reproduced in paper Appendix Figure 9.
