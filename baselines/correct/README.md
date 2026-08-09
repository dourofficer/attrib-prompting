# CORRECT baseline

Reproduction of **CORRECT** (COndensed eRror RECognition via knowledge
Transfer; the paper is vendored at
[`vendored/CORRECT/correct.pdf`](../../vendored/CORRECT/correct.pdf)) — a
training-free, retrieval-based failure-attribution method. Instead of judging a
trajectory in isolation, an LLM detector is shown **error schemata** distilled
offline from *other* annotated failures and retrieved by trajectory similarity.

## The variant this repo implements

The vendored code has two prompt families: a local-vLLM path used for the
paper's CORRECT-Error table, and the **cloud path**
([`src/Lib/cloud_paper.py`](../../vendored/CORRECT/src/Lib/cloud_paper.py)) the
authors keep byte-identical to their paper runs — the only path ever used with
closed-source detectors. Per project decision this repo implements the **cloud
variant for all datasets and backends**: the "THOUGHT TEMPLATE FOR GUIDANCE"
schema injection, the cloud schema-generation prompt (step-numbered history +
a trailing block carrying the source trajectory's gold agent/step — retrieved
schemata deliberately show labeled exemplars), and the cloud unicode scrubbing.

## Pipeline

Three stages, all resumable, orchestrated by `sweep.py` (each stage is also a
directly-invocable module):

| stage | module | writes |
|---|---|---|
| 1. schemagen | `schemagen.py` — one LLM call per trajectory distills a schema from the gold labels | `outputs/<ds>/<subset>/<schema_model>/schemagen/<id>.json` |
| 2. similarity | `similarity.py` — BGE-M3 embeddings → ranked neighbour lists (self excluded) | `outputs/<ds>/<subset>/_similarities/bge-m3.json` (+ `.meta.json`) |
| 3. detection | `predict.py` — all-at-once prompt + top-k retrieved neighbour schemata | `.../<subset>/<model>/<method>/<id>.json` |

One `schema_model` per config distills the schemata; **all detectors share
them** (the paper's design — a strong generator, many detectors). Stage-1/2
artifacts are corpus-scoped and GT-independent; both detection settings reuse
them.

Methods: **`correct`** (schema-guided, needs stages 1–2) and
**`correct_baseline`** (the vendored k=0 baseline prompt, no artifacts) — the
paper's baseline rows, isolating the effect of the schemata.

Retrieved-schema count `num_schemata` (paper §A.3): Who&When
algorithm-generated **1**, hand-crafted **10**, CORRECT-Error **5**;
TraceElephant is not in the paper — we default to **10** (its trajectories are
long GAIA-style runs like hand-crafted).

## GT settings

The repo-wide GT axis (GUIDE.md) applies, with one inversion: **the vendored
cloud path never puts the task answer in the detection prompt**, so for this
baseline `--gt without` is the parity-tested paper setting **and the default**
(prompting is the opposite: its vendored prompt carries the answer). `--gt
with` inserts the vendored answer line `The Answer for the problem is: ...`
(bytes from the vendored local path) after the problem line. As everywhere:
with-GT detection lands under `outputs/`, without-GT mirrors into
`outputs-nogt/`.

Ground truth in **stage 1** is a corpus property, not a flag: the schemagen
prompt always interpolates `Ground Truth: {ground_truth}` (vendored, gold-
conditioned). On `data/correct-error` the restored answers fill that slot as
the paper's Fig. 9 intends; point `data_dir` at `data/correct-error-nogt` to
reproduce the as-released corpus where the slot renders blank.

## Running

```bash
# Full pipeline, one dataset (paper setting; local models):
DATASET=ww GPU=0 bash scripts/correct/run.sh

# One model / subset / stage:
DATASET=ww SUBSET=hand-crafted MODEL=gpt-4o bash scripts/correct/run.sh      # needs OPENAI_API_KEY
DATASET=correct-error STAGES=schemagen MODEL=gpt-5 bash scripts/correct/run.sh
GT=with DATASET=ww MODEL=qwen3.5-9b bash scripts/correct/run.sh              # with-GT extension

# Or the sweep directly:
python -m baselines.correct.sweep --config baselines/correct/configs/ww.yaml [--dry-run]
```

Everything is idempotent per trajectory (file existence = resume ledger). API
detectors can consume locally-generated schemata and vice versa: run the
schemagen stage from one config, name the same `schema_model:` in the other.

## Evaluation

Shared report, correct methods, per-seed splits (ww/te seeds 1–20, ce 1–3):

```bash
python -m baselines.correct.report --config baselines/correct/configs/report_ww.yaml [--check-only]
# --gt with evaluates the with-GT tree; default (without) reads outputs-nogt/.
```

## Faithfulness notes (the details that bite)

- **Two base prompts, three byte diffs.** The k=0 baseline prompt
  (`cloud_paper.py:173-192`) differs from the schema-guided base
  (`cloud_paper.py:364-375`): no space before the newline after the problem, a
  triple-quoted *indented* JSON example, and the tail `Reason for Mistake: \n`
  vs `(Your reason)\n`. Both are reproduced verbatim.
- **Scrub asymmetry.** The schema-guided path ASCII-scrubs user *and* system
  prompt with `clean_text` (every non-ASCII char → space, including the `•`
  bullets of the injection block); the baseline path only maps smart
  quotes/dashes (`_clean_unicode_content`). We apply each at message-build
  time — our backends send messages verbatim, and the vendored call-time
  `_clean_unicode_content` is a no-op on already-scrubbed text.
- **Retrieval is the Who&When top-k slice** (`inference_whoandwhen.py:222-283`):
  `similar_indices[:k]`, keep only neighbours that have schemata — *silently
  fewer* than k, empty for unknown ids, no random fallback. Self never appears
  (stage 2 drops self-similarity). The CE variant's scan-until-filled loop is
  not adapted.
- **Schemata carry gold labels.** The cloud generator prompt ends with a format
  block containing the source trajectory's gold `Agent Name:`/`Step Number:` —
  retrieved schemata show other trajectories' answers by design. Leakage of the
  *query's* label is prevented only by self-exclusion.
- **Similarity ties follow file order.** The vendored ranking visits files in
  lexicographic `listdir` order and Python's stable sort preserves it for exact
  ties — do not "fix" to numeric order.
- **Sampling.** Local detection is greedy (`temperature 0.0/top_p 1.0`, the
  vendored CORRECT inference defaults); schemagen uses the vendored `0.7/0.95/
  1024`. API specs send exactly their declared `params` — the vendored cloud
  path sets `max_tokens` only (8192 in the runner script; never temperature),
  and exports `OPENAI_REASONING_EFFORT=medium` for gpt-5 (declared as a param
  in our api configs).

## Deliberate deviations (all infrastructure-level; prompts/decisions verbatim)

1. **Agent key is `role`, always.** The vendored autodetect (use `name` if the
   first entry has it) targeted the *original* Who&When layout; this repo's
   algorithm-generated data swaps the fields (`role` = agent name, `name` =
   `user`/`assistant`), so `role` reproduces what the vendored code yields on
   the original data. We also do not replicate the `is_handcrafted="False"`
   truthiness bug that made the paper's cloud runs label algorithm-generated
   turns `user:`/`assistant:`.
2. **Schemata as per-trajectory JSONs** keyed by trajectory id, instead of one
   `error_schemata.txt` whose 1-based enumeration must coincide with file
   numbering (silently mis-keys retrieval if any file is skipped). Schema text
   bytes are unchanged; resume comes free.
3. **`strip_think`** on schema text and before parsing, so local reasoning
   backbones work (the vendored GPT outputs have no think blocks).
4. **Parsing shared with prompting** (`parse_all_at_once`): the vendored
   `CORRECT/src/evaluate.py` regexes are the identical family; prompting's
   paren/markdown tolerance applies uniformly across baselines.
5. **Retries/concurrency from the shared backends** (the vendored code has no
   retry logic — a failed call is a lost prediction); batching/threading via
   the shared runner instead of the vendored 10-file batch windows and sleeps.
   Neither changes prompt bytes.
6. **Evaluation via the shared report** (agent@1 + step@1 on per-seed splits)
   instead of the vendored stdout-log + `evaluate.py` (step accuracy only,
   whole-corpus). The vendored `±tolerance` Acc@k metric is not reproduced.

## Tests

`tests/test_correct_{parity,methods,similarity,pipeline}.py` — CPU-only,
keyless. The parity tests drive the vendored modules themselves (fake OpenAI
client, so the vendored scrubbing runs for real) and assert byte-identical
messages, injection branches, schemagen prompts, retrieval decisions and
similarity rankings; the pipeline tests run every stage end-to-end on the dummy
backend, including resume and the report integration.
