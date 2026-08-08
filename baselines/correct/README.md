# CORRECT baseline (local re-implementation)

Re-implementation of **CORRECT** (*Condensed Error Recognition via Knowledge
Transfer in Multi-agent Systems*, ICML 2026 — vendored at
[`baselines/CORRECT/`](../CORRECT)) adapted to this repo's models, datasets and
conventions. Standalone: no imports from `baselines/CORRECT` (only from
`baselines/prompting`, like the chief baseline).

CORRECT is training-free schema-guided error localization, in three stages:

1. **Schema generation** (`schemagen.py`, vLLM) — for every annotated
   trajectory, an LLM distills the gold error (agent / step / reason + full
   conversation) into a reusable "error schema"; one vendored-format
   `error_schemata.txt` per (model, subset).
2. **Trajectory similarities** (`similarity.py`, HF transformers) — BGE-M3
   embeddings of each trajectory, cosine-ranked neighbour lists with self
   excluded (the leave-one-out mask: a trajectory never sees its own schema).
3. **Schema-guided detection** (`predict.py`, vLLM) — the top-k most-similar
   trajectories' schemata are appended to the vendored all-at-once prompt; the
   model outputs `Agent Name:` / `Step Number:`, parsed with the vendored
   regexes into the house predictions JSONL.

## Running

```bash
# everything: (qwen3.5-9b, deepseek-8b) × (ww, traceelephant, correct-error)
GPU=0 bash baselines/correct/scripts/run_correct.sh

# subsets of the grid (env knobs, matching the SVD/CRR scripts)
GPU=0 MODELS="qwen3.5-9b" DATASETS="ww" bash baselines/correct/scripts/run_correct.sh
GPU=0 DRY_RUN=1 bash baselines/correct/scripts/run_correct.sh              # preview
GPU=0 EXTRA_SET="--set overwrite=true" bash baselines/correct/scripts/run_correct.sh

# one dataset by hand
python -m baselines.correct.sweep --config baselines/correct/configs/ww.yaml [--dry-run]

# report (completion check + per-seed val/test tables next to SVD/CRR)
python -m baselines.correct.report --config baselines/correct/configs/report_ww.yaml
```

All stages are idempotent (existing outputs are skipped unless
`--set overwrite=true`). Outputs land under `outputs-<ds>/correct/`:

```
outputs-<ds>/correct/
├── schemata/<model>/<subset>/error_schemata.txt      # stage 1 (per model)
├── similarities/<subset>_trajectory_similarities.json # stage 2 (model-independent)
└── <model>/<subset>/predictions_method-correct.jsonl  # stage 3
```

**Prerequisite:** the embedding model must be available at
`../hub/BAAI/bge-m3` (or point `embed_model_path` elsewhere / at the HF id).

## Evaluation protocol & data use (read before comparing)

CORRECT is **transductive over gold labels** — the faithful reproduction of the
paper's protocol, but a different supervision regime than the other baselines:

- **Every trajectory is evaluated, and every trajectory supervises the
  others.** No data is set aside for schema generation: stage 1 distills a
  schema from *each* trajectory's gold annotations (`mistake_agent`,
  `mistake_step`, `mistake_reason`, plus question, gold answer, and the full
  conversation), over the whole subset.
- **The retrieval pool for a trajectory `t` is the entire subset minus `t`
  itself** (self is excluded by construction of the similarity lists — the
  paper's only masking: "we mask each trajectory itself and avoid receiving
  its own error schema"). The pipeline is completely split-agnostic: the
  train/val/test split exists only in `report.py`, which selects which
  prediction *rows* are averaged per seed. So a test trajectory's retrieved
  schemata can come from train, val, **or other test trajectories** of that
  same seed.
- **The query trajectory's own prompt contains no gold information about
  itself** — no labels and, unlike the ww/traceelephant prompting/chief
  prompts, no ground-truth answer either (`gt_in_prompt: false` everywhere).

Fairness implications when reading the tables:

- prompting / chief never see any gold error labels (though their ww and
  traceelephant prompts include the gold *answer*, which CORRECT's do not);
  SVD/CRR fits its reference unsupervised on train and touches labels only via
  val-based config selection. CORRECT, by contrast, answers "how well does
  schema transfer work *given an annotated corpus of the other failures*" —
  strictly more supervision, per its published protocol. Footnote this when
  comparing.
- A split-respecting variant (filter `similarities[t]` to train(+val) ids
  before the top-k in `predict.py`) is easy to add but is deliberately **not**
  implemented: it would deviate from the vendored protocol and make
  predictions per-seed (one inference pass per seed instead of one).

## Setup choices

- **Schema generator = the detector backbone** (per-model schema caches). The
  paper uses a strong external generator (GPT-5 / Qwen2.5-72B); here each
  (model, dataset) cell is fully self-contained, consistent with how chief runs
  all of its stages with one backbone.
- **k (retrieved schemata), per the paper where available**: Who&When
  algorithm-generated **1**, hand-crafted **10**, CORRECT-Error **5**;
  TraceElephant is not in the paper → **10** (the hand-crafted setting; long
  trajectories of comparable style). Config axis `num_schemata` (int or
  per-subset map).
- `--num_schemata 0` runs the vendored no-schema LLM-as-a-Judge baseline
  (method name `correct-base`). Note it is *not* the same measurement as the
  prompting baseline's `all_at_once`: CORRECT's vLLM prompt excludes the gold
  answer and does not number the steps. Supported but not run by default.

## Faithfulness to the vendored implementation

Both vendored inference scripts (`inference_whoandwhen.py`,
`inference_correct_error.py`) route open models through the **same vLLM code
path** (`Lib/local_model.py`), which is what this package replicates:

- The base all-at-once prompt, the schema-injection wording
  ("Here's a error schema…" / "Here are error schemata…" + "You can neglect
  it…"), the system prompts, the schema-generation prompt, the greedy sampling
  (`temperature=0.0, top_p=1.0, max_tokens=1024`; schema-gen
  `0.7/0.95/1024`), the "Agent Name:"-block response trim, and the
  `evaluate.py` parsing regexes are all **verbatim**. In this path the active
  prompt **excludes the gold answer** and does **not** number the conversation
  steps (both variants are commented out in the vendored source; the paper
  confirms the answer is excluded).
- Retrieval: both vendored variants are implemented behind
  `scan_until_filled` — `false` = Who&When script (inspect only the top-k
  neighbours), `true` = CORRECT-Error script (scan until k schemata found,
  capped at 5k checked). With complete per-model schema caches (ours) the two
  are identical.
- Schema #n ↔ trajectory `n.json` alignment relies on the numeric filename
  sort shared by all three stages, as vendored.

A parity test (`tests/test_parity.py`) drives the **vendored functions
end-to-end** (vllm stubbed, fake capture classes monkeypatched in) on identical
dummy trajectories/schemata and asserts: byte-identical templated prompts
(base, single-/multi-schema injected, no-schema fallback, schema-gen),
identical response trims, deep-equal schemata-file round-trips through both
vendored loaders, deep-equal retrievals from both vendored analyzers (complete
and holey caches), parse agreement with the vendored `evaluate.py`, and
identical similarity rankings through both codepaths with a deterministic fake
encoder. Run from the repo root:

```bash
python -m pytest baselines/correct/tests/test_parity.py -q   # or
python -m baselines.correct.tests.test_parity
```

Real-data cross-check: our regenerated `ww/algorithm-generated` similarities
agree with the author-shipped
`baselines/CORRECT/data/similarities_whoandwhen/…` file on 90/126 top-1
neighbours. Full rankings differ because the cosine scores are near-ties
(median top1–top2 gap ≈ 0.007; mean top-1 sim ≈ 0.90), so environment-level
numerics (library/kernel versions, GPU, the authors' exact data copy) reorder
them — the ranking *code* is parity-tested as identical. We always use our own
regenerated artifacts, never the shipped ones, so schema/similarity indices are
self-consistent by construction.

### Deliberate deviations (infrastructure only)

1. **vLLM via `PromptEngine`** (reused from `baselines/prompting`) instead of a
   raw `LLM(...)` per stage — same single batched `.chat()` call, house
   dtype/seed/chat-template handling. The vendored print-batching loop in the
   schema generator (batches of 32) is likewise one batched call — identical
   results, sampling is per-prompt.
2. **No YaRN `rope_scaling`** — the vendored `factor=4` (inference) /
   `factor=10` (schema-gen) hack worked around Qwen2.5's 32k context;
   qwen3.5-9b and deepseek-8b natively support ≥128k (`max_model_len: 131072`,
   as chief/prompting).
3. **Reasoning-model handling** (same fixes as chief/prompting; deepseek-8b
   always emits `<think>`):
   `strip_think` (hardened variant) is applied **before** the vendored
   "Agent Name:" trim at inference and **before writing schemata** (a schema
   with an embedded reasoning trace would pollute every future retrieval
   prompt); `enable_thinking` is a config toggle through the adapter's chat
   template; the run script bumps deepseek to `gen_max_tokens=8192` for *both*
   vLLM stages (thinking + answer share the budget — the vendored 1024 would
   truncate mid-think); the corrected deepseek tokenizer dir
   (`baselines/prompting/tokenizers/deepseek-8b`) is passed via
   `tokenizer_paths`; parsing strips markdown decoration first (deepseek bolds
   `**Agent Name:**`) — a no-op on vendored-style outputs.
4. **Agent identity is `history[t]["role"]`** for every dataset — this is what
   the vendored `_agent_label(entry, "role")` reads, and the `name`-preferred
   branch falls back to `role` anyway on this repo's data (parity-tested under
   both `is_handcrafted` flags).
5. **`ground_truth` key in schema-gen** — the vendored generator reads the raw
   CORRECT-Error key `groundtruth`; this repo's normalized data stores
   `ground_truth` (empty string on correct-error, populated elsewhere — the
   inference prompt never contains it either way).
6. **Predictions as JSONL at write time** (house format consumed by the shared
   report) instead of stdout logs + a separate `evaluate.py`; the regexes are
   verbatim and applied to the vendored-style trimmed response, and `raw` is
   stored untouched so rows can be re-parsed.
7. **Schema blocks keyed by numeric filename** instead of a 1-based enumeration
   position. Byte-identical on this repo's data (contiguous `1.json…N.json`,
   all valid); keying by filename simply removes the misalignment the vendored
   enumeration would cause if a file were skipped, since retrieval looks
   schemata up by file number.

## Conventions that bite

- Run everything **from the repo root** (`python -m baselines.correct.…`);
  model paths in the configs are `../hub/...` relative to it.
- The three stages must see the **same subset directory** — the schema cache
  and neighbour lists are keyed by numeric filename, so regenerating a subset's
  data invalidates both artifacts (delete `outputs-<ds>/correct/schemata/…`
  and `…/similarities/…` and rerun).
- Per-model schema caches mean stage 1 runs once per (model, subset); the
  similarity JSON is model-independent and computed once per subset.
- The report reuses `baselines/prompting/report.py` (`methods: [correct]`),
  which reproduces the CRR/SVD per-seed val/test splits byte-identically —
  step@1 compares ints with no offset (0-based everywhere), agent@1 is the
  `standardize_role`-lowered substring rule.
