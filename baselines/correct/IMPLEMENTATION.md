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
2. **Schemata as per-trajectory JSONs** keyed by trajectory id
   (`artifacts/<ds>/<subset>/schemagen/<schema_model>/<id>.json`), instead of one
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