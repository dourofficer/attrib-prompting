"""Method-agnostic infrastructure shared by all baselines (prompting, chief, correct).

- ``common``   — vendored helpers: corpus file listing/loading, ``split_data``,
  ``standardize_role``. Bit-exactness constraints documented in the module.
- ``backends`` — inference backends (vllm | openai | dummy) behind one
  ``generate(message_lists) -> list[str]`` protocol.
- ``runner``   — drivers that execute per-trajectory method programs (lockstep
  batched for vLLM, streaming for APIs) and the atomic per-trajectory
  ``OutputWriter`` whose files double as the resume ledger.
"""
