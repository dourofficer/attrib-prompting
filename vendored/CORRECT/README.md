# CORRECT

**CORRECT: Condensed Error Recognition via Knowledge Transfer in Multi-agent Systems**

[**Paper**](https://arxiv.org/pdf/2509.24088) · [**Dataset (Hugging Face)**](https://huggingface.co/datasets/yifanyu/CORRECT-Error)

CORRECT is a training-free framework that distills past multi-agent-system
(MAS) failures into compact, reusable error schemata and reuses them at
inference time to localize decisive errors in new failure trajectories. This
repository also releases CORRECT-Error, a 2,226-trajectory benchmark for
step-level error attribution across seven MAS tasks.

## The CORRECT-Error benchmark

CORRECT-Error is a benchmark for **step-level error attribution** in multi-agent
systems. Each example is a complete multi-agent *failure* trajectory — the full
agent-by-agent interaction produced by
[Magentic-One](https://github.com/microsoft/autogen) on a standard task —
annotated with the single **decisive error** that caused the failure: which
agent made it (`mistake_agent`), at which step (`mistake_step`), and why
(`mistake_reason`), together with the task `question` and `groundtruth`.

It spans seven tasks (GAIA, HotpotQA, Musique, WikiMQA, ARC, Math500, MMLU-Pro)
and two generator models, for 2,226 labeled trajectories in total:

| Generator model | Trajectories |
|-----------------|-------------:|
| GPT-4o-mini     |          318 |
| GPT-5-Nano      |        1,908 |
| **Total**       |    **2,226** |

The task is to read a trajectory and localize its decisive error (the agent and
step). See [`data/README.md`](data/README.md) for the per-record schema; the
dataset is on Hugging Face at
https://huggingface.co/datasets/yifanyu/CORRECT-Error.

## Repository structure

```
src/        inference and schema/similarity generation code
data/       error schemata and trajectory-similarity mappings
scripts/    shell entry point for each step
```

The error schemata (`data/schemata_*`) and similarity mappings
(`data/similarities_*`) used in the paper are included in the repository.
Only the trajectory data is downloaded separately, as shown below.

## Installation

```bash
pip install -r requirements.txt
```



### CORRECT-Error (Table 2)

```bash
bash scripts/download_data.sh                # download trajectories from Hugging Face
bash scripts/run_inference_correct_error.sh  # GPT-5-Nano split, k=5 (paper §A.3)
```

`SPLIT=gpt4omini` runs the GPT-4o-mini split.

### Who&When (Table 1)

```bash
bash scripts/download_whoandwhen.sh          # clone the upstream Who&When data
bash scripts/run_inference_whoandwhen.sh     # Qwen-2.5-7B detector
```

Select a cloud detector with the `MODEL` variable:

```bash
MODEL=gpt-5 OPENAI_API_KEY=sk-... bash scripts/run_inference_whoandwhen.sh
MODEL=gemini-2.5-flash GOOGLE_APPLICATION_CREDENTIALS=key.json \
  GOOGLE_CLOUD_PROJECT=my-project bash scripts/run_inference_whoandwhen.sh
```

Each script documents its options in a header comment.

## Scoring

Both runners write per-prediction text files (`Prediction for <id>.json: ...`).
Score them with `src/evaluate.py`, the strict exact-match evaluator used in the
paper. The reported metric is step-level accuracy; `Step Accuracy (tolerance 0)`
is the exact-match number the paper reports, so pass `--tolerance 0`.

```bash
# CORRECT-Error (Table 2): ground truth = the downloaded trajectories
python src/evaluate.py --tolerance 0 \
  --eval_file outputs/all_at_once_similarity_5schemata_qwen-7b_gpt5nano_musique.txt \
  --data_path data/correct_error_gpt5nano/musique/individual_trajectories

# Who&When (Table 1): ground truth = the downloaded subset
python src/evaluate.py --tolerance 0 \
  --eval_file outputs_whoandwhen/all_at_once_10schemata_qwen-7b_hand_crafted.txt \
  --data_path data/whoandwhen/Hand-Crafted
```

## Regenerating schemata and similarities

The included artifacts cover the paper setup. Regenerate them only to use a
different model or a modified trajectory set.

| Task | Script | Module |
|------|--------|--------|
| Error schemata (local vLLM) | `scripts/generate_schemata.sh` | `src/error_schema_generator.py` |
| Error schemata (OpenAI / Gemini) | `scripts/generate_schemata_cloud.sh` | `src/error_schema_generator_cloud.py` |
| Similarities — CORRECT-Error gpt-4o-mini | `scripts/generate_similarities.sh` | `src/generate_trajectory_similarities.py` |
| Similarities — CORRECT-Error gpt-5-nano | `scripts/generate_similarities_gpt5nano.sh` | `src/generate_trajectory_similarities.py` |
| Similarities — Who&When | _(run module directly)_ | `src/generate_trajectory_similarities.py` |

## Loading the dataset

```python
from datasets import load_dataset

ds = load_dataset("yifanyu/CORRECT-Error", split="test")
row = ds[0]
print(row["dataset"], row["generator_model"], row["question_id"])
print("Decisive error at step", row["mistake_step"], "by", row["mistake_agent"])
```

The benchmark is hosted at https://huggingface.co/datasets/yifanyu/CORRECT-Error.
See [`data/README.md`](data/README.md) for the record schema and the layout of
the schemata and similarity files.

## Citation

```bibtex
@inproceedings{yu2026correct,
  title     = {{CORRECT}: Condensed Error Recognition via Knowledge Transfer in Multi-agent Systems},
  author    = {Yu, Yifan and Li, Moyan and Xu, Shaoyuan and Fu, Jinmiao and
               Hou, Xinhai and Lai, Fan and Wang, Bryan},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year      = {2026}
}
```

## Acknowledgements

Inference and schema-generation code is adapted from
[Agents_Failure_Attribution](https://github.com/mingyin1/Agents_Failure_Attribution)
(Zhang et al., ICML 2025). The CORRECT-Error trajectories were generated with
[Magentic-One](https://github.com/microsoft/autogen) (Fourney et al., 2024).
See [`NOTICE`](NOTICE) for full attribution.

## License

Apache 2.0. See [`LICENSE`](LICENSE).
