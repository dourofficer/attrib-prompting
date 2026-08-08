"""Trajectory-similarity precomputation — one subset per invocation.

Port of ``baselines/CORRECT/src/generate_trajectory_similarities.py``, verbatim
in behaviour: each trajectory is rendered as ``"Question: {q}"`` plus one
``"{agent}: {content}"`` line per turn, embedded with BAAI/bge-m3 (HF
``AutoModel``, batches of 8, tokenizer truncation at 8192), mean-pooled over the
attention mask, L2-normalized; pairwise cosine similarities give, per
trajectory, the ranked list of all *other* trajectory file-numbers by descending
similarity (self excluded — this is the leave-one-out mask retrieval relies on).

The embedder is a similarity encoder, not a generator, so it stays on HF
transformers exactly as vendored (vLLM is for generation).

Usage
-----
python -m baselines.correct.similarity \
    --input  data/ww/hand-crafted \
    --output outputs-ww/correct/similarities/hand-crafted_trajectory_similarities.json \
    --model  ../hub/BAAI/bge-m3
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def read_trajectory_json(file_path) -> str:
    """Read a trajectory JSON file and concatenate all conversation content."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        history = data.get("history", [])
        question = data.get("question", "")

        text_parts = [f"Question: {question}"] if question else []
        for entry in history:
            # Handle both handcrafted and non-handcrafted formats.
            agent_name = entry.get("role", entry.get("name", "Unknown"))
            content = entry.get("content", "")
            text_parts.append(f"{agent_name}: {content}")

        return "\n".join(text_parts)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""


def mean_pooling(model_output, attention_mask):
    """Mean pooling - take attention mask into account for correct averaging."""
    import torch

    token_embeddings = model_output[0]  # First element contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9)


def embed_texts(texts: list[str], model_name: str, batch_size: int = 8,
                max_length: int = 8192) -> np.ndarray:
    """Embed texts with the vendored encode → mean-pool → L2-normalize recipe."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")

    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        encoded_input = tokenizer(batch_texts, padding=True, truncation=True,
                                  max_length=max_length, return_tensors="pt")
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

        with torch.no_grad():
            model_output = model(**encoded_input)

        batch_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
        batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
        embeddings.extend(batch_embeddings.cpu().numpy())

    return np.array(embeddings)


def rank_neighbours(embeddings: np.ndarray, file_indices: list[int]) -> dict[int, list[int]]:
    """Vendored ranking: cosine similarity, self excluded, descending order."""
    from sklearn.metrics.pairwise import cosine_similarity

    similarity_matrix = cosine_similarity(embeddings)

    similarity_mappings: dict[int, list[int]] = {}
    for i, file_idx in enumerate(file_indices):
        similarities = similarity_matrix[i]
        similarity_pairs = []
        for j, other_idx in enumerate(file_indices):
            if i != j:  # Exclude self-similarity
                similarity_pairs.append((other_idx, similarities[j]))
        similarity_pairs.sort(key=lambda x: x[1], reverse=True)
        similarity_mappings[file_idx] = [idx for idx, _ in similarity_pairs]

    return similarity_mappings


def compute_trajectory_similarities(dataset_path: str, model_name: str,
                                    batch_size: int = 8,
                                    max_length: int = 8192) -> dict[int, list[int]]:
    """Compute the ranked-neighbour map for every trajectory in a subset dir."""
    trajectory_texts: list[str] = []
    file_indices: list[int] = []

    for f in sorted(os.listdir(dataset_path)):
        if f.endswith(".json") and f != "file_mapping.json":
            try:
                file_num = int(f.replace(".json", ""))
            except ValueError:
                print(f"Skipping file with non-numeric name: {f}")
                continue
            text = read_trajectory_json(os.path.join(dataset_path, f))
            if text:
                trajectory_texts.append(text)
                file_indices.append(file_num)

    print(f"Found {len(trajectory_texts)} trajectory files")
    if not trajectory_texts:
        return {}

    print("Computing embeddings...")
    embeddings = embed_texts(trajectory_texts, model_name, batch_size, max_length)

    print("Computing pairwise similarities...")
    return rank_neighbours(embeddings, file_indices)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate CORRECT trajectory-similarity mappings.")
    p.add_argument("--input", required=True, help="Subset directory of trajectory JSONs.")
    p.add_argument("--output", required=True, help="Output JSON file path.")
    p.add_argument("--model", default="BAAI/bge-m3",
                   help="Embedding model (HF name or local path).")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_length", type=int, default=8192)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    out_path = Path(args.output)
    if out_path.exists() and not args.overwrite:
        print(f"skip (exists): {out_path}")
        return

    similarities = compute_trajectory_similarities(
        args.input, args.model, args.batch_size, args.max_length)
    if not similarities:
        print("No similarities computed")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(similarities, f, indent=2)
    print(f"  wrote {out_path}  ({len(similarities)} trajectories)")


if __name__ == "__main__":
    main()
