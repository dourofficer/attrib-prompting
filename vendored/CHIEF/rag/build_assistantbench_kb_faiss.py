# filename: build_assistantbench_kb.py
import os
import json
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

DATA_FILE = "data/assistant_bench_v1.0_dev.jsonl"
KB_FILE = "kb/assistantbench_kb.json"
FAISS_INDEX_FILE = "index/assistantbench.index"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(EMBED_MODEL)
def build_knowledge_base():
    data = []
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading AssistantBench"):
            item = json.loads(line)
            task = item.get("task", "").strip()
            explanation = item.get("explanation", "").strip()
            if not task or not explanation:
                continue
            text = f"Task: {task}\nExplanation: {explanation}"
            data.append({
                "id": item.get("id", ""),
                "text": text
            })
    print(f"✅ Loaded {len(data)} valid samples.")
    return data

def embed_and_save(data):
    texts = [item["text"] for item in data]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

    kb = []
    for i, item in enumerate(data):
        kb.append({
            "id": item["id"],
            "text": item["text"],
            "embedding": embeddings[i].tolist()
        })

    with open(KB_FILE, "w", encoding="utf-8") as f:
        json.dump(kb, f, ensure_ascii=False, indent=2)
    print(f"💾 Saved knowledge base to {KB_FILE}")
    return kb

def build_faiss_index(kb):
    embeddings = np.array([np.array(x["embedding"], dtype="float32") for x in kb])
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, FAISS_INDEX_FILE)

    print(f"📂 Saved FAISS index → {FAISS_INDEX_FILE}")

def query_faiss(query, model, index_file, top_k=3):
    index = faiss.read_index(index_file)
    query_vec = model.encode([query], normalize_embeddings=True)
    D, I = index.search(query_vec, top_k)
    for idx, score in zip(I[0], D[0]):
        print(f"\n🔹 Score: {score:.3f}")

def main():
    print(f"🚀 Building AssistantBench knowledge base using local model: {EMBED_MODEL}")
    data = build_knowledge_base()
    kb = embed_and_save(data)
    build_faiss_index(kb)
    print("✅ All done!")
    """
    query_faiss("Find gyms near Tompkins Square Park with early classes", model, "assistantbench.index")
    """

if __name__ == "__main__":
    main()
