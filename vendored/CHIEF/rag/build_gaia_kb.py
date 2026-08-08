# filename: build_gaia_kb.py
import pandas as pd
import json
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ===== 配置 =====
GAIA_FILE = "data/GAIA_dataset.parquet"
OUTPUT_INDEX = "index/gaia.index"
OUTPUT_JSON = "kb/gaia_kb.json"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

print("📂 正在加载 GAIA parquet 文件...")
df = pd.read_parquet(GAIA_FILE)

records = []
for i, row in tqdm(df.iterrows(), total=len(df), desc="构建知识条目"):
    q = str(row.get("Question", "")).strip()
    meta = row.get("Annotator Metadata", "")

    steps = ""
    if isinstance(meta, dict):
        steps = meta.get("Steps", "")
    elif isinstance(meta, str):
        try:
            meta_dict = json.loads(meta)
            steps = meta_dict.get("Steps", "")
        except:
            steps = meta

    combined = f"Question: {q}\nSteps: {steps}".strip()
    records.append({
        "id": i,
        "question": q,
        "steps": steps,
        "combined_text": combined
    })

print("⚙️ 加载 embedding 模型:", EMBED_MODEL)
model = SentenceTransformer(EMBED_MODEL)

texts = [r["combined_text"] for r in records]
embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

dim = embeddings.shape[1]
faiss.normalize_L2(embeddings)
index = faiss.IndexFlatIP(dim)
index.add(embeddings)

faiss.write_index(index, OUTPUT_INDEX)
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print(f"\n✅ 知识库构建完成！")
print(f"📁 FAISS 索引: {OUTPUT_INDEX}")
print(f"📄 JSON 数据: {OUTPUT_JSON}")
print(f"📊 共收录 {len(records)} 条知识样本。")
