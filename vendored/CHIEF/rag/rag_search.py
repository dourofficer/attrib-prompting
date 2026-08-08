# filename: rag_search.py
import faiss
import json
from sentence_transformers import SentenceTransformer


class RAGRetriever:
    def __init__(
        self,
        gaia_index_path="index\\gaia.index",
        gaia_json_path="kb\\gaia_kb.json",
        assist_index_path="index\\assistantbench.index",
        assist_json_path="kb\\assistantbench_kb.json",
        embed_model="sentence-transformers/all-MiniLM-L6-v2"
    ):

        print("⚙️ 正在加载索引与嵌入模型...")
        self.model = SentenceTransformer(embed_model)


        self.index_gaia = faiss.read_index(gaia_index_path)
        self.index_assist = faiss.read_index(assist_index_path)


        with open(gaia_json_path, "r", encoding="utf-8") as f:
            self.records_gaia = json.load(f)
        with open(assist_json_path, "r", encoding="utf-8") as f:
            self.records_assist = json.load(f)

        print("✅ RAGRetriever 初始化完成。")

    def _encode(self, text: str):
        vec = self.model.encode([text], convert_to_numpy=True)
        faiss.normalize_L2(vec)
        return vec

    def search(self, query: str, top_k: int = 2):
        query_vec = self._encode(query)

        D_g, I_g = self.index_gaia.search(query_vec, top_k)
        gaia_results = [
            {
                "source": "GAIA",
                "score": float(D_g[0][i]),
                "question": self.records_gaia[I_g[0][i]]["question"],
                "steps": self.records_gaia[I_g[0][i]]["steps"]
            }
            for i in range(len(I_g[0]))
        ]

        D_a, I_a = self.index_assist.search(query_vec, top_k)
        assist_results = [
            {
                "source": "AssistantBench",
                "score": float(D_a[0][i]),
                "text": self.records_assist[I_a[0][i]]["text"],

            }
            for i in range(len(I_a[0]))
        ]

        combined = gaia_results + assist_results
        combined_sorted = sorted(combined, key=lambda x: x["score"], reverse=True)

        return combined_sorted[1:top_k]
