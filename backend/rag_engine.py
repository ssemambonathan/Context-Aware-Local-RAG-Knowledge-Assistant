import hashlib
import httpx
import json
import chromadb
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any, AsyncGenerator

OLLAMA_BASE_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen2.5-coder"

class HybridRAGEngine:
    def __init__(self, persist_dir: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="knowledge_base",
            metadata={"hnsw:space": "cosine"}
        )
        self._bm25_index = None
        self._all_docs = []
        self._all_metas = []
        self._refresh_bm25_cache()

    def _refresh_bm25_cache(self):
        count = self.collection.count()
        if count == 0:
            self._all_docs = []
            self._all_metas = []
            self._bm25_index = None
            return

        res = self.collection.get(include=["documents", "metadatas"])
        self._all_docs = res["documents"]
        self._all_metas = res["metadatas"]
        tokenized_corpus = [doc.lower().split() for doc in self._all_docs]
        self._bm25_index = BM25Okapi(tokenized_corpus)

    async def get_embeddings(self, texts: List[str], model: str = EMBED_MODEL) -> List[List[float]]:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{OLLAMA_BASE_URL}/api/embed",
                json={"model": model, "input": texts}
            )
            resp.raise_for_status()
            return resp.json()["embeddings"]

    async def index_chunks(self, chunks: List[Dict[str, Any]], embed_model: str = EMBED_MODEL) -> int:
        if not chunks:
            return 0

        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        ids = [
            hashlib.sha256(f"{c['metadata']['source']}:{c['text']}".encode()).hexdigest()[:24]
            for c in chunks
        ]

        embeddings = await self.get_embeddings(texts, model=embed_model)
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
        self._refresh_bm25_cache()
        return len(chunks)

    def hybrid_search(self, question: str, q_embed: List[float], top_k: int = 5) -> Tuple[List[str], List[Dict], List[float]]:
        if not self._all_docs:
            return [], [], []

        # 1. Vector Search via Chroma
        chroma_res = self.collection.query(
            query_embeddings=[q_embed],
            n_results=min(top_k * 2, len(self._all_docs)),
            include=["documents", "metadatas", "distances"]
        )
        v_docs = chroma_res["documents"][0]
        v_metas = chroma_res["metadatas"][0]
        v_dists = chroma_res["distances"][0]

        # Convert cosine distance to similarity score [0, 1]
        v_scores = {doc: max(0.0, 1.0 - dist) for doc, dist in zip(v_docs, v_dists)}

        # 2. BM25 Keyword Search
        tokenized_q = question.lower().split()
        bm25_scores_raw = self._bm25_index.get_scores(tokenized_q)
        max_bm25 = max(bm25_scores_raw) if len(bm25_scores_raw) > 0 and max(bm25_scores_raw) > 0 else 1.0
        
        bm25_scores = {
            self._all_docs[i]: float(bm25_scores_raw[i] / max_bm25)
            for i in range(len(self._all_docs))
        }

        # 3. Reciprocal Rank Fusion / Combined Scoring (Alpha = 0.6 vector, 0.4 keyword)
        combined = {}
        meta_map = {self._all_docs[i]: self._all_metas[i] for i in range(len(self._all_docs))}

        all_candidate_docs = set(v_scores.keys()).union(set([self._all_docs[i] for i in range(len(self._all_docs)) if bm25_scores_raw[i] > 0]))

        for doc in all_candidate_docs:
            v_score = v_scores.get(doc, 0.0)
            b_score = bm25_scores.get(doc, 0.0)
            combined[doc] = (0.6 * v_score) + (0.4 * b_score)

        sorted_docs = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]

        final_docs = [d[0] for d in sorted_docs]
        final_metas = [meta_map[d] for d in final_docs]
        final_scores = [d[1] for d in sorted_docs]

        return final_docs, final_metas, final_scores

    async def query_stream(
        self,
        question: str,
        top_k: int = 5,
        chat_model: str = CHAT_MODEL,
        embed_model: str = EMBED_MODEL
    ) -> AsyncGenerator[str, None]:
        if self.collection.count() == 0:
            yield f"data: {json.dumps({'type': 'sources', 'sources': []})}\n\n"
            yield f"data: {json.dumps({'type': 'chunk', 'content': 'Your knowledge base is currently empty. Index a directory first.'})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

        q_embed = (await self.get_embeddings([question], model=embed_model))[0]
        docs, metas, scores = self.hybrid_search(question, q_embed, top_k=top_k)

        sources = []
        for doc, meta, score in zip(docs, metas, scores):
            sources.append({
                "source": meta.get("source"),
                "symbol": meta.get("symbol"),
                "lines": meta.get("lines"),
                "similarity": round(score, 3),
                "preview": doc[:200] + "..."
            })

        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

        if not docs or max(scores) < 0.15:
            yield f"data: {json.dumps({'type': 'chunk', 'content': 'I could not find sufficiently relevant code or documentation to answer this accurately.'})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

        context_block = "\n\n---\n\n".join(docs)
        system_prompt = (
            "You are a senior software architect answering questions strictly from the user's "
            "indexed codebase and documentation.\n"
            "Rules:\n"
            "1. Base your answer ONLY on the provided context.\n"
            "2. Cite exact file paths and function/class names when referencing code.\n"
            "3. If context is lacking, say so explicitly."
        )

        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": chat_model,
                    "stream": True,
                    "options": {"temperature": 0.1},
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Context:\n{context_block}\n\nQuestion: {question}"}
                    ]
                }
            ) as response:
                async for line in response.aiter_lines():
                    if line:
                        data = json.loads(line)
                        token = data.get("message", {}).get("content", "")
                        if token:
                            yield f"data: {json.dumps({'type': 'chunk', 'content': token})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    def get_stats(self) -> Dict[str, Any]:
        count = self.collection.count()
        unique_sources = sorted({m.get("source", "unknown") for m in self._all_metas})
        return {"total_chunks": count, "indexed_files": unique_sources}

    def clear(self):
        self.client.delete_collection("knowledge_base")
        self.collection = self.client.get_or_create_collection(
            name="knowledge_base",
            metadata={"hnsw:space": "cosine"}
        )
        self._refresh_bm25_cache()
