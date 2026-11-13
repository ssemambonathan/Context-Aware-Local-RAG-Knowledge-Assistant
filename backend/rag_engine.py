import hashlib
import httpx
import chromadb
from typing import List, Dict, Any

OLLAMA_BASE_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen2.5-coder"

class LocalRAGEngine:
    def __init__(self, persist_dir: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="knowledge_base",
            metadata={"hnsw:space": "cosine"}
        )

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
        return len(chunks)

    async def query(
        self,
        question: str,
        top_k: int = 5,
        max_distance: float = 0.65,
        chat_model: str = CHAT_MODEL,
        embed_model: str = EMBED_MODEL
    ) -> Dict[str, Any]:
        if self.collection.count() == 0:
            return {
                "answer": "Your knowledge base is currently empty. Index a local directory or upload files first.",
                "sources": []
            }

        q_embed = (await self.get_embeddings([question], model=embed_model))[0]
        results = self.collection.query(
            query_embeddings=[q_embed],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"]
        )

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]

        relevant_contexts = []
        sources = []
        for doc, meta, dist in zip(docs, metas, dists):
            if dist <= max_distance:
                relevant_contexts.append(doc)
                sources.append({
                    "source": meta.get("source"),
                    "symbol": meta.get("symbol"),
                    "lines": meta.get("lines"),
                    "similarity": round(1 - dist, 3),
                    "preview": doc[:220] + "..."
                })

        if not relevant_contexts:
            return {
                "answer": "I could not find sufficiently relevant code or documentation in your indexed knowledge base to answer this accurately.",
                "sources": []
            }

        context_block = "\n\n---\n\n".join(relevant_contexts)
        system_prompt = (
            "You are a senior principal software architect answering questions strictly from the user's "
            "indexed codebase and technical documentation.\n"
            "Rules:\n"
            "1. Base your answer ONLY on the provided context.\n"
            "2. Cite exact file paths, function/class names, and line numbers when referencing code.\n"
            "3. If the context does not contain the answer, say so explicitly—never guess or invent APIs."
        )

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": chat_model,
                    "stream": False,
                    "options": {"temperature": 0.1},
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Context:\n{context_block}\n\nQuestion: {question}"}
                    ]
                }
            )
            resp.raise_for_status()
            answer = resp.json()["message"]["content"]

        return {"answer": answer, "sources": sources}

    def get_stats(self) -> Dict[str, Any]:
        count = self.collection.count()
        items = self.collection.get(include=["metadatas"]) if count > 0 else {"metadatas": []}
        unique_sources = sorted({m.get("source", "unknown") for m in items["metadatas"]})
        return {"total_chunks": count, "indexed_files": unique_sources}

    def clear(self):
        self.client.delete_collection("knowledge_base")
        self.collection = self.client.get_or_create_collection(
            name="knowledge_base",
            metadata={"hnsw:space": "cosine"}
        )
