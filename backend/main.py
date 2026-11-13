import os
import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional

from chunker import parse_and_chunk_file
from rag_engine import LocalRAGEngine, OLLAMA_BASE_URL, CHAT_MODEL, EMBED_MODEL

app = FastAPI(title="Context-Aware Local RAG Assistant")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = LocalRAGEngine()

ALLOWED_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".kt",
    ".cpp", ".c", ".h", ".sql", ".md", ".txt", ".json", ".yaml", ".yml", ".pdf"
}
IGNORE_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", ".next"}


class DirectoryIngestRequest(BaseModel):
    directory_path: str
    embed_model: str = EMBED_MODEL


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    chat_model: str = CHAT_MODEL
    embed_model: str = EMBED_MODEL


@app.get("/")
async def serve_ui():
    frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")
    return FileResponse(frontend_path)


@app.get("/api/health")
async def health_check():
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            models = [m["name"] for m in resp.json().get("models", [])]
        return {"ollama_connected": True, "models": models, "stats": rag.get_stats()}
    except Exception as e:
        return {"ollama_connected": False, "error": str(e), "models": [], "stats": rag.get_stats()}


@app.post("/api/ingest/directory")
async def ingest_directory(req: DirectoryIngestRequest):
    if not os.path.isdir(req.directory_path):
        raise HTTPException(status_code=400, detail="Directory does not exist on local machine.")

    all_chunks = []
    files_processed = 0

    for root, dirs, files in os.walk(req.directory_path):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext in ALLOWED_EXTENSIONS:
                full_path = os.path.join(root, fname)
                rel_path = os.path.relpath(full_path, req.directory_path)
                try:
                    chunks = parse_and_chunk_file(full_path)
                    for c in chunks:
                        c["metadata"]["source"] = rel_path
                    all_chunks.extend(chunks)
                    files_processed += 1
                except Exception:
                    continue

    indexed = await rag.index_chunks(all_chunks, embed_model=req.embed_model)
    return {"files_processed": files_processed, "chunks_indexed": indexed, "stats": rag.get_stats()}


@app.post("/api/ingest/files")
async def ingest_files(files: List[UploadFile] = File(...), embed_model: str = EMBED_MODEL):
    all_chunks = []
    for f in files:
        content = await f.read()
        chunks = parse_and_chunk_file(f.filename, content_bytes=content)
        all_chunks.extend(chunks)

    indexed = await rag.index_chunks(all_chunks, embed_model=embed_model)
    return {"files_processed": len(files), "chunks_indexed": indexed, "stats": rag.get_stats()}


@app.post("/api/query")
async def ask_question(req: QueryRequest):
    try:
        return await rag.query(
            question=req.question,
            top_k=req.top_k,
            chat_model=req.chat_model,
            embed_model=req.embed_model
        )
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Ollama request failed: {str(e)}")


@app.delete("/api/index")
async def reset_index():
    rag.clear()
    return {"status": "cleared", "stats": rag.get_stats()}
