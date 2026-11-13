# 🧠 Context-Aware Local RAG Knowledge Assistant

A 100% private, offline-first Retrieval-Augmented Generation (RAG) system designed for software engineers and architects. Powered by **FastAPI**, **ChromaDB**, **AST-aware code parsing**, and **Ollama**, this application indexes your local codebases, PDF notes, and technical documentation so you can ask hyper-specific architectural questions without sending a single line of proprietary code to the cloud.

---

## ✨ Key Features

* **Zero-Leakage Local Execution:** Runs entirely on your machine using local LLMs (`qwen2.5-coder`) and local embedding models (`nomic-embed-text`) via Ollama.
* **AST-Aware Code Chunking:** Uses Python's Abstract Syntax Tree (`ast`) to extract complete functions, classes, and module scopes with exact line numbers—preventing broken syntax and preserving architectural context.
* **Multi-Format Document Ingestion:** Supports recursive directory scanning for source code (`.py`, `.js`, `.ts`, `.tsx`, `.go`, `.rs`, `.java`, `.kt`, `.cpp`, `.sql`, etc.) as well as direct file uploads for **PDFs** (via `PyMuPDF`) and **Markdown/JSON/YAML** docs.
* **Hallucination Guardrails & Citations:** Enforces cosine-similarity thresholds and strict system prompts so the model only answers from retrieved context, displaying clickable source cards with file paths, symbol names, line numbers, and similarity percentages.
* **Zero-Build Developer UI:** Comes with a responsive, dark-mode React + Tailwind web interface served directly by FastAPI.

---

## 🏗️ System Architecture

```text
┌──────────────────────┐      ┌─────────────────────────┐      ┌─────────────────────┐
│   React Web UI       │─────▶│    FastAPI Backend      │─────▶│   ChromaDB (Local)  │
│  (frontend/index.html)│◀─────│  (main.py / rag_engine) │◀─────│  Persistent Vectors │
└──────────────────────┘      └────────────┬────────────┘      └─────────────────────┘
                                           │
                          ┌────────────────┴────────────────┐
                          ▼                                 ▼
              ┌───────────────────────┐         ┌───────────────────────┐
              │  AST & PDF Chunker    │         │    Ollama Runtime     │
              │     (chunker.py)      │         │  /api/embed & /chat   │
              └───────────────────────┘         └───────────────────────┘
```

---

## 📂 Project Structure

```text
local-rag-assistant/
├── backend/
│   ├── main.py             # FastAPI server, directory walker & API routes
│   ├── rag_engine.py       # ChromaDB vector storage & Ollama client
│   ├── chunker.py          # AST-aware Python parser & sliding-window text/PDF chunker
│   └── requirements.txt    # Python dependencies
├── frontend/
│   └── index.html          # Single-file React + Tailwind CSS developer workspace
└── README.md               # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
* **Python 3.10+**
* **[Ollama](https://ollama.com/)** installed and running locally on port `11434`.

### 2. Pull Required Local Models
Download the recommended embedding and code-reasoning models via Ollama:
```bash
ollama pull nomic-embed-text
ollama pull qwen2.5-coder
```

### 3. Install Backend Dependencies
Clone the repository, set up a virtual environment, and install the required Python packages:
```bash
cd backend
python -m venv .venv

# Activate on macOS/Linux:
source .venv/bin/activate

# Activate on Windows:
# .venv\Scripts\activate

pip install -r requirements.txt
```

### 4. Launch the Application
Start the FastAPI server using Uvicorn:
```bash
uvicorn main:app --reload --port 8000
```
Open your browser and navigate to **`http://localhost:8000`**.

---

## 💡 Usage Guide

1. **Check Ollama Status:** Verify the green **"Ollama Online"** indicator in the top-left sidebar. You can customize the chat or embedding model names directly in the sidebar inputs.
2. **Index a Codebase:** Paste the absolute path to any local repository (e.g., `/Users/you/projects/my-backend`) into the **Index Local Codebase** field and click **Scan & Index Folder**. Common build/dependency folders (`node_modules`, `.git`, `.venv`, `dist`) are automatically ignored.
3. **Upload Reference Docs:** Use the **Upload PDFs / Docs / Code** picker to ingest architecture RFCs, API specs, or database schemas.
4. **Query Your Architecture:** Ask questions in the chat panel such as:
   * *"How does user authentication validate JWT tokens across middleware?"*
   * *"Which functions interact with the orders table, and where are transactions handled?"*
   * *"Summarize the retry logic described in the uploaded system design PDF."*

---

## 🛠️ API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the frontend React web application. |
| `GET` | `/api/health` | Checks Ollama connectivity, available models, and ChromaDB index stats. |
| `POST` | `/api/ingest/directory` | Recursively parses and indexes a local directory path. |
| `POST` | `/api/ingest/files` | Uploads and indexes one or more files via `multipart/form-data`. |
| `POST` | `/api/query` | Embeds a user query, retrieves top-k matching chunks, and generates a cited answer. |
| `DELETE` | `/api/index` | Wipes the local ChromaDB vector collection to start fresh. |

---

## 📄 License

MIT License — free to use, modify, and extend for personal or enterprise workflows.