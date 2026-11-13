import ast
import os
import fitz  # PyMuPDF
from typing import List, Dict, Any

def chunk_python_ast(source_code: str, file_path: str) -> List[Dict[str, Any]]:
    """Extracts top-level classes and functions using Python's AST."""
    chunks = []
    lines = source_code.splitlines()
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return chunk_text_sliding(source_code, file_path, "python")

    covered_lines = set()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = node.lineno - 1
            end = getattr(node, "end_lineno", start + 1)
            for i in range(start, end):
                covered_lines.add(i)
            snippet = "\n".join(lines[start:end])
            symbol_type = "class" if isinstance(node, ast.ClassDef) else "function"
            chunks.append({
                "text": f"File: {file_path} ({symbol_type}: {node.name}, lines {start+1}-{end})\n\n{snippet}",
                "metadata": {
                    "source": file_path,
                    "symbol": node.name,
                    "type": symbol_type,
                    "lines": f"{start+1}-{end}"
                }
            })

    # Capture module-level imports/constants not inside functions or classes
    leftover = [lines[i] for i in range(len(lines)) if i not in covered_lines and lines[i].strip()]
    if leftover:
        chunks.append({
            "text": f"File: {file_path} (module-level definitions)\n\n" + "\n".join(leftover),
            "metadata": {"source": file_path, "symbol": "module_scope", "type": "module", "lines": "various"}
        })

    return chunks or chunk_text_sliding(source_code, file_path, "python")


def chunk_text_sliding(text: str, source: str, doc_type: str, chunk_size: int = 1200, overlap: int = 200) -> List[Dict[str, Any]]:
    """Splits text/code into overlapping chunks while respecting line boundaries."""
    chunks = []
    start = 0
    text_len = len(text)
    chunk_idx = 0

    while start < text_len:
        end = min(start + chunk_size, text_len)
        if end < text_len:
            newline_pos = text.rfind("\n", start, end)
            if newline_pos > start + (chunk_size // 2):
                end = newline_pos + 1

        segment = text[start:end].strip()
        if segment:
            chunks.append({
                "text": f"Source: {source} (chunk {chunk_idx + 1})\n\n{segment}",
                "metadata": {
                    "source": source,
                    "symbol": f"chunk_{chunk_idx + 1}",
                    "type": doc_type,
                    "lines": f"char {start}-{end}"
                }
            })
            chunk_idx += 1
        start = end - overlap if end < text_len else text_len

    return chunks


def parse_and_chunk_file(file_path: str, content_bytes: bytes = None) -> List[Dict[str, Any]]:
    """Routes file ingestion based on extension."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        doc = fitz.open(stream=content_bytes, filetype="pdf") if content_bytes else fitz.open(file_path)
        all_chunks = []
        for page_num, page in enumerate(doc, start=1):
            page_text = page.get_text("text").strip()
            if page_text:
                page_chunks = chunk_text_sliding(page_text, f"{file_path} (p.{page_num})", "pdf")
                all_chunks.extend(page_chunks)
        return all_chunks

    raw_text = content_bytes.decode("utf-8", errors="ignore") if content_bytes is not None else open(file_path, "r", encoding="utf-8", errors="ignore").read()

    if ext == ".py":
        return chunk_python_ast(raw_text, file_path)

    return chunk_text_sliding(raw_text, file_path, ext.lstrip(".") or "text")
