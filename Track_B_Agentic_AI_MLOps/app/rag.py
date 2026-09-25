from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import chromadb
import requests
from pypdf import PdfReader

from .config import settings


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap
    return chunks


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    raise ValueError("Unsupported file type. Upload PDF, TXT, or MD.")


class RagStore:
    def __init__(self):
        self.embedding_model = "models/gemini-embedding-001"
        self.client = chromadb.PersistentClient(path=settings.chroma_dir)
        self.collection = self.client.get_or_create_collection(
            name="week15_documents",
            metadata={"hnsw:space": "cosine"},
        )

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is missing.")

        url = (
            "https://generativelanguage.googleapis.com/"
            "v1beta/models/gemini-embedding-001:batchEmbedContents"
        )
        payload = {
            "requests": [
                {
                    "model": self.embedding_model,
                    "content": {"parts": [{"text": text}]},
                    "outputDimensionality": 384,
                }
                for text in texts
            ]
        }
        response = requests.post(
            url,
            headers={
                "x-goog-api-key": settings.gemini_api_key,
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        return [item["values"] for item in data["embeddings"]]

    def ingest(self, path: Path) -> int:
        chunks = chunk_text(extract_text(path))
        if not chunks:
            return 0

        embeddings = self._embed(chunks)
        ids = [str(uuid4()) for _ in chunks]
        metadatas = [
            {"filename": path.name, "chunk_id": i}
            for i in range(len(chunks))
        ]
        self.collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        return len(chunks)

    def search(self, query: str, k: int | None = None) -> list[dict]:
        if self.collection.count() == 0:
            return []

        top_k = max(1, k or settings.rag_top_k)
        query_embedding = self._embed([query])[0]
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        rows: list[dict] = []
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        for document, metadata, distance in zip(documents, metadatas, distances):
            metadata = metadata or {}
            rows.append(
                {
                    "filename": str(metadata.get("filename", "unknown")),
                    "chunk_id": int(metadata.get("chunk_id", 0)),
                    "text": document or "",
                    "distance": float(distance),
                }
            )
        return rows
