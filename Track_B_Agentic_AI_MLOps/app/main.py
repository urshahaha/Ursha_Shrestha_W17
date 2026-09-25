import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from .agent import run_agent
from .config import settings
from .llm import answer_question
from .rag import RagStore
from .schemas import AgentAnswer, AskRequest, AssistantAnswer, IngestResponse

app = FastAPI(title="Week 16 Self-Checking RAG Assistant", version="2.0.0")
rag_store = RagStore()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "provider": settings.llm_provider,
        "indexed_chunks": rag_store.collection.count(),
        "agent_max_steps": settings.agent_max_steps,
    }


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".pdf", ".txt", ".md"}:
        raise HTTPException(400, "Upload PDF, TXT, or MD.")
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    target = upload_dir / Path(file.filename or "document.txt").name
    try:
        with target.open("wb") as out:
            shutil.copyfileobj(file.file, out)
        return IngestResponse(filename=target.name, chunks_indexed=rag_store.ingest(target))
    except Exception as exc:
        raise HTTPException(500, f"Ingestion failed: {exc}") from exc


@app.post("/ask", response_model=AssistantAnswer)
async def ask(request: AskRequest):
    """Original W15 single-pass endpoint kept as a baseline."""
    try:
        return answer_question(request.question, rag_store.search(request.question))
    except Exception as exc:
        raise HTTPException(503, f"Assistant unavailable: {exc}") from exc


@app.post("/agent/ask", response_model=AgentAnswer)
async def agent_ask(request: AskRequest):
    """W16 agentic endpoint. The model chooses each next action inside a bounded loop."""
    try:
        return run_agent(request.question, rag_store)
    except Exception as exc:
        raise HTTPException(503, f"Agent unavailable: {exc}") from exc
