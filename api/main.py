import os
import time
import uuid

from fastapi import FastAPI
from pydantic import BaseModel, Field

from agent.guardrails import mask_pii
from api.logging_util import write_log
from config import KB_DIR
from mcp_svc.server import mcp_app

app = FastAPI(title="Nykaa Support Agent", lifespan=mcp_app.lifespan)
app.mount("/mcp", mcp_app)


@app.get("/")
def root():
    return {
        "name": "Nykaa Support Agent",
        "docs": "/docs",
        "ask": "POST /ask",
        "add_document": "POST /add-document",
        "mcp": "/mcp",
    }


class AskRequest(BaseModel):
    query: str
    session_id: str = "default"


class AskResponse(BaseModel):
    answer: str
    intent: str
    grounded: bool
    sources: list
    record_id: str | None = None
    escalation_score: float | None = None
    escalate: bool | None = None
    trace_id: str


class AddDocumentRequest(BaseModel):
    title: str = Field(..., min_length=1)
    text: str = Field(..., min_length=8)
    doc_id: str | None = None


class AddDocumentResponse(BaseModel):
    doc_id: str
    path: str
    chunks_added: int
    trace_id: str


def _safe_doc_id(title, given):
    if given:
        return given.replace(" ", "_").lower()
    keep = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in title.lower())
    return keep.strip("_") or ("doc_%s" % uuid.uuid4().hex[:8])


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    from agent.graph import run_agent

    trace_id = uuid.uuid4().hex
    t0 = time.time()
    masked = mask_pii(req.query)
    result = run_agent(masked, session_id=req.session_id)
    ms = int((time.time() - t0) * 1000)
    write_log(
        {
            "trace_id": trace_id,
            "path": "/ask",
            "query": masked,
            "intent": result.get("intent"),
            "latency_ms": ms,
        }
    )
    return AskResponse(
        answer=result["answer"],
        intent=result["intent"],
        grounded=result["grounded"],
        sources=result["sources"],
        record_id=result.get("record_id"),
        escalation_score=result.get("escalation_score"),
        escalate=result.get("escalate"),
        trace_id=trace_id,
    )


@app.post("/add-document", response_model=AddDocumentResponse)
def add_document(req: AddDocumentRequest):
    from rag.chunking import sentence_chunks
    from rag.embeddings import embed_texts
    from rag.retrieval import get_collection

    trace_id = uuid.uuid4().hex
    t0 = time.time()
    masked_title = mask_pii(req.title)
    masked_text = mask_pii(req.text)
    doc_id = _safe_doc_id(masked_title, req.doc_id)
    path = os.path.join(KB_DIR, doc_id + ".txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(masked_text)
    pieces = sentence_chunks(masked_text)
    col = get_collection("sentence")
    ids = ["added-%s-%d" % (doc_id, i) for i in range(len(pieces))]
    vectors = embed_texts(pieces)
    col.add(
        ids=ids,
        documents=pieces,
        embeddings=vectors,
        metadatas=[{"doc_id": doc_id, "topic": masked_title, "strategy": "sentence"}] * len(pieces),
    )
    ms = int((time.time() - t0) * 1000)
    write_log(
        {
            "trace_id": trace_id,
            "path": "/add-document",
            "query": masked_title,
            "chunks_added": len(pieces),
            "latency_ms": ms,
        }
    )
    return AddDocumentResponse(
        doc_id=doc_id,
        path=path,
        chunks_added=len(pieces),
        trace_id=trace_id,
    )
