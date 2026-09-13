import os

from config import (
    CHROMA_DIR,
    FIXED_COLLECTION,
    SENTENCE_COLLECTION,
    SIMILARITY_THRESHOLD,
    TOP_K,
)
from rag.chunking import build_chunk_records
from rag.embeddings import embed_texts


def get_client():
    import chromadb

    os.makedirs(CHROMA_DIR, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_DIR)


def collection_name(strategy):
    if strategy == "fixed":
        return FIXED_COLLECTION
    if strategy == "sentence":
        return SENTENCE_COLLECTION
    raise ValueError("unknown strategy")


def build_collections():
    client = get_client()
    built = {}
    for strategy in ("fixed", "sentence"):
        name = collection_name(strategy)
        try:
            client.delete_collection(name)
        except Exception:
            pass
        col = client.create_collection(name=name, metadata={"hnsw:space": "cosine"})
        records = build_chunk_records(strategy)
        texts = [r["text"] for r in records]
        vectors = embed_texts(texts)
        col.add(
            ids=[r["id"] for r in records],
            documents=texts,
            embeddings=vectors,
            metadatas=[
                {"doc_id": r["doc_id"], "topic": r["topic"], "strategy": r["strategy"]}
                for r in records
            ],
        )
        built[strategy] = {"name": name, "chunks": len(records)}
    return built


def get_collection(strategy):
    client = get_client()
    return client.get_collection(collection_name(strategy))


def retrieve(query, strategy="sentence", top_k=TOP_K):
    col = get_collection(strategy)
    qvec = embed_texts([query])[0]
    result = col.query(
        query_embeddings=[qvec],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    rows = []
    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    dists = result.get("distances", [[]])[0]
    ids = result.get("ids", [[]])[0]
    for i in range(len(docs)):
        dist = dists[i]
        sim = 1.0 - float(dist)
        rows.append(
            {
                "id": ids[i],
                "text": docs[i],
                "doc_id": metas[i].get("doc_id"),
                "topic": metas[i].get("topic"),
                "distance": float(dist),
                "similarity": sim,
            }
        )
    return rows


def top1_similarity(query, strategy="sentence"):
    rows = retrieve(query, strategy=strategy, top_k=1)
    if not rows:
        return 0.0
    return rows[0]["similarity"]


def context_from_hits(hits, threshold=SIMILARITY_THRESHOLD):
    kept = [h for h in hits if h["similarity"] >= threshold]
    text = " ".join(h["text"] for h in kept)
    sources = []
    for h in kept:
        if h["doc_id"] not in sources:
            sources.append(h["doc_id"])
    return text, sources, kept
