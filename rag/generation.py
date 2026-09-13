from config import SIMILARITY_THRESHOLD, TOP_K
from mock_llm import generate_from_context
from rag.retrieval import context_from_hits, retrieve


def grounded_answer(query, strategy="sentence", top_k=TOP_K, threshold=SIMILARITY_THRESHOLD):
    hits = retrieve(query, strategy=strategy, top_k=top_k)
    top1 = hits[0]["similarity"] if hits else 0.0
    below = top1 < threshold
    context, sources, kept = context_from_hits(hits, threshold=threshold)
    answer = generate_from_context(query, context, below_threshold=below)
    return {
        "query": query,
        "answer": answer,
        "top1_similarity": top1,
        "below_threshold": below,
        "sources": sources,
        "hits": hits,
        "kept": kept,
        "context": context,
        "strategy": strategy,
    }
