from rag.retrieval import retrieve

CALIBRATION_IN_SCOPE = [
    "What is the return window for apparel and beauty products?",
    "How long does a COD refund take after the warehouse receives the return?",
    "What is the delivery SLA for metro pin codes?",
]

CALIBRATION_OUT_SCOPE = [
    "Who won the cricket world cup last year?",
    "How do I file income tax returns in India?",
]

DEMO_IN_SCOPE = [
    "What is the return window for electronics?",
    "How many days does a COD refund take?",
    "When is reverse pickup cancelled?",
    "Can I redeem loyalty points on a COD order?",
    "Do you ship orders outside India?",
]

DEMO_OUT_SCOPE = [
    "Who won the cricket world cup last year?",
]

COMPARE_QUERIES = [
    {
        "query": "What is the return window for electronics?",
        "relevant": ["return_window"],
    },
    {
        "query": "How many days does a COD refund take?",
        "relevant": ["cod_refund"],
    },
    {
        "query": "When is reverse pickup cancelled?",
        "relevant": ["reverse_pickup"],
    },
    {
        "query": "Can I redeem loyalty points on a COD order?",
        "relevant": ["loyalty_points"],
    },
    {
        "query": "Do you ship orders outside India?",
        "relevant": ["international_shipping"],
    },
]

TRIAD_QUERIES = [
    {"id": 1, "topic": "return_window", "query": "What is the return window for beauty and electronics?", "kind": "kb"},
    {"id": 2, "topic": "cod_refund", "query": "How long do COD refunds take after warehouse receipt?", "kind": "kb"},
    {"id": 3, "topic": "delivery_slas", "query": "What is the delivery SLA for metro pin codes?", "kind": "kb"},
    {"id": 4, "topic": "reverse_pickup", "query": "When is reverse pickup offered and when is it cancelled?", "kind": "kb"},
    {"id": 5, "topic": "warranty", "query": "What warranty applies to electronics and beauty products?", "kind": "kb"},
    {"id": 6, "topic": "cancellation", "query": "Can I cancel an order after it is packed or shipped?", "kind": "kb"},
    {"id": 7, "topic": "loyalty_points", "query": "How do I redeem Nykaa loyalty points and when do they expire?", "kind": "kb"},
    {"id": 8, "topic": "payment_failure", "query": "What happens if my card or UPI payment fails?", "kind": "kb"},
    {"id": 9, "topic": "size_exchange", "query": "How does size exchange work for apparel and footwear?", "kind": "kb"},
    {"id": 10, "topic": "damaged_item", "query": "How do I claim a damaged item after delivery?", "kind": "kb"},
    {"id": 11, "topic": "international_shipping", "query": "Does Nykaa accept international shipping or overseas addresses?", "kind": "kb"},
    {"id": 12, "topic": "escalation_matrix", "query": "When does a support ticket move from L1 to L2 or L3?", "kind": "kb"},
    {"id": 13, "topic": "out_of_scope", "query": "Who won the cricket world cup last year?", "kind": "out_of_scope"},
    {"id": 14, "topic": "edge", "query": "Can I return a used lipstick after 45 days?", "kind": "edge"},
    {"id": 15, "topic": "cod_refund", "query": "What is the COD refund timeline for a returned footwear order?", "kind": "kb"},
]


def unique_docs(hits):
    seen = []
    for h in hits:
        if h["doc_id"] not in seen:
            seen.append(h["doc_id"])
    return seen


def precision_at_k(retrieved_docs, relevant_docs, k=3):
    top = retrieved_docs[:k]
    if k == 0:
        return 0.0
    hit = len(set(top) & set(relevant_docs))
    return hit / float(k)


def recall_at_k(retrieved_docs, relevant_docs, k=3):
    top = retrieved_docs[:k]
    if not relevant_docs:
        return 0.0
    hit = len(set(top) & set(relevant_docs))
    return hit / float(len(relevant_docs))


def score_strategy(strategy, queries=None, k=3):
    rows = queries if queries is not None else COMPARE_QUERIES
    results = []
    for item in rows:
        hits = retrieve(item["query"], strategy=strategy, top_k=k)
        docs = unique_docs(hits)
        p = precision_at_k(docs, item["relevant"], k)
        r = recall_at_k(docs, item["relevant"], k)
        results.append(
            {
                "query": item["query"],
                "relevant": item["relevant"],
                "retrieved_docs": docs,
                "hits": hits,
                "precision_at_3": p,
                "recall_at_3": r,
                "p_arith": "%d/%d" % (len(set(docs) & set(item["relevant"])), k),
                "r_arith": "%d/%d" % (len(set(docs) & set(item["relevant"])), len(item["relevant"])),
            }
        )
    avg_p = sum(x["precision_at_3"] for x in results) / len(results)
    avg_r = sum(x["recall_at_3"] for x in results) / len(results)
    return {"strategy": strategy, "rows": results, "avg_precision": avg_p, "avg_recall": avg_r}


def overlap_score(a, b):
    import re

    wa = set(w for w in re.findall(r"[a-z0-9]+", a.lower()) if len(w) > 2)
    wb = set(w for w in re.findall(r"[a-z0-9]+", b.lower()) if len(w) > 2)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / float(len(wa | wb))


def triad_row(item, strategy="sentence"):
    from rag.generation import grounded_answer

    out = grounded_answer(item["query"], strategy=strategy)
    context = out["context"]
    answer = out["answer"]
    query = item["query"]
    ctx_rel = overlap_score(query, context)
    grounded = overlap_score(answer, context) if context else 0.0
    ans_rel = overlap_score(answer, query)
    if out["below_threshold"]:
        ctx_rel = min(ctx_rel, 0.15)
        grounded = 1.0
        ans_rel = 0.2
    return {
        "id": item["id"],
        "topic": item["topic"],
        "kind": item["kind"],
        "query": query,
        "answer": answer,
        "context_relevance": round(ctx_rel, 3),
        "groundedness": round(grounded, 3),
        "answer_relevance": round(ans_rel, 3),
        "top1_similarity": round(out["top1_similarity"], 3),
        "below_threshold": out["below_threshold"],
        "sources": out["sources"],
    }
