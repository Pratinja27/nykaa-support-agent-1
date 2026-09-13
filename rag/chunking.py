import os
import re

from config import FIXED_CHUNK_OVERLAP, FIXED_CHUNK_SIZE, KB_DIR

DOC_TOPICS = {
    "return_window": "Return window by product category",
    "cod_refund": "COD refund timelines",
    "delivery_slas": "Delivery SLAs",
    "reverse_pickup": "Reverse-pickup eligibility",
    "warranty": "Warranty terms by category",
    "cancellation": "Order-cancellation policy",
    "loyalty_points": "Loyalty-points redemption policy",
    "payment_failure": "Payment-failure/retry policy",
    "size_exchange": "Size-exchange policy",
    "damaged_item": "Damaged-item claim process",
    "international_shipping": "International shipping restrictions",
    "escalation_matrix": "Customer-support escalation matrix",
}


def load_documents():
    docs = []
    for name in sorted(os.listdir(KB_DIR)):
        if not name.endswith(".txt"):
            continue
        doc_id = name[:-4]
        path = os.path.join(KB_DIR, name)
        with open(path, encoding="utf-8") as f:
            text = f.read().strip()
        docs.append(
            {
                "doc_id": doc_id,
                "topic": DOC_TOPICS.get(doc_id, doc_id),
                "text": text,
                "path": path,
            }
        )
    return docs


def fixed_size_chunks(text, size=FIXED_CHUNK_SIZE, overlap=FIXED_CHUNK_OVERLAP):
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


def sentence_chunks(text):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def build_chunk_records(strategy):
    if strategy not in ("fixed", "sentence"):
        raise ValueError("strategy must be fixed or sentence")
    records = []
    for doc in load_documents():
        pieces = (
            fixed_size_chunks(doc["text"])
            if strategy == "fixed"
            else sentence_chunks(doc["text"])
        )
        for i, piece in enumerate(pieces):
            records.append(
                {
                    "id": "%s-%s-%d" % (strategy, doc["doc_id"], i),
                    "text": piece,
                    "doc_id": doc["doc_id"],
                    "topic": doc["topic"],
                    "strategy": strategy,
                }
            )
    return records
