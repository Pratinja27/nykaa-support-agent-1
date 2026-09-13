import re

from config import MOCK_LLM


def classify_intent(query, last_record_id=None):
    text = query.lower()
    has_id = bool(re.search(r"NYK-\d{4}", query, re.I))
    order_words = ["order status", "track", "where is my order", "my order", "shipment status"]
    wants_status = any(w in text for w in order_words) or (
        "status" in text and ("order" in text or has_id or last_record_id)
    )
    if has_id or (wants_status and (has_id or last_record_id)):
        return "order_status"
    if wants_status and re.search(r"order|shipment|parcel|delivery", text):
        return "order_status"
    return "policy"


def extract_record_id(query, last_record_id=None):
    m = re.search(r"NYK-\d{4}", query, re.I)
    if m:
        return m.group(0).upper()
    return last_record_id


def _words(text):
    return set(w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2)


def generate_from_context(query, context, below_threshold=False):
    if not MOCK_LLM:
        raise RuntimeError("only MOCK_LLM is enabled for this project")
    if below_threshold or not context or not context.strip():
        return "I don't know. The knowledge base does not cover this question."
    q = _words(query)
    sentences = re.split(r"(?<=[.!?])\s+", context.strip())
    ranked = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        overlap = len(q & _words(s))
        ranked.append((overlap, s))
    ranked.sort(key=lambda x: x[0], reverse=True)
    picked = [s for ov, s in ranked if ov > 0][:3]
    if not picked:
        picked = [ranked[0][1]] if ranked else []
    if not picked:
        return "I don't know. The knowledge base does not cover this question."
    return " ".join(picked)
