import re

PHONE_RE = re.compile(r"\b[6-9]\d{9}\b")
CARD_LAST4_RE = re.compile(
    r"(?i)(\b(?:card(?:\s+(?:ending|no\.?|number))?|last\s*4|ending)\s*[:\-]?\s*)(\d{4})\b"
)
CARD_X_RE = re.compile(r"(?i)\bxxxx(\d{4})\b")

INJECTION_PATTERNS = [
    re.compile(r"ignore (all )?(previous|above|prior) (instructions|rules|prompts)", re.I),
    re.compile(r"forget (your|all) (instructions|rules)", re.I),
    re.compile(r"you are now", re.I),
    re.compile(r"system prompt", re.I),
    re.compile(r"reveal (your )?(hidden )?prompt", re.I),
    re.compile(r"override (the )?(guard|policy|rules)", re.I),
    re.compile(r"jailbreak", re.I),
]


def mask_pii(text):
    out = PHONE_RE.sub("**********", text)
    out = CARD_LAST4_RE.sub(lambda m: m.group(1) + "****", out)
    out = CARD_X_RE.sub("xxxx****", out)
    return out


def detect_injection(text):
    hits = []
    for pat in INJECTION_PATTERNS:
        if pat.search(text):
            hits.append(pat.pattern)
    return {"blocked": bool(hits), "patterns": hits}


def groundedness_ok(answer, context):
    if not answer:
        return False
    low = answer.lower()
    if "i don't know" in low:
        return True
    if not context or not context.strip():
        return False
    aw = set(w for w in re.findall(r"[a-z0-9]+", low) if len(w) > 3)
    cw = set(w for w in re.findall(r"[a-z0-9]+", context.lower()) if len(w) > 3)
    if not aw:
        return False
    return (len(aw & cw) / float(len(aw))) >= 0.35


def apply_input_guards(raw_query):
    masked = mask_pii(raw_query)
    inj = detect_injection(masked)
    return {
        "raw_query": raw_query,
        "masked_query": masked,
        "injection": inj,
    }
