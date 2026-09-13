import jsonschema

RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["answer", "intent", "grounded", "sources"],
    "properties": {
        "answer": {"type": "string"},
        "intent": {"type": "string", "enum": ["policy", "order_status", "blocked"]},
        "grounded": {"type": "boolean"},
        "sources": {"type": "array", "items": {"type": "string"}},
        "record_id": {"type": ["string", "null"]},
        "escalation_score": {"type": ["number", "null"]},
        "escalate": {"type": ["boolean", "null"]},
    },
    "additionalProperties": True,
}


def validate_response(payload):
    jsonschema.validate(payload, RESPONSE_SCHEMA)
    return payload
