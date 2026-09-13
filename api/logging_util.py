import json
import os
import time
import uuid

from config import LOG_PATH


def new_trace_id():
    return uuid.uuid4().hex


def write_log(entry):
    folder = os.path.dirname(LOG_PATH)
    os.makedirs(folder, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def log_request(trace_id, path, masked_query, extra=None):
    row = {
        "trace_id": trace_id,
        "path": path,
        "query": masked_query,
        "ts": time.time(),
    }
    if extra:
        row.update(extra)
    write_log(row)
    return row
