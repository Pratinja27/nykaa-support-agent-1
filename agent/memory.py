import json
import os

from config import MEMORY_PATH


def _empty():
    return {"sessions": {}}


def load_store():
    if not os.path.exists(MEMORY_PATH):
        return _empty()
    with open(MEMORY_PATH, encoding="utf-8") as f:
        return json.load(f)


def save_store(store):
    folder = os.path.dirname(MEMORY_PATH)
    os.makedirs(folder, exist_ok=True)
    with open(MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2)


def get_session(session_id):
    store = load_store()
    sess = store["sessions"].get(session_id)
    if not sess:
        sess = {"turns": [], "last_record_id": None}
    return sess


def reset_session(session_id):
    store = load_store()
    store["sessions"][session_id] = {"turns": [], "last_record_id": None}
    save_store(store)
    return store["sessions"][session_id]


def append_turn(session_id, role, content, record_id=None):
    store = load_store()
    sess = store["sessions"].get(session_id) or {"turns": [], "last_record_id": None}
    sess["turns"].append({"role": role, "content": content})
    if record_id:
        sess["last_record_id"] = record_id
    store["sessions"][session_id] = sess
    save_store(store)
    return sess
