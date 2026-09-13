import os

ROOT = os.path.dirname(os.path.abspath(__file__))

SEED = 42
N_ORDERS = 50

CATEGORIES = ["Apparel", "Electronics", "Home", "Footwear", "Beauty"]
CATEGORY_WEIGHTS = [0.20, 0.16, 0.14, 0.18, 0.32]

STATUSES = ["Placed", "Shipped", "Delivered", "Returned", "Refunded"]
STATUS_WEIGHTS = [0.16, 0.22, 0.40, 0.12, 0.10]

MIN_AMOUNT = 199
MAX_AMOUNT = 14999
DELAYED_PROB = 0.22

KB_DIR = os.path.join(ROOT, "knowledge_base")
CHROMA_DIR = os.path.join(ROOT, "chroma_db")
MEMORY_PATH = os.path.join(ROOT, "data", "memory.json")
LOG_PATH = os.path.join(ROOT, "logs", "requests.jsonl")
CHECKPOINT_PATH = os.path.join(ROOT, "checkpoints.sqlite")
TRANSCRIPT_DIR = os.path.join(ROOT, "transcripts")

FIXED_COLLECTION = "nykaa_fixed"
SENTENCE_COLLECTION = "nykaa_sentence"
ACTIVE_COLLECTION = SENTENCE_COLLECTION

EMBED_MODEL = "all-MiniLM-L6-v2"
FIXED_CHUNK_SIZE = 220
FIXED_CHUNK_OVERLAP = 40
TOP_K = 3

SIMILARITY_THRESHOLD = 0.35

DELAYED_WEIGHT = 0.60
RECENCY_WEIGHT = 0.40
ESCALATION_THRESHOLD = 0.63

RETRY_MAX_ATTEMPTS = 3
RETRY_INITIAL_INTERVAL = 0.2
RETRY_MAX_INTERVAL = 2.0
RETRY_JITTER = 0.1

NODE_TIMEOUT_SEC = 1.0
GLOBAL_TIMEOUT_SEC = 2.0
MOCK_LLM = True

API_HOST = "127.0.0.1"
API_PORT = 8001
API_PORT_CANDIDATES = (8001, 8002, 8080, 8081, 8088, 8888, 9000, 9001)


def pick_free_port(host=API_HOST, candidates=API_PORT_CANDIDATES):
    import socket

    for port in candidates:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind((host, port))
            return port
        except OSError:
            continue
        finally:
            sock.close()
    raise RuntimeError("no free port in %s" % (candidates,))
