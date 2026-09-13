import os
import sys

from config import API_HOST, API_PORT, CATEGORIES, KB_DIR, ROOT, STATUSES, TRANSCRIPT_DIR
from dataset import ORDERS, category_counts, delayed_percentage, status_counts
from rag.chunking import DOC_TOPICS, load_documents

NEEDED_TRANSCRIPTS = [
    "task1_dataset.txt",
    "task4_threshold.txt",
    "task4_grounded_generation.txt",
    "task5_rag_comparison.txt",
    "task6_order_status.txt",
    "task7_agent_routes.txt",
    "task8_memory.txt",
    "task10_guardrails.txt",
    "task11_fastapi.txt",
    "task12_logging.txt",
    "task13_rag_triad.txt",
    "task14_mcp.txt",
    "task15_checkpointing.txt",
    "task16_timeouts_retries.txt",
]


def ok(label, passed):
    print(("%s  %s" % ("PASS" if passed else "FAIL", label)))
    return passed


def main():
    print("Nykaa Support Agent — ready check")
    print("port: %s:%s" % (API_HOST, API_PORT))
    print("python:", sys.version.split()[0])
    print()
    failed = 0

    py_ok = sys.version_info >= (3, 10)
    failed += not ok("python 3.10+", py_ok)

    cats = category_counts()
    stats = status_counts()
    delayed = delayed_percentage()
    failed += not ok("orders >= 40 (%d)" % len(ORDERS), len(ORDERS) >= 40)
    failed += not ok("every category >= 3", all(cats.get(c, 0) >= 3 for c in CATEGORIES))
    failed += not ok("every status present", all(stats.get(s, 0) >= 1 for s in STATUSES))
    failed += not ok("delayed 10-30%% (%.2f)" % delayed, 10 <= delayed <= 30)

    docs = load_documents()
    ids = {d["doc_id"] for d in docs}
    failed += not ok("kb docs >= 12 (%d)" % len(docs), len(docs) >= 12)
    failed += not ok("12 required topics present", set(DOC_TOPICS) <= ids)

    for name in NEEDED_TRANSCRIPTS:
        path = os.path.join(TRANSCRIPT_DIR, name)
        failed += not ok("transcript %s" % name, os.path.isfile(path) and os.path.getsize(path) > 20)

    for rel in [
        "dataset.py",
        "README.md",
        "requirements.txt",
        "run_server.py",
        "build_index.py",
        "run_demos.py",
        "agent/graph.py",
        "agent/tools.py",
        "agent/memory.py",
        "agent/guardrails.py",
        "agent/schemas.py",
        "rag/chunking.py",
        "rag/embeddings.py",
        "rag/retrieval.py",
        "rag/evaluation.py",
        "api/main.py",
        "mcp_svc/server.py",
        "mcp_svc/client.py",
        "resilience/checkpointing.py",
        "resilience/retries.py",
        "resilience/timeouts.py",
    ]:
        failed += not ok("file %s" % rel, os.path.isfile(os.path.join(ROOT, rel)))

    print()
    if failed:
        print("READY: NO  (%d check(s) failed)" % failed)
        return 1
    print("READY: YES")
    print("Next: python run_server.py")
    print("Open: http://%s:%s" % (API_HOST, API_PORT))
    print("Docs: http://%s:%s/docs" % (API_HOST, API_PORT))
    print("MCP:  http://%s:%s/mcp" % (API_HOST, API_PORT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
