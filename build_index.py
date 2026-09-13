from rag.chunking import build_chunk_records, load_documents
from rag.retrieval import build_collections


def main():
    docs = load_documents()
    print("kb documents:", len(docs))
    for d in docs:
        print(" -", d["doc_id"], "|", d["topic"])
    print("fixed chunks:", len(build_chunk_records("fixed")))
    print("sentence chunks:", len(build_chunk_records("sentence")))
    built = build_collections()
    print("collections:", built)


if __name__ == "__main__":
    main()
