from rag.chunking import DOC_TOPICS, build_chunk_records, load_documents


def test_twelve_topics():
    docs = load_documents()
    ids = {d["doc_id"] for d in docs}
    assert len(docs) >= 12
    assert set(DOC_TOPICS) <= ids


def test_doc_length():
    for doc in load_documents():
        sentences = [s for s in doc["text"].replace("?", ".").split(".") if s.strip()]
        assert 2 <= len(sentences) <= 5


def test_two_chunk_strategies():
    fixed = build_chunk_records("fixed")
    sent = build_chunk_records("sentence")
    assert len(fixed) > 0
    assert len(sent) > 0
    assert {c["strategy"] for c in fixed} == {"fixed"}
    assert {c["strategy"] for c in sent} == {"sentence"}
