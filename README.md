# Nykaa Support Agent

**Track:** E-commerce & Retail — Nykaa

A retrieval-augmented, tool-using support agent for Nykaa customer support — policy questions ("what's the return window on beauty products?") and order lookups ("where is NYK-1012?"). Built on a LangGraph agent, ChromaDB retrieval (two chunking strategies), FastAPI, an MCP tool server, SQLite checkpointing, and timeout/retry resilience. Runs entirely on `MOCK_LLM` — no API key, no paid account, no external LLM calls anywhere in the project.

## Run it

pip install -r requirements.txt
python dataset.py          # generate + validate the 50-order dataset
python build_index.py      # embed the KB into both Chroma collections
python run_demos.py        # regenerate every transcript in transcripts/
pytest -q
python run_server.py       # http://127.0.0.1:8001, docs at /docs

With the server running, in a second terminal:

python -m mcp_svc.client   # standalone MCP client, run as a module from project root

## Dataset

`dataset.py`, seed `42`, 50 orders, deterministic.

| | |
|---|---|
| Categories | Apparel, Electronics, Home, Footwear, Beauty (weights 0.20/0.16/0.14/0.18/0.32) |
| Statuses | Placed, Shipped, Delivered, Returned, Refunded (weights 0.16/0.22/0.40/0.12/0.10) |
| Amount range | ₹199–₹14,999, banded per category |
| Delayed shipment | 22% (validated to stay within 10–30%; script raises otherwise) |

Observed: Footwear 6, Beauty 18, Apparel 12, Home 8, Electronics 6 · Placed 9, Shipped 11, Delivered 20, Returned 5, Refunded 5.

## Knowledge base

13 documents in `knowledge_base/` — all 12 required topics (return window, COD refund, delivery SLAs, reverse pickup, warranty, cancellation, loyalty points, payment failure, size exchange, damaged item, international shipping, escalation matrix) plus one bonus (gift wrap).

## RAG

Two chunking strategies → two Chroma collections, embedded with local `all-MiniLM-L6-v2`:

- **Fixed-size + overlap** (`nykaa_fixed`) — 220 chars, 40 overlap
- **Sentence-based** (`nykaa_sentence`) — one sentence per chunk

Threshold calibrated empirically, not assumed: in-scope similarities 0.49 / 0.76 / 0.70, out-of-scope 0.13 / 0.25 → threshold set at **0.35**. Below it, the agent answers "I don't know."

Both strategies compared on the same 5 queries with Precision@3/Recall@3 and per-query arithmetic — both scored P@3 = 0.333, R@3 = 1.000, but sentence chunking pulled fewer off-topic parent docs, so it's the strategy used at runtime.

## Order tool & escalation

escalation_score = 0.60 * delayed_flag + 0.40 * (days_since_created / 30)

A real 0–1 score, clamped, not a boolean. Threshold **0.63**, set at the 80th percentile of `days_since_created` on this dataset.

## LangGraph agent

5 nodes: `guard → classify → (rag | order) → format`, one conditional edge routing policy questions to RAG and order questions to the order tool.

- **Memory** — JSON-persisted per session; demonstrated both remembering an order ID across turns and a fresh session with no carryover.
- **Structured output** — every response validated against a JSON Schema before returning.
- **Guardrails** — input PII masking (phone, card last-4), prompt-injection detection, and an output groundedness check that refuses ungrounded answers. All three have deliberate firing demonstrations.

## FastAPI

`POST /ask`, `POST /add-document` — Pydantic request/response models. Every call logs one JSONL line with a trace ID and latency to `logs/requests.jsonl`; PII is masked before it reaches the model or the log.

## RAG-triad evaluation

15 queries (12 KB topics + 1 extra + 1 out-of-scope + 1 edge case), each scored on context relevance, groundedness, and answer relevance, plus averages across all 15.

## MCP

`check_order_status` exposed as an MCP tool via `fastmcp`, served at `/mcp` on the same FastAPI app. `mcp_svc/client.py` is a fully separate process that connects over HTTP and looks up two different record IDs.

## Resilience

- **Checkpointing** (`langgraph-checkpoint-sqlite`, `checkpoints.sqlite`) — graph runs 2 nodes, is interrupted, resumes on the same thread, and completes, with proof (an execution log) that the completed nodes were not re-run.
- **Timeouts** — per-node (1.0s) and global (2.0s), both demonstrated firing cleanly against simulated slow calls.
- **Retries** — exponential backoff, max 3 attempts, 0.2s initial / 2.0s max interval, 0.1s jitter; simulated failure recovers on attempt 3.

## Evidence

All of the above is demonstrated with real transcripts in `transcripts/` (tracked in git, not gitignored — it's graded evidence) and regenerated fresh by `run_demos.py`. `check_ready.py` verifies the dataset, KB, and every required transcript are present before submission.

## Layout

dataset.py       knowledge_base/     rag/           agent/
api/             mcp_svc/            resilience/    tests/
transcripts/     README.md          requirements.txt