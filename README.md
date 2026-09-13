# Nykaa Support Agent

Nykaa  
E-commerce & Retail

This repository is a keyless MOCK_LLM support agent for Nykaa retail queries. It covers dataset generation, two RAG chunking strategies, a LangGraph agent with memory, tools and guardrails, FastAPI, RAG-triad evaluation, MCP, SQLite checkpointing, timeouts and retries.

No API key is required. No paid account or credit card is required. Graded transcripts were produced with MOCK_LLM.

## Port

Use **8001** on every machine.

| What | URL |
| --- | --- |
| App | http://127.0.0.1:8001 |
| Swagger docs | http://127.0.0.1:8001/docs |
| Ask | POST http://127.0.0.1:8001/ask |
| Add document | POST http://127.0.0.1:8001/add-document |
| MCP | http://127.0.0.1:8001/mcp |

If 8001 is busy, `run_server.py` prints the next free port from 8002, 8080, 8081, 8088, 8888, 9000, 9001. The MCP client default is 8001.

## Take this project to another device

1. Copy `nykaa-support-agent.zip` only. Do not copy `.venv` or `chroma_db`.
2. On the other device install **Python 3.10 or newer**. Check with `py -3 --version` or `python --version`.
3. Unzip the folder. Keep the unzipped folder name `nykaa-support-agent`.
4. First run needs internet (pip packages + local `all-MiniLM-L6-v2` download). After that it runs offline with MOCK_LLM.
5. Windows: double-click `setup.bat` or run it from that folder. Mac/Linux: `chmod +x setup.sh` then `./setup.sh`.
6. When setup finishes, start the API with `start.bat` or `python run_server.py`.
7. Open http://127.0.0.1:8001

## Run the whole project (manual)

From the unzipped `nykaa-support-agent` folder:

```
py -3 -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
python dataset.py
python build_index.py
python run_demos.py
python -m pytest -q
python check_ready.py
python run_server.py
```

On Mac/Linux replace the first two lines with `python3 -m venv .venv` and `source .venv/bin/activate`.

With the server running, open a **second** terminal in the same folder and test MCP:

```
.\.venv\Scripts\activate
python mcp_svc\client.py
```

Try a policy question:

```
curl -X POST http://127.0.0.1:8001/ask -H "Content-Type: application/json" -d "{\"query\":\"What is the COD refund time?\",\"session_id\":\"s1\"}"
```

Or use http://127.0.0.1:8001/docs and the `POST /ask` box.

Try an order question: `What is the status of order NYK-1001?`

## Dataset design

File: `dataset.py`  
Reproduce with `py -3 dataset.py` (same seed, same list).

| Item | Value |
| --- | --- |
| Seed | 42 |
| Orders | 50 |
| Category weights | Apparel 0.20, Electronics 0.16, Home 0.14, Footwear 0.18, Beauty 0.32 |
| Status weights | Placed 0.16, Shipped 0.22, Delivered 0.40, Returned 0.12, Refunded 0.10 |
| Amount range | INR 199 to 14999 |
| Delayed-shipment probability | 0.22 |

Price-range reasoning: Nykaa mixes drugstore beauty, apparel and a smaller electronics/home range, so 199-14999 INR covers everyday SKUs without luxury outliers.

Observed counts with seed 42:

- Category: Footwear 6, Beauty 18, Apparel 12, Home 8, Electronics 6
- Status: Placed 9, Delivered 20, Shipped 11, Refunded 5, Returned 5
- Delayed shipment: 22.00% (inside 10-30%; no row was hand-edited)

Every order has `record_id`, `category`, `status`, `order_value_inr`, `days_since_created` (0-30) and `delayed_shipment` (boolean).

## Knowledge base

12 documents in `knowledge_base/`, 2-5 sentences each:

1. Return window by product category
2. COD refund timelines
3. Delivery SLAs
4. Reverse-pickup eligibility
5. Warranty terms by category
6. Order-cancellation policy
7. Loyalty-points redemption policy
8. Payment-failure/retry policy
9. Size-exchange policy
10. Damaged-item claim process
11. International shipping restrictions
12. Customer-support escalation matrix

## RAG

Two chunking strategies, two Chroma collections, same local embedder `all-MiniLM-L6-v2`.

| Strategy | Collection | Rule |
| --- | --- | --- |
| Fixed size + overlap | `nykaa_fixed` | 220 characters, 40 overlap |
| Sentence | `nykaa_sentence` | one sentence per chunk |

Build both indexes:

```
py -3 build_index.py
```

User query -> top-k chunks -> MOCK_LLM answer using only retrieved context. If top-1 cosine is below the calibrated threshold the answer is `I don't know`.

### Threshold calibration

Top-1 cosine similarity was measured on the sentence collection.

In-scope (3):

- What is the return window for apparel and beauty products?
- How long does a COD refund take after the warehouse receives the return?
- What is the delivery SLA for metro pin codes?

Out-of-scope (2):

- Who won the cricket world cup last year?
- How do I file income tax returns in India?

Measured top-1 cosine (sentence collection):

- In-scope: 0.4904, 0.7629, 0.6976
- Out-of-scope: 0.1298, 0.2497
- Chosen threshold: 0.35 (in the gap between 0.25 and 0.49)

Same numbers are also in `transcripts/task4_threshold.txt`.

Grounded-generation demos: 5 in-scope queries plus 1 out-of-scope fallback in `transcripts/task4_grounded_generation.txt`.

### Strategy comparison

Same 5 in-scope queries, both collections, Precision@3 and Recall@3 after mapping chunks back to parent documents and deduplicating. Per-query arithmetic is in `transcripts/task5_rag_comparison.txt`.

Both collections scored average Precision@3 = 0.333 and Recall@3 = 1.000. Sentence retrieval had 3 off-topic parent docs in the top-3 versus 6 for fixed, so the recommended collection is `nykaa_sentence`.

## Order tool and escalation

`check_order_status(record_id)` returns `status`, `order_value_inr` and `escalation_score`.

```
escalation_score = 0.60 * delayed_flag + 0.40 * (days_since_created / 30)
```

`delayed_flag` is 1 when `delayed_shipment` is true, else 0. The score is a number in 0-1, not a boolean.

On seed 42 the 80th percentile of `days_since_created` is 23 and the 80th percentile of this score is 0.632. Escalation threshold is **0.63**, matching that score percentile so only the oldest / delayed tail is flagged.

## LangGraph agent

Nodes: `guard` -> `classify` -> (`rag` | `order`) -> `format` (5 nodes).  
One conditional edge after `classify`: policy questions go to RAG, order-status questions go to `check_order_status`. Both routes are shown in `transcripts/task7_agent_routes.txt`.

Conversation memory is a JSON file at `data/memory.json`. Transcript 1 keeps `NYK-1012` across turns. Transcript 2 is a fresh session with no remembered id. See `transcripts/task8_memory.txt`.

Every agent reply is validated against the JSON Schema in `agent/schemas.py`.

## Guardrails

1. Input PII masking for 10-digit phone numbers and payment-card last 4. Customer name and delivery address are out of scope. Model and logs only see the masked string.
2. Prompt-injection detection on phrases such as "ignore previous instructions".
3. Output groundedness check against retrieved context.

Deliberate firing cases: `transcripts/task10_guardrails.txt`.

## FastAPI

```
python run_server.py
```

Port is **8001**.

| Method | Path | Body |
| --- | --- | --- |
| POST | `/ask` | `{"query": "...", "session_id": "s1"}` |
| POST | `/add-document` | `{"title": "...", "text": "..."}` |

Pydantic models are on both requests and responses. Each request writes one JSONL line to `logs/requests.jsonl` with `trace_id` and `latency_ms`. PII is masked before the model and before the log.

MCP is mounted on the same app at `http://127.0.0.1:8001/mcp` (not the site root).

## RAG triad

Exactly 15 queries under MOCK_LLM: one for each of the 12 KB topics, one extra COD/return query, one out-of-scope query and one edge query. Each row has context relevance, groundedness and answer relevance. Averages of all three are at the bottom of `transcripts/task13_rag_triad.txt`.

## MCP

Package used: `fastmcp`.

`check_order_status` is wrapped as an MCP tool with a docstring. The HTTP server lives at `/mcp`.

The client is a separate process:

```
py -3 mcp_svc/client.py
```

It calls two record ids (`NYK-1001`, `NYK-1010`) and prints the standardized MCP responses.

## SQLite checkpointing

Package used: `langgraph-checkpoint-sqlite`.  
Database: `checkpoints.sqlite`. Thread id is the key.

`py -3 -c "from resilience.checkpointing import run_checkpoint_demo; print(run_checkpoint_demo())"`

The demo runs `intake` and `route`, stops, then resumes the same thread. `transcripts/task15_checkpointing.txt` and `transcripts/checkpoint_exec.log` show the first two nodes were loaded and not executed again.

## Timeouts and retries

| Setting | Value |
| --- | --- |
| Per-node timeout | 1.0s (simulated 3s call) |
| Global timeout | 2.0s (simulated 2.4s graph) |
| Retry max attempts | 3 |
| Initial interval | 0.2s |
| Max interval | 2.0s |
| Jitter | 0.1s |
| Backoff | exponential, `delay *= 2` |

Attempt 1 and 2 fail, attempt 3 succeeds. Transcript: `transcripts/task16_timeouts_retries.txt`.

First embedder download is local and free (`all-MiniLM-L6-v2`). MOCK_LLM stays on in `config.py`.

## Layout

```
dataset.py
knowledge_base/
rag/
agent/
api/
mcp_svc/
resilience/
tests/
transcripts/
README.md
requirements.txt
```
