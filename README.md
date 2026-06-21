# EU Financial Regulation Hybrid RAG System

This project is an end-to-end Retrieval-Augmented Generation (RAG) system built over EU financial regulation texts (MiFID II, PSD2, GDPR, DORA). It demonstrates advanced retrieval techniques including query refinement, hybrid search (lexical + dense), reciprocal rank fusion, and schema-constrained LLM generation.

## Architecture

1. **Ingestion & Indexing**: Raw HTML acts are scraped from EUR-Lex, parsed into discrete articles, and embedded into two distinct indices:
   - A Lexical BM25 index (`rank_bm25`).
   - A Dense Vector index (`Qdrant` + `sentence-transformers`).
2. **Query Refinement Layer**: User queries are analyzed by an LLM (Azure OpenAI) to classify their intent (`lookup`, `conceptual`, or `compound`) and rewrite/decompose them for optimal retrieval.
3. **Hybrid Retrieval**: Decomposed queries hit both the BM25 and Dense indices. Results are fused and ranked using Reciprocal Rank Fusion (RRF).
4. **Generation**: The top context chunks are passed to the LLM to generate an answer strictly constrained by a JSON schema, ensuring claims are explicitly cited to the relevant EU Article.

## Setup Instructions

1. Install dependencies using `uv`:
   ```bash
   uv sync
   ```
2. Set your Azure OpenAI environment variables in a `.env` file:
   ```
   AZURE_OPENAI_API_KEY="..."
   AZURE_OPENAI_ENDPOINT="..."
   AZURE_OPENAI_DEPLOYMENT="..."
   ```
3. Scrape the data and build the indices:
   ```bash
   uv run python scripts/ingest.py
   uv run python scripts/build_index.py
   ```
4. Ask a question via the CLI or start the API:
   ```bash
   uv run python scripts/ask.py "What are the rules around algorithmic trading in MiFID II?"
   uv run uvicorn server:app --reload
   ```

## Design Decisions: Why Hybrid Search for Legal Texts?

Legal documents present a unique challenge for RAG systems:
- **Lexical/Keyword matching (BM25)** excels at "lookup" queries. When a user asks "What does Article 25 of MiFID II say?", BM25 immediately zeroes in on exact lexical matches like "Article 25". Dense embeddings often struggle with these exact numerical or ID references.
- **Dense semantic search** excels at "conceptual" queries. When a user asks "Can we process data without consent?", dense embeddings map the semantic intent to the "Right to erasure" or "Lawfulness of processing" articles even if the exact keyword isn't present.

By combining both using **Reciprocal Rank Fusion (RRF)**, the system achieves robust recall across all query types, mitigating the weaknesses of relying on a single retrieval method. Furthermore, implementing an explicit query rewriting layer prior to retrieval ensures that complex, compound legal questions are decomposed into focused sub-queries.

## Evaluation Results
Run the evaluation harness using:
```bash
uv run python eval/run_eval.py
```
*(Check `eval/results.md` for the latest metrics comparing BM25, Dense, and Hybrid performance across query types).*

## Path to Enterprise Production

While the core RAG logic (Hybrid Search, Reranking, Refinement) is production-grade, the infrastructure must be scaled to support enterprise concurrency, robust data management, and security.

### 1. Asynchronous Architecture & Inference Serving
- **Current State**: `SentenceTransformer` and `CrossEncoder` run synchronously on the CPU within the FastAPI request cycle, blocking concurrent users.
- **Production Spec**: Offload embedding and reranking models to dedicated GPU inference servers (e.g., **NVIDIA Triton**, **vLLM**, or **TensorRT-LLM**). Ensure all FastAPI route handlers (`async def`) and database drivers use non-blocking asynchronous IO.

### 2. Distributed Vector & Document Storage
- **Current State**: Documents are stored in a local flat file (`corpus.jsonl`). Lexical search uses a Pickled Python object (`BM25Okapi`). Vectors use a local SQLite Qdrant directory.
- **Production Spec**: 
  - Migrate raw text, document metadata, and chat histories to a relational database (**PostgreSQL**).
  - Migrate local Qdrant to a distributed vector database (e.g., **Qdrant Cloud**, **Pinecone**, or **Milvus**) for horizontal scaling.
  - Replace the pickled BM25 implementation with a distributed inverted-index search engine like **Elasticsearch** or **OpenSearch**.

### 3. Event-Driven Ingestion Pipeline
- **Current State**: Ingestion and indexing are synchronous. Calling the `build_index` function freezes the server while it wipes and recomputes the entire corpus.
- **Production Spec**: Implement an event-driven ingestion pipeline. When a new regulatory act is uploaded, dispatch an asynchronous task via a message broker (**RabbitMQ** or **Kafka**) to background workers (e.g., **Celery**). Workers should extract text, chunk it, embed it, and perform an *incremental upsert* into the vector database without affecting live search traffic.

### 4. Telemetry & LLM Observability
- **Current State**: Metrics are computed locally via static scripts.
- **Production Spec**: Integrate LLM observability platforms (e.g., **Langfuse**, **Arize Phoenix**, or **Helicone**) to trace token usage, monitor live Ragas metrics (Faithfulness, Answer Relevancy), and flag hallucination degradation in real-time. Use **Datadog** or **Prometheus/Grafana** for traditional API latency tracking.

### 5. Security & Access Control
- **Current State**: Open API with hardcoded, wildcard CORS.
- **Production Spec**: 
  - Add **OAuth2 / JWT Authentication** (e.g., Azure AD, Auth0).
  - Implement API Rate Limiting to prevent LLM quota exhaustion.
  - Introduce **Prompt Injection Guardrails** (e.g., NeMo Guardrails) before routing queries to the generator LLM.


## Completed Enhancements
- [x] **Frontend UI**: Built a responsive, glassmorphic HTML/VanillaJS frontend with dynamic citation parsing and an administration dashboard.
- [x] **Granular Inline Citations**: Refined the LLM generation prompt to output exact inline citations and leveraged browser Text Fragments (`#:~:text=`) to auto-scroll directly to the cited EUR-Lex article.
- [x] **Ragas Evaluation**: Automated QA generation and benchmarked pipeline outputs against GPT-4 evaluators.
