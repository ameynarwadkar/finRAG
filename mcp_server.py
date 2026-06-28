"""
MCP Server for EU Financial Regulation RAG System.

Exposes the finRAG pipeline as tools that any MCP-compatible AI agent
(Claude Desktop, Cursor, etc.) can call to query EU regulations.

Run with:
    uv run python mcp_server.py
"""

import json
from fastmcp import FastMCP
from app.refiner import refine_query
from app.retrieval import hybrid_reranked_search, search_bm25, search_dense
from app.generator import generate_answer

mcp = FastMCP("EU FinReg RAG")


@mcp.tool()
def query_regulation(question: str) -> dict:
    """
    Ask a question about EU financial regulations and get an LLM-generated
    answer with chunk-level citations.

    Runs the full RAG pipeline: query refinement → hybrid search (BM25 + Dense + RRF)
    → cross-encoder reranking → schema-constrained LLM generation.

    Args:
        question: A natural-language question about EU financial regulations.
                  Examples:
                  - "What are the strong customer authentication requirements under PSD2?"
                  - "How does MiFID II regulate algorithmic trading?"
                  - "What is the right to erasure under GDPR?"

    Returns:
        A dict with 'answer', 'cited_articles' (source_file, chunk_id,
        quote_or_paraphrase), and 'confidence' (high/medium/low).
    """
    # Step 1: Refine the query (classify + rewrite/decompose)
    refinement = refine_query(question)

    # Step 2: Retrieve across all sub-queries
    all_contexts = []
    seen_ids = set()
    for subq in refinement.get("rewritten_queries", [question]):
        results = hybrid_reranked_search(subq, k=5)
        for res in results:
            source_file = f"{res['source_file']}_{res['chunk_id']}"
            if source_file not in seen_ids:
                seen_ids.add(source_file)
                all_contexts.append(res)

    # Step 3: Generate a cited answer
    generation = generate_answer(question, all_contexts)

    return {
        "question": question,
        "query_type": refinement.get("query_type", "unknown"),
        "answer": generation.get("answer", ""),
        "cited_articles": generation.get("cited_articles", []),
        "confidence": generation.get("confidence", "low"),
    }


@mcp.tool()
def search_articles(query: str, top_k: int = 5) -> list[dict]:
    """
    Search for relevant EU regulation articles without generating an LLM answer.
    Returns the raw retrieved and reranked articles with their metadata and scores.

    Useful when you need the source text of specific articles, or want to
    inspect what the retrieval system finds before asking for a full answer.

    Args:
        query: A search query (natural language or keyword-based).
        top_k: Number of top results to return (default: 5, max: 20).

    Returns:
        A list of chunk dicts, each containing: source_file, chunk_id,
        section_heading, text, and relevance scores.
    """
    top_k = min(max(top_k, 1), 20)
    results = hybrid_reranked_search(query, k=top_k)

    # Return a clean subset of fields
    clean_results = []
    for res in results:
        clean_results.append({
            "source_file": res.get("source_file", ""),
            "chunk_id": res.get("chunk_id", ""),
            "section_heading": res.get("section_heading", ""),
            "text": res.get("text", "")[:500],  # Truncate for readability
            "cross_encoder_score": res.get("cross_score"),
        })
    return clean_results


@mcp.tool()
def list_regulations() -> dict:
    """
    List all EU regulations in the corpus with their article counts.

    Returns the available regulations (e.g. PSD2, MiFID II, GDPR, DORA)
    and how many articles each contains. Use this to understand what
    the knowledge base covers before running a query.

    Returns:
        A dict with 'total_articles' count and a 'regulations' list,
        where each entry has 'name' and 'article_count'.
    """
    stats: dict[str, int] = {}
    try:
        with open("data/corpus.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line.strip())
                source_file = doc.get("source_file", "UNKNOWN").upper()
                stats[source_file] = stats.get(source_file, 0) + 1
    except FileNotFoundError:
        return {"total_articles": 0, "regulations": [], "error": "Corpus file not found."}

    regulations = [
        {"name": name, "article_count": count}
        for name, count in sorted(stats.items())
    ]
    return {
        "total_articles": sum(stats.values()),
        "regulations": regulations,
    }


if __name__ == "__main__":
    mcp.run()
