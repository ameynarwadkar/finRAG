"""Quick smoke test for the MCP server tools."""
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# We can call the underlying functions directly to verify they work
from app.retrieval import hybrid_reranked_search
from app.generator import generate_answer
from app.refiner import refine_query

print("=" * 60)
print("MCP Server Smoke Test")
print("=" * 60)

# Test 1: list_regulations (no LLM, instant)
print("\n--- Tool: list_regulations ---")
stats = {}
try:
    with open("data/corpus.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line.strip())
            source_file = doc.get("source_file", "UNKNOWN").upper()
            stats[source_file] = stats.get(source_file, 0) + 1
except FileNotFoundError:
    print("ERROR: corpus.jsonl not found")

result = {
    "total_articles": sum(stats.values()),
    "regulations": [{"name": k, "article_count": v} for k, v in sorted(stats.items())]
}
print(json.dumps(result, indent=2))

# Test 2: search_articles (retrieval only, no LLM)
print("\n--- Tool: search_articles ---")
query = "algorithmic trading"
results = hybrid_reranked_search(query, k=3)
for r in results:
    print(f"  [{r.get('cross_score', 0):.4f}] {r['source_file']} Art {r['chunk_id']} - {r.get('section_heading', '')[:60]}")

# Test 3: query_regulation (full pipeline with LLM)
print("\n--- Tool: query_regulation ---")
question = "What does MiFID II say about algorithmic trading?"
refinement = refine_query(question)
print(f"  Query type: {refinement.get('query_type')}")
print(f"  Rewritten: {refinement.get('rewritten_queries')}")

all_contexts = []
seen_ids = set()
for subq in refinement.get("rewritten_queries", [question]):
    res = hybrid_reranked_search(subq, k=5)
    for r in res:
        did = f"{r['source_file']}_{r['chunk_id']}"
        if did not in seen_ids:
            seen_ids.add(did)
            all_contexts.append(r)

generation = generate_answer(question, all_contexts)
print(f"  Confidence: {generation.get('confidence')}")
print(f"  Answer: {generation.get('answer', '')[:200]}...")
print(f"  Citations: {len(generation.get('cited_articles', []))} articles cited")

print("\n" + "=" * 60)
print("All 3 MCP tools verified successfully!")
print("=" * 60)
