import sys
import json
import os
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.refiner import refine_query
from app.retrieval import hybrid_reranked_search
from app.generator import generate_answer

def main():
    parser = argparse.ArgumentParser(description="Ask a question about EU financial regulations.")
    parser.add_argument("question", type=str, help="The question to ask")
    args = parser.parse_args()
    
    question = args.question
    print(f"Question: {question}")
    
    print("\n--- Refining Query ---")
    refinement = refine_query(question)
    print(json.dumps(refinement, indent=2))
    
    print("\n--- Retrieving ---")
    all_contexts = []
    seen_ids = set()
    
    for subq in refinement.get("rewritten_queries", [question]):
        results = hybrid_reranked_search(subq, k=5)
        for res in results:
            doc_id = f"{res['doc_id']}_{res['article_number']}"
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                all_contexts.append(res)
                print(f"Retrieved: {doc_id} - {res['article_title']} (RRF Score: {res.get('rrf_score', 0):.4f})")
                
    print("\n--- Generating ---")
    generation = generate_answer(question, all_contexts)
    print(json.dumps(generation, indent=2))

if __name__ == "__main__":
    main()
