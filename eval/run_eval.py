import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.retrieval import search_bm25, search_dense, hybrid_search

def load_dataset():
    with open("eval/dataset.json", "r") as f:
        return json.load(f)

def calculate_metrics(results, expected_id, k=5):
    retrieved_ids = [f"{r['doc_id']}_{r['article_number']}" for r in results[:k]]
    
    precision = 1.0 if expected_id in retrieved_ids else 0.0
    recall = 1.0 if expected_id in retrieved_ids else 0.0
    
    mrr = 0.0
    for rank, rid in enumerate(retrieved_ids):
        if rid == expected_id:
            mrr = 1.0 / (rank + 1)
            break
            
    return precision, recall, mrr

def run_evaluation():
    os.makedirs("eval", exist_ok=True)
    eval_queries = load_dataset()
    
    results_md = "# Retrieval Evaluation Results\n\n"
    results_md += "| Method | Query Type | Precision@5 | Recall@5 | MRR |\n"
    results_md += "|---|---|---|---|---|\n"
    
    metrics = {
        "bm25": {"lookup": [0,0,0], "conceptual": [0,0,0], "compound": [0,0,0]},
        "dense": {"lookup": [0,0,0], "conceptual": [0,0,0], "compound": [0,0,0]},
        "hybrid": {"lookup": [0,0,0], "conceptual": [0,0,0], "compound": [0,0,0]}
    }
    counts = {"lookup": 0, "conceptual": 0, "compound": 0}
    
    for item in eval_queries:
        q = item["query"]
        qtype = item["query_type"]
        expected = item["expected_article_id"]
        counts[qtype] += 1
        
        bm25_res = search_bm25(q, k=5)
        dense_res = search_dense(q, k=5)
        hybrid_res = hybrid_search(q, k=5)
        
        p, r, m = calculate_metrics(bm25_res, expected)
        metrics["bm25"][qtype] = [metrics["bm25"][qtype][0]+p, metrics["bm25"][qtype][1]+r, metrics["bm25"][qtype][2]+m]
        
        p, r, m = calculate_metrics(dense_res, expected)
        metrics["dense"][qtype] = [metrics["dense"][qtype][0]+p, metrics["dense"][qtype][1]+r, metrics["dense"][qtype][2]+m]
        
        p, r, m = calculate_metrics(hybrid_res, expected)
        metrics["hybrid"][qtype] = [metrics["hybrid"][qtype][0]+p, metrics["hybrid"][qtype][1]+r, metrics["hybrid"][qtype][2]+m]

    for method in ["bm25", "dense", "hybrid"]:
        for qtype in ["lookup", "conceptual", "compound"]:
            c = counts[qtype]
            if c > 0:
                p = metrics[method][qtype][0] / c
                r = metrics[method][qtype][1] / c
                m = metrics[method][qtype][2] / c
                results_md += f"| {method} | {qtype} | {p:.2f} | {r:.2f} | {m:.2f} |\n"
                
    with open("eval/results.md", "w") as f:
        f.write(results_md)
    print("Evaluation complete. Check eval/results.md")

if __name__ == "__main__":
    run_evaluation()
