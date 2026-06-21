import pickle
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
try:
    from flashrank import Ranker, RerankRequest
except ImportError:
    Ranker = None

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

bm25 = None
bm25_docs = []
dense_model = None
qdrant = None
cross_encoder = None
flash_ranker = None

def load_indices():
    global bm25, bm25_docs, dense_model, qdrant, cross_encoder, flash_ranker
    
    if qdrant:
        try:
            qdrant.close()
        except:
            pass

    # Load BM25
    try:
        with open(os.path.join(BASE_DIR, "data/bm25_index.pkl"), "rb") as f:
            bm25 = pickle.load(f)
        with open(os.path.join(BASE_DIR, "data/bm25_docs.pkl"), "rb") as f:
            bm25_docs = pickle.load(f)
    except Exception:
        bm25 = None
        bm25_docs = []

    # Load Dense
    try:
        if not dense_model:
            dense_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
        qdrant = QdrantClient(path=os.path.join(BASE_DIR, "data/qdrant"))
    except Exception:
        dense_model = None
        qdrant = None

    # Load Cross-Encoder
    try:
        if not cross_encoder:
            cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    except Exception:
        cross_encoder = None

    # Load FlashRank
    try:
        if Ranker and not flash_ranker:
            flash_ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir=os.path.join(BASE_DIR, "data", "flashrank"))
    except Exception:
        flash_ranker = None

load_indices()


def search_bm25(query: str, k=10):
    if not bm25: return []
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = scores.argsort()[-k:][::-1]
    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            doc = bm25_docs[idx].copy()
            doc["score"] = scores[idx]
            results.append(doc)
    return results

def search_dense(query: str, k=10):
    if not qdrant or not dense_model: return []
    vector = dense_model.encode(query).tolist()
    try:
        response = qdrant.query_points(
            collection_name="eu_regs",
            query=vector,
            limit=k
        )
        return [{"score": res.score, **res.payload} for res in response.points]
    except Exception as e:
        print(f"Error searching Qdrant: {e}")
        return []

def hybrid_search(query: str, k=10, c=60):
    bm25_results = search_bm25(query, k=k)
    dense_results = search_dense(query, k=k)
    
    rrf_scores = {}
    docs_by_id = {}
    
    for rank, doc in enumerate(bm25_results):
        doc_id = f"{doc['doc_id']}_{doc['article_number']}"
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (rank + 1 + c)
        docs_by_id[doc_id] = doc
        
    for rank, doc in enumerate(dense_results):
        doc_id = f"{doc['doc_id']}_{doc['article_number']}"
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (rank + 1 + c)
        docs_by_id[doc_id] = doc
        
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    final_results = []
    for doc_id, score in sorted_docs[:k]:
        doc = docs_by_id[doc_id]
        doc["rrf_score"] = score
        final_results.append(doc)
        
    return final_results

def hybrid_reranked_search(query: str, k=5, retrieve_k=20):
    # Fetch more candidates initially
    candidates = hybrid_search(query, k=retrieve_k)
    
    if not cross_encoder or not candidates:
        return candidates[:k]
        
    # Score query-document pairs
    pairs = [[query, doc["text"]] for doc in candidates]
    scores = cross_encoder.predict(pairs)
    
    for idx, score in enumerate(scores):
        candidates[idx]["cross_score"] = float(score)
        
    # Sort candidates by cross-encoder score
    reranked = sorted(candidates, key=lambda x: x["cross_score"], reverse=True)
    
    return reranked[:k]

def flash_reranked_search(query: str, k=5, retrieve_k=20):
    candidates = hybrid_search(query, k=retrieve_k)
    
    if not flash_ranker or not candidates:
        return candidates[:k]
        
    # Format for FlashRank
    passages = []
    for i, doc in enumerate(candidates):
        passages.append({
            "id": i,
            "text": doc["text"],
            "meta": doc
        })
        
    rerankRequest = RerankRequest(query=query, passages=passages)
    reranked_results = flash_ranker.rerank(rerankRequest)
    
    # Reconstruct the original candidate format
    final_results = []
    for res in reranked_results[:k]:
        doc = res["meta"]
        doc["flash_score"] = res["score"]
        final_results.append(doc)
        
    return final_results
