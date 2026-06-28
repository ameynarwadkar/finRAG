import pickle
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from langfuse import observe
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

current_session_id = None

def load_indices(session_id="default"):
    global bm25, bm25_docs, dense_model, qdrant, cross_encoder, flash_ranker, current_session_id
    
    if current_session_id == session_id and qdrant is not None and bm25 is not None:
        return # Already loaded
        
    if qdrant:
        try:
            qdrant.close()
        except:
            pass
            
    current_session_id = session_id
    session_dir = os.path.join(BASE_DIR, "sessions", session_id)
    os.makedirs(session_dir, exist_ok=True)

    # Load BM25
    try:
        with open(os.path.join(session_dir, "bm25_index.pkl"), "rb") as f:
            bm25 = pickle.load(f)
        with open(os.path.join(session_dir, "bm25_docs.pkl"), "rb") as f:
            bm25_docs = pickle.load(f)
    except Exception:
        bm25 = None
        bm25_docs = []

    # Load Dense
    try:
        if not dense_model:
            dense_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
        qdrant = QdrantClient(path=os.path.join(session_dir, "qdrant"))
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


@observe(as_type="retrieval")
def search_bm25(query: str, k=10, source_filter: str = None):
    if not bm25: return []
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = scores.argsort()[::-1]
    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            doc = bm25_docs[idx].copy()
            if source_filter and doc.get("source_file") != source_filter:
                continue
            doc["score"] = scores[idx]
            results.append(doc)
            if len(results) >= k:
                break
    return results

@observe(as_type="retrieval")
def search_dense(query: str, k=10, source_filter: str = None):
    if not qdrant or not dense_model: return []
    vector = dense_model.encode(query).tolist()
    
    from qdrant_client.http import models
    query_filter = None
    if source_filter:
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="source_file",
                    match=models.MatchValue(value=source_filter)
                )
            ]
        )

    try:
        response = qdrant.query_points(
            collection_name="general_docs",
            query=vector,
            limit=k,
            query_filter=query_filter
        )
        return [{"score": res.score, **res.payload} for res in response.points]
    except Exception as e:
        print(f"Error searching Qdrant: {e}")
        return []

def hybrid_search(query: str, k=10, c=60, source_filter: str = None):
    bm25_results = search_bm25(query, k=k, source_filter=source_filter)
    dense_results = search_dense(query, k=k, source_filter=source_filter)
    
    rrf_scores = {}
    docs_by_id = {}
    
    for rank, doc in enumerate(bm25_results):
        source_file = f"{doc['source_file']}_{doc['chunk_id']}"
        rrf_scores[source_file] = rrf_scores.get(source_file, 0.0) + 1.0 / (rank + 1 + c)
        docs_by_id[source_file] = doc
        
    for rank, doc in enumerate(dense_results):
        source_file = f"{doc['source_file']}_{doc['chunk_id']}"
        rrf_scores[source_file] = rrf_scores.get(source_file, 0.0) + 1.0 / (rank + 1 + c)
        docs_by_id[source_file] = doc
        
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    final_results = []
    for source_file, score in sorted_docs[:k]:
        doc = docs_by_id[source_file]
        doc["rrf_score"] = score
        final_results.append(doc)
        
    return final_results

@observe(as_type="retrieval")
def hybrid_reranked_search(query: str, k=5, retrieve_k=20, source_filter: str = None):
    # Fetch more candidates initially
    candidates = hybrid_search(query, k=retrieve_k, source_filter=source_filter)
    
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

def flash_reranked_search(query: str, k=5, retrieve_k=20, source_filter: str = None):
    candidates = hybrid_search(query, k=retrieve_k, source_filter=source_filter)
    
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
