import json
import pickle
import uuid
from pathlib import Path
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

def build_indices(qdrant_client=None, session_id="default", new_chunks=None):
    session_dir = Path(f"sessions/{session_id}")
    corpus_path = session_dir / "corpus.jsonl"
    
    if not corpus_path.exists():
        print(f"Corpus not found at {corpus_path}")
        return
        
    articles = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            articles.append(json.loads(line.strip()))
            
    # BM25 Indexing (Must rebuild entire corpus)
    print("Building BM25 index...")
    tokenized_corpus = [doc["text"].lower().split() for doc in articles]
    bm25 = BM25Okapi(tokenized_corpus)
    with open(session_dir / "bm25_index.pkl", "wb") as f:
        pickle.dump(bm25, f)
    with open(session_dir / "bm25_docs.pkl", "wb") as f:
        pickle.dump(articles, f)
        
    # Dense Indexing
    print("Building Dense Vector index...")
    model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    
    close_after = False
    if qdrant_client is None:
        qdrant_client = QdrantClient(path=str(session_dir / "qdrant"))
        close_after = True
    
    # Check if collection exists
    try:
        qdrant_client.get_collection("general_docs")
        collection_exists = True
    except Exception:
        collection_exists = False

    if new_chunks is None or not collection_exists:
        qdrant_client.recreate_collection(
            collection_name="general_docs",
            vectors_config=VectorParams(size=model.get_sentence_embedding_dimension(), distance=Distance.COSINE),
        )
        chunks_to_index = articles
    else:
        chunks_to_index = new_chunks
        
    if chunks_to_index:
        texts = [doc["text"] for doc in chunks_to_index]
        embeddings = model.encode(texts)
        
        points = []
        for emb, doc in zip(embeddings, chunks_to_index):
            # Deterministic UUID based on file and chunk
            doc_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc['source_file']}_{doc['chunk_id']}"))
            
            # Deduplication: Check if near-duplicate exists (cosine sim > 0.95)
            is_duplicate = False
            if collection_exists:
                try:
                    res = qdrant_client.search(
                        collection_name="general_docs",
                        query_vector=emb.tolist(),
                        limit=1
                    )
                    if res and res[0].score > 0.95:
                        print(f"Skipping duplicate chunk: {doc['source_file']} {doc['chunk_id']} (similarity: {res[0].score:.3f})")
                        is_duplicate = True
                except Exception:
                    pass
            
            if not is_duplicate:
                points.append(PointStruct(id=doc_uuid, vector=emb.tolist(), payload=doc))
            
        if points:
            print(f"Upserting {len(points)} new chunks to Qdrant...")
            qdrant_client.upsert(collection_name="general_docs", points=points)
        else:
            print("No new chunks to upsert (all were duplicates).")
        
    print("Indexing complete.")
    if close_after:
        qdrant_client.close()

if __name__ == "__main__":
    build_indices()
