import json
import pickle
from pathlib import Path
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

def build_indices():
    corpus_path = Path("data/corpus.jsonl")
    if not corpus_path.exists():
        print("Corpus not found at data/corpus.jsonl")
        return
        
    articles = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            articles.append(json.loads(line.strip()))
            
    # BM25 Indexing
    print("Building BM25 index...")
    tokenized_corpus = [doc["text"].lower().split() for doc in articles]
    bm25 = BM25Okapi(tokenized_corpus)
    with open("data/bm25_index.pkl", "wb") as f:
        pickle.dump(bm25, f)
    with open("data/bm25_docs.pkl", "wb") as f:
        pickle.dump(articles, f)
        
    # Dense Indexing
    print("Building Dense Vector index...")
    model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    client = QdrantClient(path="data/qdrant")
    
    client.recreate_collection(
        collection_name="eu_regs",
        vectors_config=VectorParams(size=model.get_sentence_embedding_dimension(), distance=Distance.COSINE),
    )
    
    texts = [doc["text"] for doc in articles]
    embeddings = model.encode(texts)
    
    points = [
        PointStruct(id=i, vector=emb.tolist(), payload=doc)
        for i, (emb, doc) in enumerate(zip(embeddings, articles))
    ]
    
    client.upsert(collection_name="eu_regs", points=points)
    print("Indexing complete.")

if __name__ == "__main__":
    build_indices()
