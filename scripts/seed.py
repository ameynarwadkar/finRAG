import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
from app import retrieval as retrieval_module
from scripts.build_index import build_indices

def seed():
    session_id = "default"
    os.makedirs(f"sessions/{session_id}", exist_ok=True)
    corpus_path = f"sessions/{session_id}/corpus.jsonl"
    
    # Check if already seeded
    if os.path.exists(corpus_path):
        print("Corpus already exists. Skipping seed.")
        return

    print("Creating sample documentation corpus...")
    docs = [
        {
            "source_file": "SAMPLE_DOC",
            "chunk_id": "1",
            "section_heading": "Introduction to AnyRAG",
            "text": "AnyRAG is a highly advanced Retrieval-Augmented Generation system. It uses a Hybrid RRF retrieval methodology combining Dense Vectors and BM25 Sparse embeddings, followed by a Cross-Encoder reranker.",
            "source_url": ""
        },
        {
            "source_file": "SAMPLE_DOC",
            "chunk_id": "2",
            "section_heading": "Deployment",
            "text": "AnyRAG is containerized using Docker and Docker-Compose. The FastAPI backend serves the React/Vanilla JS dashboard directly, while Qdrant runs embedded within the Python process to manage the vector space locally.",
            "source_url": ""
        }
    ]
    
    with open(corpus_path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")
            
    print("Building Vector Indices for the seed corpus...")
    retrieval_module.load_indices(session_id)
    build_indices(retrieval_module.qdrant, session_id=session_id)
    print("Seeding complete! You can now ask questions about AnyRAG in the UI.")

if __name__ == "__main__":
    seed()
