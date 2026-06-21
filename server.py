from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import json
from scripts.build_index import build_indices
import app.retrieval
from app.refiner import refine_query
from app.retrieval import hybrid_reranked_search
from app.generator import generate_answer

app = FastAPI(title="EU FinReg RAG")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str

class IngestRequest(BaseModel):
    doc_id: str
    article_number: str
    article_title: str
    text: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/query")
def query_endpoint(req: QueryRequest):
    refinement = refine_query(req.question)
    
    all_contexts = []
    seen_ids = set()
    
    for subq in refinement.get("rewritten_queries", [req.question]):
        results = hybrid_reranked_search(subq, k=5)
        for res in results:
            doc_id = f"{res['doc_id']}_{res['article_number']}"
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                all_contexts.append(res)
                
    generation = generate_answer(req.question, all_contexts)
    
    return {
        "question": req.question,
        "refinement": refinement,
        "retrieval_path": "hybrid",
        "generation": generation,
        "sources": all_contexts
    }

@app.get("/stats")
def get_stats():
    stats = {}
    try:
        with open("data/corpus.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line.strip())
                doc_id = doc.get("doc_id", "UNKNOWN").upper()
                stats[doc_id] = stats.get(doc_id, 0) + 1
    except Exception:
        pass
    # Convert to a list of objects for the frontend table
    return [{"doc_id": k, "count": v} for k, v in stats.items()]

@app.post("/ingest")
def ingest_data(req: IngestRequest):
    new_doc = {
        "doc_id": req.doc_id,
        "article_number": req.article_number,
        "article_title": req.article_title,
        "text": req.text,
        "source_url": ""
    }
    
    with open("data/corpus.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(new_doc) + "\n")
        
    return {"status": "success", "message": "Document added to corpus. Remember to rebuild indices!"}

@app.post("/build_index")
def rebuild_index():
    build_indices()
    app.retrieval.load_indices()
    return {"status": "success", "message": "Indices successfully rebuilt"}

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
