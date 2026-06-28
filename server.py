from fastapi import FastAPI, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import json
import shutil
import os
from pathlib import Path
from langfuse import observe

from scripts.build_index import build_indices
from scripts.ingest import process_all, process_file
import app.retrieval as retrieval_module
from app.refiner import refine_query
from app.retrieval import hybrid_reranked_search
from app.generator import generate_answer
from app.history import rewrite_query_with_history

app = FastAPI(title="AnyRAG")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str
    session_id: str = "default"
    retrieval_method: str = "hybrid_rrf"
    source_filter: str = None

class IngestRequest(BaseModel):
    source_file: str
    chunk_id: str
    section_heading: str
    text: str
    session_id: str = "default"

class BuildIndexRequest(BaseModel):
    session_id: str = "default"

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/v1/ask")
@observe(name="v1_ask_trace")
def query_endpoint(req: QueryRequest):
    import re
    lower_q = req.question.strip().lower()
    if re.match(r'^(hi|hello|hey|how are you|greetings|good morning|good evening|what\'s up)\b', lower_q):
        return {
            "question": req.question,
            "refinement": {},
            "retrieval_path": "conversational",
            "generation": "Hello! I am AnyRAG. You can upload documents using the panel on the right, or ask questions about previously uploaded documents. How can I help you today?",
            "sources": []
        }
        
    retrieval_module.load_indices(req.session_id)
    
    if retrieval_module.qdrant is None or retrieval_module.bm25 is None:
        return {
            "question": req.question,
            "refinement": {},
            "retrieval_path": "none",
            "generation": "Error: The knowledge base for this session is empty or has not been indexed yet. Please upload documents first.",
            "sources": []
        }
        
    # Load conversation history
    history_file = f"sessions/{req.session_id}/history.json"
    history = []
    if os.path.exists(history_file):
        try:
            with open(history_file, "r") as f:
                history = json.load(f)
        except Exception:
            pass
            
    # Rewrite query using history context
    standalone_query = rewrite_query_with_history(req.question, history)

    # Check Semantic Cache
    query_embedding = None
    if retrieval_module.dense_model:
        query_embedding = retrieval_module.dense_model.encode(standalone_query).tolist()
        from app.cache import generator_cache
        cached_generation = generator_cache.get_semantic(query_embedding)
        if cached_generation:
            history.append({"role": "user", "content": req.question})
            history.append({"role": "assistant", "content": cached_generation.get("answer", "")})
            os.makedirs(f"sessions/{req.session_id}", exist_ok=True)
            with open(history_file, "w") as f:
                json.dump(history, f)
            return {
                "question": req.question,
                "refinement": {},
                "retrieval_path": "semantic_cache",
                "generation": cached_generation,
                "sources": []
            }

    refinement = refine_query(standalone_query)
    
    all_contexts = []
    seen_ids = set()
    
    for subq in refinement.get("rewritten_queries", [standalone_query]):
        if req.retrieval_method == "dense":
            results = retrieval_module.search_dense(subq, k=5, source_filter=req.source_filter)
        elif req.retrieval_method == "bm25":
            results = retrieval_module.search_bm25(subq, k=5, source_filter=req.source_filter)
        else:
            results = hybrid_reranked_search(subq, k=5, source_filter=req.source_filter)
            
        for res in results:
            source_file = f"{res['source_file']}_{res['chunk_id']}"
            if source_file not in seen_ids:
                seen_ids.add(source_file)
                all_contexts.append(res)
                
    if req.retrieval_method == "hybrid_rrf":
        # Global sort across all sub-queries
        all_contexts.sort(key=lambda x: x.get("cross_score", -10.0), reverse=True)
        # Enforce global token/chunk budget
        all_contexts = all_contexts[:5]
        
        # "Lost in the Middle" reordering
        reordered = []
        for i, ctx in enumerate(all_contexts):
            if i % 2 == 0:
                reordered.insert(0, ctx)
            else:
                reordered.append(ctx)
        all_contexts = reordered

    # Graceful "I don't know" / Short-circuit
    is_summarize_intent = bool(re.search(r'\b(summarize|summarise|summary|overview|what does.*say|tell me about)\b', lower_q))
    
    if req.retrieval_method == "hybrid_rrf":
        best_score = max([ctx.get("cross_score", -10.0) for ctx in all_contexts]) if all_contexts else -10.0
        # Bypass strict threshold if it's a summarize intent, but still require at least some context
        if is_summarize_intent and len(all_contexts) > 0:
            is_low_confidence = False
        else:
            is_low_confidence = best_score < -2.0
    else:
        is_low_confidence = len(all_contexts) == 0
        
    if is_low_confidence:
        found_docs = list(set([f"{c.get('source_file')} - {c.get('section_heading')}" for c in all_contexts[:3]]))
        doc_list = "\n".join([f"- {doc}" for doc in found_docs]) if found_docs else "- None"
        
        generation = {
            "answer": f"I don't have enough confident information in the retrieved documents to answer this question accurately without hallucinating.\n\n**What I found tangentially related:**\n{doc_list}\n\nThese documents matched some keywords but do not seem to directly answer your query. You may want to check them manually.",
            "citations": [],
            "confidence": "low",
            "confidence_metrics": {
                "retrieval_confidence": 0.0,
                "citation_coverage": 0.0,
                "completeness": 0.0,
                "composite_score": 0.0
            }
        }
    else:
        generation = generate_answer(standalone_query, all_contexts)
        if query_embedding is not None:
            from app.cache import generator_cache
            generator_cache.set_semantic(query_embedding, generation)
        
    # Save conversation history
    history.append({"role": "user", "content": req.question})
    history.append({"role": "assistant", "content": generation.get("answer", "")})
    os.makedirs(f"sessions/{req.session_id}", exist_ok=True)
    with open(history_file, "w") as f:
        json.dump(history, f)
    
    return {
        "question": req.question,
        "refinement": refinement,
        "retrieval_path": "hybrid",
        "generation": generation,
        "sources": all_contexts
    }

@app.get("/v1/documents")
def get_stats(session_id: str = Query("default")):
    stats = {}
    try:
        with open(f"sessions/{session_id}/corpus.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line.strip())
                source_file = doc.get("source_file", "UNKNOWN").upper()
                stats[source_file] = stats.get(source_file, 0) + 1
    except Exception:
        pass
    return [{"source_file": k, "count": v} for k, v in stats.items()]

@app.post("/v1/ingest")
def ingest_data(req: IngestRequest):
    new_doc = {
        "source_file": req.source_file,
        "chunk_id": req.chunk_id,
        "section_heading": req.section_heading,
        "text": req.text,
        "source_url": ""
    }
    
    os.makedirs(f"sessions/{req.session_id}", exist_ok=True)
    with open(f"sessions/{req.session_id}/corpus.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(new_doc) + "\n")
        
    return {"status": "success", "message": "Document added to corpus. Remember to rebuild indices!"}

@app.post("/v1/upload")
async def upload_file(session_id: str = Form("default"), files: list[UploadFile] = File(...)):
    raw_dir = f"sessions/{session_id}/raw"
    os.makedirs(raw_dir, exist_ok=True)
    
    filenames = []
    all_new_chunks = []
    
    for file in files:
        file_location = f"{raw_dir}/{file.filename}"
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)
        filenames.append(file.filename)
        
        chunks = process_file(Path(file_location), session_id)
        all_new_chunks.extend(chunks)
    
    try:
        retrieval_module.load_indices(session_id)
        build_indices(retrieval_module.qdrant, session_id=session_id, new_chunks=all_new_chunks)
        retrieval_module.load_indices(session_id)
        return {"status": "success", "message": f"Successfully uploaded and incrementally indexed {len(filenames)} files."}
    except Exception as e:
        return {"status": "error", "message": f"Uploaded but failed to index: {str(e)}"}

@app.post("/v1/build_index")
def rebuild_index(req: BuildIndexRequest):
    retrieval_module.load_indices(req.session_id)
    build_indices(retrieval_module.qdrant, session_id=req.session_id)
    retrieval_module.load_indices(req.session_id)
    return {"status": "success", "message": "Indices successfully rebuilt"}

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
