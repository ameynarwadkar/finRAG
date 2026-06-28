import os
import json
from pathlib import Path
from bs4 import BeautifulSoup
import pypdf
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity

RAW_DIR = Path("data/raw")
CORPUS_FILE = Path("data/corpus.jsonl")

def ensure_dirs():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs("data", exist_ok=True)

def semantic_chunking(text, threshold=0.75):
    paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 30]
    if not paragraphs: return [text]
    model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    embeddings = model.encode(paragraphs)
    
    chunks = []
    current_chunk = paragraphs[0]
    
    for i in range(1, len(paragraphs)):
        sim = cosine_similarity([embeddings[i-1]], [embeddings[i]])[0][0]
        if sim > threshold:
            current_chunk += "\n\n" + paragraphs[i]
        else:
            chunks.append(current_chunk)
            current_chunk = paragraphs[i]
    chunks.append(current_chunk)
    return chunks

def apply_chunking(text, strategy):
    if strategy == "recursive":
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        return splitter.split_text(text)
    else: # default to semantic
        return semantic_chunking(text)

def format_chunks(raw_chunks, file_path, strategy):
    chunks = []
    for i, c in enumerate(raw_chunks):
        if not c.strip(): continue
        chunks.append({
            "source_file": file_path.stem,
            "chunk_id": f"{strategy}_{i+1}",
            "section_heading": f"{file_path.stem} - Chunk {i+1}",
            "text": c.strip(),
            "source_url": str(file_path),
            "chunk_strategy": strategy
        })
    return chunks

def parse_pdf(file_path, strategy):
    full_text = ""
    try:
        with open(file_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted: full_text += extracted + "\n\n"
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
    return format_chunks(apply_chunking(full_text, strategy), file_path, strategy)

def parse_html(file_path, strategy):
    full_text = ""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")
            full_text = soup.get_text(separator="\n\n", strip=True)
    except Exception as e:
        print(f"Error reading HTML {file_path}: {e}")
    return format_chunks(apply_chunking(full_text, strategy), file_path, strategy)

def parse_text_markdown(file_path, strategy):
    full_text = ""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            full_text = f.read()
    except Exception as e:
        print(f"Error reading TXT/MD {file_path}: {e}")
    return format_chunks(apply_chunking(full_text, strategy), file_path, strategy)

def get_session_dir(session_id):
    path = Path(f"sessions/{session_id}")
    path.mkdir(parents=True, exist_ok=True)
    (path / "raw").mkdir(parents=True, exist_ok=True)
    return path

def process_file(file_path, session_id, strategy="semantic"):
    session_dir = get_session_dir(session_id)
    corpus_file = session_dir / "corpus.jsonl"
    
    ext = file_path.suffix.lower()
    chunks = []
    if ext == ".pdf":
        chunks = parse_pdf(file_path, strategy)
    elif ext in [".html", ".htm"]:
        chunks = parse_html(file_path, strategy)
    elif ext in [".txt", ".md"]:
        chunks = parse_text_markdown(file_path, strategy)
    else:
        print(f"Unsupported format: {ext}")
        return []
        
    with open(corpus_file, "a", encoding="utf-8") as f:
        for item in chunks:
            f.write(json.dumps(item) + "\n")
            
    return chunks

def process_all(session_id="default", strategy="semantic"):
    session_dir = get_session_dir(session_id)
    raw_dir = session_dir / "raw"
    corpus_file = session_dir / "corpus.jsonl"
    
    # clear existing corpus
    if corpus_file.exists():
        corpus_file.unlink()
        
    all_chunks = []
    if not any(raw_dir.iterdir()):
        print(f"No files found in {raw_dir}.")
        return []
        
    for file_path in raw_dir.glob("*"):
        if file_path.is_file():
            print(f"Processing {file_path.name} with {strategy} chunking...")
            chunks = process_file(file_path, session_id, strategy)
            all_chunks.extend(chunks)
            
    print(f"Successfully processed {len(all_chunks)} total chunks.")
    return all_chunks
