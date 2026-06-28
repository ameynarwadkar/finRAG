import os
import json
import random
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

# Setup Azure OpenAI Client
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
base_endpoint = endpoint.split("/openai")[0] if "/openai" in endpoint else endpoint
api_version = "2024-02-01"
if "api-version=" in endpoint:
    api_version = endpoint.split("api-version=")[1].split("&")[0]

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=api_version,
    azure_endpoint=base_endpoint
)

deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")

GENERATION_PROMPT = """
You are an expert dataset creator for a Retrieval-Augmented Generation (RAG) system.
Given a document chunk, generate TWO distinct question-and-answer pairs based STRICTLY on the chunk.

Types of questions to generate:
1. "simple_factoid": A direct question that tests basic semantic retrieval.
2. "keyword_heavy": A question that uses specific jargon, acronyms, or numbers from the text to test sparse (BM25) retrieval.

Reply ONLY with a valid JSON object containing a "qa_pairs" array.
Schema:
{
  "qa_pairs": [
    {
      "question": "The generated question",
      "ground_truth": "The accurate, concise answer",
      "question_type": "simple_factoid"
    },
    {
      "question": "The generated question",
      "ground_truth": "The accurate, concise answer",
      "question_type": "keyword_heavy"
    }
  ]
}
"""

def generate_dataset(session_id="default", samples=15):
    corpus_path = f"sessions/{session_id}/corpus.jsonl"
    eval_dir = "eval"
    os.makedirs(eval_dir, exist_ok=True)
    out_path = os.path.join(eval_dir, "golden_dataset.json")
    
    if not os.path.exists(corpus_path):
        print(f"Error: {corpus_path} not found. Please ingest documents first.")
        return

    # Load corpus
    articles = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            articles.append(json.loads(line.strip()))
            
    if not articles:
        print("Corpus is empty.")
        return
        
    print(f"Loaded {len(articles)} chunks. Selecting {min(samples, len(articles))} for generation...")
    sampled_chunks = random.sample(articles, min(samples, len(articles)))
    
    dataset = []
    
    for idx, chunk in enumerate(sampled_chunks):
        print(f"Processing chunk {idx+1}/{len(sampled_chunks)}...")
        
        chunk_text = f"Title: {chunk.get('section_heading', 'Unknown')}\n\nContent:\n{chunk.get('text', '')}"
        
        try:
            response = client.chat.completions.create(
                model=deployment_name,
                messages=[
                    {"role": "system", "content": GENERATION_PROMPT},
                    {"role": "user", "content": chunk_text}
                ],
                response_format={ "type": "json_object" }, # Actually, we requested an array, but JSON object mode expects {}
                temperature=0.3
            )
            content = response.choices[0].message.content
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict):
                    qa_pairs_list = parsed.get("qa_pairs", [])
                else:
                    qa_pairs_list = parsed
                    
                if isinstance(qa_pairs_list, list):
                    for pair in qa_pairs_list:
                        pair["reference_contexts"] = [f"{chunk['source_file']}_{chunk['chunk_id']}"]
                        dataset.append(pair)
            except Exception as parse_e:
                print(f"Parse error: {parse_e}")
            
        except Exception as e:
            print(f"Failed to generate for chunk {idx}: {e}")
            
    print(f"Generated {len(dataset)} QA pairs.")
    
    # Add a couple of manual adversarial examples
    dataset.append({
        "question": "What is the penalty for exceeding the global data limit across all regions?",
        "ground_truth": "I don't know. The provided documents do not mention a global data limit penalty.",
        "question_type": "unanswerable",
        "reference_contexts": []
    })
    
    dataset.append({
        "question": "Can you explain how the specific QDR-542 module interacts with the core mainframe?",
        "ground_truth": "I don't know. The documents do not contain information about a QDR-542 module or core mainframe interaction.",
        "question_type": "unanswerable",
        "reference_contexts": []
    })
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4)
        
    print(f"Golden dataset saved to {out_path} with {len(dataset)} items.")

if __name__ == "__main__":
    import sys
    session = sys.argv[1] if len(sys.argv) > 1 else "session_dpu0osg"
    generate_dataset(session)
