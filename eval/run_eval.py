import os
import json
import sys
from types import ModuleType
# Mock the missing vertexai module for Ragas to bypass known import bug
dummy_module = ModuleType("langchain_community.chat_models.vertexai")
dummy_module.ChatVertexAI = None
sys.modules["langchain_community.chat_models.vertexai"] = dummy_module

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_openai import AzureChatOpenAI
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from dotenv import load_dotenv

# Add project root to sys path so we can import our app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app.retrieval as retrieval_module
from app.refiner import refine_query
from app.generator import generate_answer

load_dotenv()

# Force UTF-8 output to prevent Windows console crashes on math symbols
sys.stdout.reconfigure(encoding='utf-8')

def run_evaluation(session_id="session_dpu0osg"):
    print("Loading Golden Dataset...")
    dataset_path = "eval/golden_dataset.json"
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found.")
        return
        
    with open(dataset_path, "r", encoding="utf-8") as f:
        golden_data = json.load(f)
        
    print(f"Loading Vector Indices for session: {session_id}...")
    retrieval_module.load_indices(session_id)
    
    questions = []
    answers = []
    contexts = []
    ground_truths = []
    
    # 1. Run our pipeline to generate answers and contexts
    for i, item in enumerate(golden_data):
        q = item["question"]
        print(f"[{i+1}/{len(golden_data)}] Running pipeline for: {q}")
        
        refinement = refine_query(q)
        all_contexts = []
        seen_ids = set()
        
        for subq in refinement.get("rewritten_queries", [q]):
            results = retrieval_module.hybrid_reranked_search(subq, k=5)
            for res in results:
                source_file = f"{res['source_file']}_{res['chunk_id']}"
                if source_file not in seen_ids:
                    seen_ids.add(source_file)
                    all_contexts.append(res)
                    
        # Simulate our server.py Graceful "I don't know"
        best_score = max([ctx.get("cross_score", -10.0) for ctx in all_contexts]) if all_contexts else -10.0
        if best_score < 0.0:
            ans_text = "I don't have enough confident information in the retrieved documents to answer this question accurately without hallucinating."
        else:
            generation = generate_answer(q, all_contexts)
            ans_text = generation.get("answer", "")
            
        questions.append(q)
        answers.append(ans_text)
        # Ragas expects a list of strings for contexts
        contexts.append([c["text"] for c in all_contexts])
        ground_truths.append(item["ground_truth"])
        
    # 2. Build HuggingFace Dataset required by Ragas
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    hf_dataset = Dataset.from_dict(data)
    
    # 3. Configure Ragas LLM & Embeddings
    print("\nConfiguring Ragas LLM (Azure) and Embeddings (Local HuggingFace)...")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    base_endpoint = endpoint.split("/openai")[0] if "/openai" in endpoint else endpoint
    
    azure_llm = AzureChatOpenAI(
        openai_api_version="2024-02-01",
        azure_endpoint=base_endpoint,
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=0.0
    )
    
    # Using the same local embedding model we use for retrieval to save API costs
    local_embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    
    # 4. Run Evaluation
    print("\nStarting Ragas Evaluation... (This may take a few minutes)")
    result = evaluate(
        dataset=hf_dataset, 
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
        ],
        llm=azure_llm,
        embeddings=local_embeddings
    )
    
    print("\n==============================")
    print("--- RAGAS EVALUATION SCORE ---")
    print("==============================")
    print(result)
    
    # Save detailed breakdown
    df = result.to_pandas()
    out_path = "eval/evaluation_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nDetailed per-question results saved to {out_path}")

if __name__ == "__main__":
    import sys
    session = sys.argv[1] if len(sys.argv) > 1 else "session_dpu0osg"
    run_evaluation(session)
