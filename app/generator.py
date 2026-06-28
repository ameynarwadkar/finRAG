import os
import json
from dotenv import load_dotenv
from openai import AzureOpenAI
from langfuse import observe
from prompts.generation import GENERATION_PROMPT
from app.cache import generator_cache

load_dotenv()

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

@observe(as_type="generation")
def generate_answer(query: str, contexts: list):
    context_str = ""
    for idx, ctx in enumerate(contexts):
        context_str += f"--- Document {idx+1} ---\n"
        context_str += f"Doc ID: {ctx.get('source_file')}\n"
        context_str += f"Chunk: {ctx.get('chunk_id')} - {ctx.get('section_heading')}\n"
        context_str += f"Text: {ctx.get('text')}\n\n"

    full_prompt = f"Context:\n{context_str}\n\nQuestion: {query}"
    cached = generator_cache.get_exact(full_prompt)
    if cached:
        return cached

    tools = [
        {
            "type": "function",
            "function": {
                "name": "generate_answer",
                "description": "Output the final answer with citations.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string"},
                        "citations": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "document_number": {"type": "integer"},
                                    "claim": {"type": "string"}
                                },
                                "required": ["document_number", "claim"]
                            }
                        },
                        "confidence": {
                            "type": "string",
                            "enum": ["high", "medium", "low"]
                        }
                    },
                    "required": ["answer", "citations", "confidence"]
                }
            }
        }
    ]

    response = client.chat.completions.create(
        model=deployment_name,
        messages=[
            {"role": "system", "content": GENERATION_PROMPT},
            {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {query}"}
        ],
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "generate_answer"}}
    )
    
    tool_calls = response.choices[0].message.tool_calls
    if tool_calls:
        for tool_call in tool_calls:
            if tool_call.function.name == "generate_answer":
                generation = json.loads(tool_call.function.arguments)
                from app.verifier import verify_citations, score_generation
                generation = verify_citations(generation, contexts)
                final_result = score_generation(query, generation, contexts)
                generator_cache.set_exact(full_prompt, final_result)
                return final_result
            
    fallback = {"answer": "Failed to generate.", "citations": [], "confidence": "low"}
    generator_cache.set_exact(full_prompt, fallback)
    return fallback
