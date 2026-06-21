import os
import json
from dotenv import load_dotenv
from openai import AzureOpenAI
from prompts.generation import GENERATION_PROMPT

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

def generate_answer(query: str, contexts: list):
    context_str = ""
    for idx, ctx in enumerate(contexts):
        context_str += f"--- Document {idx+1} ---\n"
        context_str += f"Doc ID: {ctx.get('doc_id')}\n"
        context_str += f"Article: {ctx.get('article_number')} - {ctx.get('article_title')}\n"
        context_str += f"Text: {ctx.get('text')}\n\n"

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
                        "cited_articles": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "doc_id": {"type": "string"},
                                    "article_number": {"type": "string"},
                                    "quote_or_paraphrase": {"type": "string"}
                                },
                                "required": ["doc_id", "article_number"]
                            }
                        },
                        "confidence": {
                            "type": "string",
                            "enum": ["high", "medium", "low"]
                        }
                    },
                    "required": ["answer", "cited_articles", "confidence"]
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
                return json.loads(tool_call.function.arguments)
            
    return {"answer": "Failed to generate.", "cited_articles": [], "confidence": "low"}
