import os
import json
from dotenv import load_dotenv
from openai import AzureOpenAI
from prompts.query_refinement import QUERY_REFINEMENT_PROMPT

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

def refine_query(query: str):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "refine_query",
                "description": "Output the classified and rewritten queries.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query_type": {
                            "type": "string",
                            "enum": ["lookup", "conceptual", "compound"]
                        },
                        "rewritten_queries": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "1 query for lookup/conceptual, 2-4 for compound."
                        }
                    },
                    "required": ["query_type", "rewritten_queries"]
                }
            }
        }
    ]

    response = client.chat.completions.create(
        model=deployment_name,
        messages=[
            {"role": "system", "content": QUERY_REFINEMENT_PROMPT},
            {"role": "user", "content": query}
        ],
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "refine_query"}}
    )
    
    tool_calls = response.choices[0].message.tool_calls
    if tool_calls:
        for tool_call in tool_calls:
            if tool_call.function.name == "refine_query":
                return json.loads(tool_call.function.arguments)
            
    return {"query_type": "lookup", "rewritten_queries": [query]}
