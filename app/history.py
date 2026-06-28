import os
from openai import AzureOpenAI
from langfuse import observe

@observe(as_type="generation")
def rewrite_query_with_history(current_query: str, history: list):
    """
    history format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    """
    if not history:
        return current_query

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

    # Take last 2 turns (4 messages) to keep context small
    history_str = ""
    for msg in history[-4:]:
        role = "User" if msg["role"] == "user" else "AI"
        history_str += f"{role}: {msg['content']}\n"

    system_prompt = """You are a query rewriting assistant for a financial RAG system.
Given the conversation history and a follow-up query, rewrite the follow-up query to be a fully standalone, self-contained question that can be understood without the prior context.
If the follow-up query is already standalone, just return it unchanged.
DO NOT answer the question. ONLY output the rewritten query. Do not wrap in quotes."""

    prompt = f"History:\n{history_str}\n\nFollow-up Query: {current_query}\n\nStandalone Query:"

    response = client.chat.completions.create(
        model=deployment_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )

    return response.choices[0].message.content.strip()
