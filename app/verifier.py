import os
import json
from openai import AzureOpenAI

def get_client():
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    base_endpoint = endpoint.split("/openai")[0] if "/openai" in endpoint else endpoint
    api_version = "2024-02-01"
    if "api-version=" in endpoint:
        api_version = endpoint.split("api-version=")[1].split("&")[0]

    return AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=api_version,
        azure_endpoint=base_endpoint
    )

VERIFICATION_PROMPT = """
You are a strict citation verification judge.
Your task is to determine if a specific claim made by an AI assistant is fully supported by the provided source text.
You will be provided with:
1. The Claim
2. The Source Text

Reply with a JSON object:
{
    "supported": true/false,
    "reasoning": "brief explanation"
}
"""

def verify_citations(generation: dict, contexts: list):
    client = get_client()
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")
    
    citations = generation.get("citations", [])
    verified_citations = []
    
    for citation in citations:
        doc_idx = citation.get("document_number", 0) - 1
        claim = citation.get("claim", "")
        
        if doc_idx < 0 or doc_idx >= len(contexts):
            citation["supported"] = False
            citation["verification_reason"] = "Document number out of bounds."
            verified_citations.append(citation)
            continue
            
        source_text = contexts[doc_idx].get("text", "")
        
        user_msg = f"Claim: {claim}\n\nSource Text: {source_text}"
        
        try:
            response = client.chat.completions.create(
                model=deployment_name,
                messages=[
                    {"role": "system", "content": VERIFICATION_PROMPT},
                    {"role": "user", "content": user_msg}
                ],
                response_format={ "type": "json_object" },
                temperature=0.0
            )
            result = json.loads(response.choices[0].message.content)
            citation["supported"] = result.get("supported", False)
            citation["verification_reason"] = result.get("reasoning", "")
        except Exception as e:
            citation["supported"] = False
            citation["verification_reason"] = f"Verification failed: {str(e)}"
            
        verified_citations.append(citation)
        
    generation["citations"] = verified_citations
    return generation

COMPLETENESS_PROMPT = """
You are an evaluator scoring an AI assistant's answer.
Given a Query, the Retrieved Contexts, and the Answer, score:
1. "retrieval_confidence" (0.0 to 1.0): How relevant and sufficient are the retrieved contexts to answer the query?
2. "completeness" (0.0 to 1.0): How completely does the answer address the query?

Reply with a JSON object:
{
    "retrieval_confidence": 0.0,
    "completeness": 0.0
}
"""

def score_generation(query: str, generation: dict, contexts: list):
    client = get_client()
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")
    
    citations = generation.get("citations", [])
    total_citations = len(citations)
    supported_citations = sum(1 for c in citations if c.get("supported"))
    citation_coverage = supported_citations / total_citations if total_citations > 0 else 0.0
    
    context_str = ""
    for i, c in enumerate(contexts):
        context_str += f"[Doc {i+1}] {c.get('text', '')[:500]}...\n"
        
    user_msg = f"Query: {query}\n\nContexts: {context_str}\n\nAnswer: {generation.get('answer', '')}"
    
    try:
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "system", "content": COMPLETENESS_PROMPT},
                {"role": "user", "content": user_msg}
            ],
            response_format={ "type": "json_object" },
            temperature=0.0
        )
        scores = json.loads(response.choices[0].message.content)
        retrieval_confidence = scores.get("retrieval_confidence", 0.5)
        completeness = scores.get("completeness", 0.5)
    except Exception:
        retrieval_confidence = 0.5
        completeness = 0.5
        
    composite_score = (retrieval_confidence * 0.3) + (citation_coverage * 0.4) + (completeness * 0.3)
    
    generation["confidence_metrics"] = {
        "retrieval_confidence": round(retrieval_confidence, 2),
        "citation_coverage": round(citation_coverage, 2),
        "completeness": round(completeness, 2),
        "composite_score": round(composite_score, 2)
    }
    
    if composite_score >= 0.8:
        generation["confidence"] = "high"
    elif composite_score >= 0.5:
        generation["confidence"] = "medium"
    else:
        generation["confidence"] = "low"
        
    return generation

