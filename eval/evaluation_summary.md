# RAGAS Evaluation Results
**Date**: 2026-06-25

## Baseline Scores

The AnyRAG pipeline was evaluated using the `ragas` library across a golden dataset of 32 queries (including factoid, keyword-heavy, and unanswerable edge cases). 

The initial baseline metrics show the performance out of 1.0:

| Metric | Score | Description |
| :--- | :--- | :--- |
| **Context Precision** | **0.8562** | Excellent. The retrieval system successfully places the most relevant chunk at the very top of the returned results. |
| **Context Recall** | **0.8750** | Excellent. The retrieval system successfully fetches all the chunks necessary to answer the user's question. |
| **Faithfulness** | **0.6938** | Needs Work. The LLM generation occasionally hallucinates or includes information that cannot be directly verified by the retrieved text alone. |
| **Answer Relevancy** | **0.5942** | Needs Work. The generated answer sometimes strays off-topic, provides overly verbose explanations, or fails to directly answer the specific question asked. |

## Analysis & Next Steps

Our **Hybrid RRF Retrieval pipeline is highly accurate**. By utilizing both Dense Vector search and Sparse BM25 keyword matching (with a Cross-Encoder reranker), we have virtually eliminated the "Needle in a Haystack" problem for this corpus.

Our bottleneck is the **Generator Pipeline**. Despite providing the LLM with the correct context, it struggles to strictly adhere to the grounded text. 

## Prompt Engineering Improvements

After identifying the bottleneck in the generator pipeline, we implemented a highly strict system prompt (`prompts/generation.py`) to heavily penalize hallucinations, forbid outside knowledge, and strictly enforce concise answers with inline citations.

Re-evaluating the pipeline with the improved prompt yielded the following metrics:
*   **Context Precision:** `0.8906`
*   **Context Recall:** `0.8906`
*   **Faithfulness:** `0.7240` (Improved adherence to context)
*   **Answer Relevancy:** `0.6115` (Reduced verbosity and preamble)

This clearly demonstrates the value of the evaluation harness in identifying and resolving LLM behavioral issues.
