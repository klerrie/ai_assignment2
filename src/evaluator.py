"""
Evaluator: Score answer quality (0-10) using LangChain.
"""

import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()


def evaluate_answer(question: str, answer: str, chunks: list) -> dict:
    """
    Evaluate answer quality (0-10).
    
    Args:
        question: User question
        answer: System answer
        chunks: Retrieved chunks
        
    Returns:
        Dict with score, chunk_relevance_score, answer_accuracy_score, 
        completeness_score, and reason
    """
    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"),
        temperature=0.3
    )
    
    chunks_text = "\n\n".join([f"Chunk {i+1}: {chunk}" for i, chunk in enumerate(chunks)])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert RAG evaluator. Return JSON only."),
        ("user", """Evaluate this answer (0-10):

Question: {question}
Chunks: {chunks}
Answer: {answer}

Score on:
1. Chunk Relevance (0-3): Are chunks relevant?
2. Answer Accuracy (0-4): Is answer correct?
3. Completeness (0-3): Does it fully answer?

Return JSON: {{"score": 0-10, "chunk_relevance_score": 0-3, "answer_accuracy_score": 0-4, "completeness_score": 0-3, "reason": "explanation"}}""")
    ])
    
    parser = JsonOutputParser()
    chain = prompt | llm | parser
    
    try:
        result = chain.invoke({
            "question": question,
            "chunks": chunks_text,
            "answer": answer
        })
        result["score"] = max(0, min(10, result.get("score", 0)))
        return result
    except Exception as e:
        return {
            "score": 0,
            "chunk_relevance_score": 0,
            "answer_accuracy_score": 0,
            "completeness_score": 0,
            "reason": f"Error: {str(e)}"
        }


def main():
    """Command-line interface."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python evaluator.py <query_result.json>")
        sys.exit(1)
    
    with open(sys.argv[1], 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    result = evaluate_answer(
        data["user_question"],
        data["system_answer"],
        data["chunks_related"]
    )
    
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

