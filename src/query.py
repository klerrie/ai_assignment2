"""
Query Pipeline: Answer questions using RAG with LangChain.
"""

import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


def query_rag(question: str, top_k: int = 3) -> dict:
    """
    Query RAG system and return answer with relevant chunks.
    
    Args:
        question: User's question
        top_k: Number of chunks to retrieve
        
    Returns:
        Dict with user_question, system_answer, chunks_related
    """
    # Load vector store
    embeddings = OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    )
    vectorstore = FAISS.load_local(
        "data/vectorstore", 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    
    # Setup retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    
    # Setup LLM
    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"),
        temperature=0.7,
        max_tokens=500
    )
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful HR support assistant. Answer the question using only the provided context."),
        ("user", "Context: {context}\n\nQuestion: {question}\n\nAnswer:")
    ])
    
    # Create RAG chain using LCEL
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Get answer and source documents
    answer = rag_chain.invoke(question)
    source_docs = retriever.invoke(question)
    
    return {
        "user_question": question,
        "system_answer": answer,
        "chunks_related": [doc.page_content for doc in source_docs]
    }


def main():
    """Command-line interface."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python query.py '<question>'")
        sys.exit(1)
    
    question = sys.argv[1]
    result = query_rag(question)
    
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
