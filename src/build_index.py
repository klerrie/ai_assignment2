"""
Data Pipeline: Build vector index from FAQ document using LangChain.
"""

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()


def build_index():
    """Build and save FAISS vector store from FAQ document."""
    # Configuration
    doc_path = "data/faq_document.txt"
    output_dir = "data/vectorstore"
    chunk_size = 500
    chunk_overlap = 50
    
    print("Building index...")
    
    # Load document
    loader = TextLoader(doc_path, encoding='utf-8')
    documents = loader.load()
    print(f"✓ Loaded document ({len(documents[0].page_content)} chars)")
    
    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "]
    )
    chunks = splitter.split_documents(documents)
    print(f"✓ Created {len(chunks)} chunks")
    
    if len(chunks) < 20:
        print(f"⚠ Warning: Only {len(chunks)} chunks (minimum: 20)")
    
    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    )
    
    vectorstore = FAISS.from_documents(chunks, embeddings)
    os.makedirs(output_dir, exist_ok=True)
    vectorstore.save_local(output_dir)
    
    print(f"✓ Saved vector store to {output_dir}")
    print("✓ Done!")


if __name__ == "__main__":
    build_index()

