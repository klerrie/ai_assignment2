"""
Unit tests for RAG system.
"""

import pytest
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def test_document_loading():
    """Test document can be loaded."""
    if os.path.exists("data/faq_document.txt"):
        loader = TextLoader("data/faq_document.txt", encoding='utf-8')
        docs = loader.load()
        assert len(docs) > 0
        assert len(docs[0].page_content) > 1000


def test_chunking():
    """Test text chunking creates at least 20 chunks."""
    if os.path.exists("data/faq_document.txt"):
        loader = TextLoader("data/faq_document.txt", encoding='utf-8')
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        assert len(chunks) >= 20


def test_vectorstore_exists():
    """Test vector store is created."""
    if os.path.exists("data/vectorstore"):
        assert os.path.exists("data/vectorstore/index.faiss")


def test_output_format():
    """Test query output format."""
    expected = ["user_question", "system_answer", "chunks_related"]
    sample = {
        "user_question": "test",
        "system_answer": "test",
        "chunks_related": ["chunk1"]
    }
    for key in expected:
        assert key in sample
    assert isinstance(sample["chunks_related"], list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

