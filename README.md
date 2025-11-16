# RAG-Based FAQ Support Chatbot

A simple RAG system built with LangChain to answer questions from company documentation.

## Project Structure

```
ai_homework2/
├── data/
│   ├── faq_document.txt      # Source FAQ document (1000+ words)
│   └── vectorstore/          # Generated FAISS vector store
├── src/
│   ├── build_index.py        # Build vector index
│   ├── query.py              # Query the RAG system
│   ├── evaluator.py          # Evaluate answer quality (bonus)
│   └── run_sample_queries.py # Generate sample outputs
├── outputs/
│   └── sample_queries.json   # Sample query-response pairs
├── tests/
│   └── test_core.py          # Unit tests
├── .env                      # Environment variables
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure `.env` file:**
   - Add your OpenRouter API key to `.env`
   - Update `OPENAI_API_KEY=your-actual-key-here`

## Usage

### 1. Build Index
```bash
python src/build_index.py
```

### 2. Query System
```bash
python src/query.py "How do employees request time off?"
```

### 3. Generate Sample Outputs
```bash
python src/run_sample_queries.py
```

### 4. Evaluate Answers (Optional)
```bash
python src/evaluator.py outputs/sample_queries.json
```

## Technical Choices

- **LangChain**: Industry-standard RAG framework
- **FAISS**: Fast vector similarity search
- **RecursiveCharacterTextSplitter**: Intelligent text chunking
- **OpenAI Embeddings**: High-quality semantic embeddings
- **RetrievalQA Chain**: End-to-end RAG pipeline

## Testing

```bash
pytest tests/test_core.py -v
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenRouter API key | Required |
| `OPENAI_BASE_URL` | API endpoint | `https://openrouter.ai/api/v1` |
| `EMBEDDING_MODEL` | Embedding model | `text-embedding-3-small` |
| `LLM_MODEL` | LLM model | `gpt-4o-mini` |
| `TOP_K` | Chunks to retrieve | `3` |

