# simple_rag

A small retrieval-augmented generation app: upload a PDF, ask questions, get answers grounded in the document.

## How it works
1. **Ingest:** extract text with pypdf, split into chunks, embed with OpenAI `text-embedding-3-small`
2. **Store:** upsert vectors to Pinecone (FAISS path included for local use)
3. **Retrieve:** embed the query, take top-5 chunks by similarity
4. **Answer:** `gpt-4o-mini` answers using only the retrieved context

## Run locally
```bash
pip install -r requirements.txt
# .env: OPENAI_API_KEY=..., PINECONE_API_KEY=...
streamlit run app.py
```

## Known limitations / next steps
- Sentence-boundary chunking is naive; next: token-based chunks with overlap
- Embeddings are created one chunk at a time; next: batch calls
- No eval set yet; next: a fixed question set with expected answers