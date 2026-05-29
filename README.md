# NeuroRAG

**Proof of concept** — an end-to-end RAG pipeline over a real neuroscience PhD library. Ask questions in natural language, get cited answers grounded in the literature.

🔗 **[Live demo](https://jekke97-neurorag-app-xyz.streamlit.app)** ← replace with your actual URL

---

## What it does

You type a question — *"What does the literature say about dopamine and reward prediction errors?"* — and NeuroRAG:

1. Embeds your query into a vector using `all-MiniLM-L6-v2`
2. Retrieves the most semantically relevant excerpts from 951 neuroscience papers (46,000+ chunks) stored in Pinecone
3. Passes them to Claude as grounded context
4. Returns a synthesised answer with inline citations

No hallucination of sources — every claim is traceable to a real paper in the corpus.

---

## Architecture

```
User query
    │
    ▼
Sentence Transformer (all-MiniLM-L6-v2)
    │  384-dim vector
    ▼
Pinecone vector index  ──►  Top-K relevant chunks + metadata
    │
    ▼
Prompt builder  ──►  system prompt + retrieved context + query
    │
    ▼
Claude API  ──►  grounded answer with [1][2]... citations
    │
    ▼
Streamlit UI
```

---

## Corpus

- **951 PDFs** from a personal Zotero library accumulated during a PhD in computational neuroscience
- Topics: attention, expectation, reward, decision-making, fMRI, EEG, predictive coding, Bayesian inference, dopamine, working memory, and more
- Chunked at 300 words with 30-word overlap → ~46,000 indexed excerpts

---

## Stack

| Layer | Tool |
|---|---|
| Embeddings | `sentence-transformers` · `all-MiniLM-L6-v2` |
| Vector store | Pinecone (serverless, free tier) |
| LLM | Anthropic Claude API |
| PDF parsing | PyMuPDF (`fitz`) |
| UI | Streamlit |
| Hosting | Streamlit Community Cloud |

---

## Running locally

```bash
# 1. Clone and install
git clone https://github.com/jekke97/NeuroRAG.git
cd NeuroRAG
pip install -r requirements.txt

# 2. Set API keys
cp .env.example .env
# fill in ANTHROPIC_API_KEY and PINECONE_API_KEY in .env

# 3. Run the app
streamlit run app.py
```

> **Ingestion** (`python ingest.py`) is a one-time step that reads PDFs from a local path and uploads embeddings to Pinecone. The live demo uses a pre-populated index — you don't need to re-run it.

---

## Relevance to clinical AI

RAG is a core pattern in enterprise healthtech: it lets LLMs reason over proprietary document corpora (clinical guidelines, trial reports, patient records) without retraining. This project demonstrates the full pipeline — ingestion, retrieval, grounded generation, and a deployable interface — applied to a real scientific corpus.

---

## Author

Ettore Cerracchio — PhD in computational neuroscience, building towards applied AI engineering in healthtech.  
[LinkedIn](https://linkedin.com/in/ettorecerracchio) · [GitHub](https://github.com/jekke97)
