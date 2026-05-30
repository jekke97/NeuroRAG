# NeuroRAG

**Proof of concept** — an end-to-end agentic RAG pipeline over two real scientific corpora: a personal neuroscience PhD library and a corpus of Philips-affiliated MRI/fMRI research papers scraped from PubMed.

🔗 **[Live demo](https://myneurorag.streamlit.app)**

---

## What it does

You select a corpus, type a question, and NeuroRAG:

1. Embeds your query into a vector using `all-MiniLM-L6-v2`
2. Checks whether the question is within the corpus scope (out-of-scope questions are answered from general knowledge and flagged)
3. Retrieves the most semantically relevant excerpts from the selected Pinecone namespace
4. An **agent** evaluates whether the retrieved context is sufficient, or reformulates the search query for better retrieval
5. Passes the excerpts to Claude as grounded context and returns a cited answer
6. Automatically runs **LLM-as-judge** evaluation to score faithfulness and context precision

No hallucination of sources — every claim is traceable to a real paper in the corpus.

---

## Architecture

```
User query
    │
    ▼
Scope check (Claude)  ──►  out of scope? → answer from general knowledge (flagged)
    │ in scope
    ▼
Sentence Transformer (all-MiniLM-L6-v2)
    │  384-dim vector
    ▼
Pinecone  ──►  namespace: "philips" | "neurorag" | merged (both)
    │           Top-K relevant chunks + metadata
    ▼
Agent (Claude)  ──►  sufficient context? → answer
                 └──► insufficient?      → reformulate → re-retrieve → merge
    │
    ▼
Claude API  ──►  grounded answer with [1][2]... citations
    │
    ▼
LLM-as-judge  ──►  faithfulness score · context precision score
    │
    ▼
Streamlit UI
```

---

## Agentic loop

Three agent steps run on every query, all visible in the live UI:

1. **Scope check** — a Claude call decides if the question is within the corpus. Off-topic questions are answered from general knowledge and shown in a red warning box.
2. **Query reformulation** — after initial retrieval, a second Claude call inspects the top chunks. If they miss the core topic, it generates a better search query, re-retrieves, and merges both result sets.
3. **LLM-as-judge evaluation** — after generation, two Claude calls score the answer on faithfulness and context precision (RAGAS methodology).

---

## Corpora

| Corpus | Source | Papers | Chunks |
|---|---|---|---|
| **Philips research** | PubMed (Philips[Affiliation] + MRI/fMRI terms) · PMC full text | 2,093 | ~120,000 |
| **My library** | Personal Zotero PhD library | 951 | ~46,000 |

The Philips corpus is built by:
1. `fetch_philips_papers.py` — scrapes PubMed metadata → `philips_papers.csv`
2. `ingest_philips.py` — fetches full text from PMC (JATS XML) for open-access papers, falls back to abstract, embeds and upserts to Pinecone namespace `philips`

Both corpora live in separate Pinecone namespaces within the same index. The **Both** option merges results from both namespaces, scored by cosine similarity, with source badges on each citation.

---

## Evaluation

Every answer is automatically evaluated using the RAGAS methodology:

| Metric | What it measures |
|---|---|
| **Faithfulness** | Are the claims in the answer supported by the retrieved excerpts? Detects hallucination. |
| **Context precision** | Are the retrieved excerpts relevant to the question? Measures retrieval quality. |

Both implemented as LLM-as-judge prompts sent to Claude — self-contained, no additional dependencies.

---

## Stack

| Layer | Tool |
|---|---|
| Embeddings | `sentence-transformers` · `all-MiniLM-L6-v2` |
| Vector store | Pinecone (serverless) · two namespaces |
| LLM | Anthropic Claude API (`claude-haiku-4-5`) |
| Agent | Scope check + query reformulation + LLM-as-judge |
| PDF parsing | PyMuPDF |
| PubMed scraping | NCBI E-utilities (esearch + efetch) |
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

### Ingesting the corpora (one-time)

```bash
# Neuroscience library (requires PDFs in Google Drive path set in ingest.py)
python ingest.py

# Philips MRI/fMRI corpus
python fetch_philips_papers.py   # scrapes PubMed → philips_papers.csv
python ingest_philips.py         # fetches full text, embeds, upserts to Pinecone
```

---

## Relevance to clinical AI

RAG is a core pattern in enterprise healthtech: it lets LLMs reason over proprietary document corpora (clinical guidelines, trial reports, patient records) without retraining or exposing sensitive data to the model. This project demonstrates the full pipeline — ingestion, retrieval, agentic query refinement, grounded generation, and automated evaluation — applied to two real scientific corpora, one of which is Philips' own published MRI/fMRI research.

---

## Fine-tuning: when RAG isn't enough

RAG and fine-tuning solve different problems and are often complementary.

**RAG** is the right choice when:
- The knowledge base changes frequently (new papers, updated guidelines)
- You need source citations and verifiability
- You want to keep the base model general-purpose
- Data privacy prevents sending training data to a provider

**Fine-tuning** is the right choice when:
- You need the model to internalise a specific output format or clinical terminology
- Latency is critical and a shorter prompt is needed
- You have high-quality labelled examples of the desired behaviour
- You want a smaller, cheaper model to match a larger one on a narrow task

**For NeuroRAG specifically**, a fine-tuned model would add value as a second layer: a smaller base model (e.g. Llama 3 8B) fine-tuned on Q&A pairs generated from the corpus, then used as the generator instead of calling the full Claude API. The training pipeline would be:

1. Generate synthetic Q&A pairs from paper chunks using Claude (`chunk → question + ideal answer`)
2. Format as JSONL instruction pairs
3. Fine-tune via OpenAI's fine-tuning API or with `transformers` + `peft` (LoRA) on a local/cloud GPU
4. Swap the generator in `rag.py` for the fine-tuned model endpoint
5. Compare RAGAS scores against the base model baseline

---

## Author

Ettore Cerracchio — PhD in computational neuroscience, building towards applied AI engineering in healthtech.  
[LinkedIn](https://linkedin.com/in/ettorecerracchio) · [GitHub](https://github.com/jekke97)
