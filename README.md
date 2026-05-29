# NeuroRAG

**Proof of concept** — an end-to-end RAG pipeline over a real neuroscience PhD library. Ask questions in natural language, get cited answers grounded in the literature.

🔗 **[Live demo](https://myneurorag.streamlit.app)**

---

## What it does

You type a question — *"What does the literature say about dopamine and reward prediction errors?"* — and NeuroRAG:

1. Embeds your query into a vector using `all-MiniLM-L6-v2`
2. An **agent** evaluates whether the retrieved context is sufficient, or reformulates the search query for a better retrieval
3. Retrieves the most semantically relevant excerpts from 951 neuroscience papers (46,000+ chunks) stored in Pinecone
4. Passes them to Claude as grounded context
5. Returns a synthesised answer with inline citations
6. Optionally runs **RAGAS** evaluation to score faithfulness and context precision

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
Agent (Claude)  ──►  sufficient context? → answer
                 └──► insufficient?      → reformulate query → re-retrieve → merge chunks
    │
    ▼
Prompt builder  ──►  system prompt + retrieved context + query
    │
    ▼
Claude API  ──►  grounded answer with [1][2]... citations
    │
    ▼
RAGAS evaluator  ──►  faithfulness score · context precision score
    │
    ▼
Streamlit UI
```

---

## Agentic loop

After the initial retrieval, a lightweight Claude call inspects the top chunks and decides:

- **Answer** — context is relevant, proceed to generation
- **Reformulate** — the chunks miss the core topic; generate a better search query, re-retrieve, merge both result sets, then generate

This mirrors a pattern used in production RAG systems to handle vocabulary mismatch (the user asks about "fear learning" but the papers use "fear conditioning") and to improve answer quality on multi-facet questions. The reformulation step is shown in the UI when it fires.

---

## Evaluation with RAGAS

Every answer can be evaluated on demand using [RAGAS](https://docs.ragas.io), an open-source framework for RAG quality measurement:

| Metric | What it measures |
|---|---|
| **Faithfulness** | Are the claims in the answer actually supported by the retrieved excerpts? Detects hallucination. |
| **Context precision** | Are the retrieved excerpts relevant to the question? Measures retrieval quality. |

Both are implemented as **LLM-as-judge** prompts sent to Claude — the same methodology RAGAS uses internally — keeping the evaluation stack self-contained with no additional dependencies.

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
| Vector store | Pinecone (serverless) |
| LLM | Anthropic Claude API (`claude-haiku-4-5`) |
| Agent | Claude-based query reformulation loop |
| Evaluation | RAGAS · `langchain-anthropic` |
| PDF parsing | PyMuPDF |
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

RAG is a core pattern in enterprise healthtech: it lets LLMs reason over proprietary document corpora (clinical guidelines, trial reports, patient records) without retraining or exposing sensitive data to the model. This project demonstrates the full pipeline — ingestion, retrieval, agentic query refinement, grounded generation, and automated evaluation — applied to a real scientific corpus.

---

## Fine-tuning: when RAG isn't enough

RAG and fine-tuning solve different problems and are often complementary.

**RAG** is the right choice when:
- The knowledge base changes frequently (new papers, updated guidelines)
- You need source citations and verifiability
- You want to keep the base model general-purpose
- Data privacy prevents sending training data to a provider

**Fine-tuning** is the right choice when:
- You need the model to internalise a specific output format or clinical terminology (e.g. structured discharge summaries)
- Latency is critical and a shorter prompt is needed
- You have high-quality labelled examples of the desired behaviour
- You want a smaller, cheaper model to match a larger one on a narrow task

**For NeuroRAG specifically**, a fine-tuned model would add value as a second layer: a smaller base model (e.g. Llama 3 8B or GPT-4o mini) fine-tuned on neuroscience Q&A pairs generated from the corpus, then used as the generator instead of calling the full Claude API. The training pipeline would be:

1. Generate synthetic Q&A pairs from paper chunks using Claude (`chunk → question + ideal answer`)
2. Format as JSONL instruction pairs
3. Fine-tune via OpenAI's fine-tuning API or with `transformers` + `peft` (LoRA) on a local/cloud GPU
4. Swap the generator in `rag.py` for the fine-tuned model endpoint
5. Compare RAGAS scores against the base model baseline

This would reduce per-query cost and latency while specialising the model's vocabulary — the main practical benefit in a clinical deployment where the same narrow task runs millions of times.

---

## Author

Ettore Cerracchio — PhD in computational neuroscience, building towards applied AI engineering in healthtech.  
[LinkedIn](https://linkedin.com/in/ettorecerracchio) · [GitHub](https://github.com/jekke97)
