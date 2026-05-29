"""
Core RAG pipeline: query → agent decision → retrieve → generate with Claude.
Also runnable as CLI: python rag.py "your question here"
"""
from __future__ import annotations
import os
import sys
import json
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import anthropic

load_dotenv()

INDEX_NAME   = "neurorag"
EMBED_MODEL  = "all-MiniLM-L6-v2"
CLAUDE_MODEL = "claude-haiku-4-5"
TOP_K        = 8

SYSTEM = """\
You are NeuroRAG, a scientific assistant specializing in neuroscience and cognitive science.
Answer questions grounded strictly in the provided literature excerpts.
Cite sources using their bracket numbers [1], [2], etc. throughout your answer.
If the provided excerpts do not contain enough information, say so clearly rather than guessing."""

AGENT_SYSTEM = """\
You are a search query optimizer for a neuroscience literature database.
Given a question and the top retrieved excerpts, decide if the context is sufficient.

Reply with ONLY a JSON object — no explanation, no markdown:
{"action": "answer"}
  — if the excerpts adequately cover the question topic
{"action": "reformulate", "query": "<improved search query>"}
  — if the excerpts clearly miss the core topic and a different query would help

Reformulate only when truly needed."""

# Lazy singletons
_embed_model: SentenceTransformer | None = None
_index = None
_client: anthropic.Anthropic | None = None


def _embed() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL)
    return _embed_model


def _pinecone():
    global _index
    if _index is None:
        api_key = os.environ.get("PINECONE_API_KEY")
        if not api_key:
            raise RuntimeError("PINECONE_API_KEY not set in .env")
        pc = Pinecone(api_key=api_key)
        _index = pc.Index(INDEX_NAME)
    return _index


def _claude() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


def retrieve(query: str, k: int = TOP_K) -> list[dict]:
    vec = _embed().encode([query]).tolist()[0]
    res = _pinecone().query(vector=vec, top_k=k, include_metadata=True)
    return [
        {
            "text": match["metadata"].get("text", ""),
            "meta": {
                "authors":  match["metadata"].get("authors", ""),
                "year":     match["metadata"].get("year", ""),
                "title":    match["metadata"].get("title", ""),
                "filename": match["metadata"].get("filename", ""),
                "chunk":    match["metadata"].get("chunk", 0),
            },
            "dist": match["score"],
        }
        for match in res["matches"]
    ]


def _build_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        m = c["meta"]
        header = f"[{i}] {m['authors']} ({m['year']}) — {m['title']}"
        parts.append(f"{header}\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def _agent_decide(question: str, context_preview: str) -> dict:
    """One Claude call: decide whether to answer or reformulate the search query."""
    resp = _claude().messages.create(
        model=CLAUDE_MODEL,
        max_tokens=120,
        system=AGENT_SYSTEM,
        messages=[{
            "role": "user",
            "content": f"Question: {question}\n\nTop retrieved excerpts:\n{context_preview}",
        }],
    )
    raw = next((b.text for b in resp.content if b.type == "text"), "")
    try:
        return json.loads(raw.strip())
    except Exception:
        return {"action": "answer"}


def ask(query: str, k: int = TOP_K) -> dict:
    agent_steps = []

    # Step 1: Initial retrieval
    chunks = retrieve(query, k)
    context = _build_context(chunks)

    # Step 2: Agent decision — answer or reformulate?
    decision = _agent_decide(query, context[:2000])

    if decision.get("action") == "reformulate" and decision.get("query"):
        reformulated = decision["query"]
        agent_steps.append({"original": query, "reformulated": reformulated})

        new_chunks = retrieve(reformulated, k)
        seen = {f"{c['meta']['filename']}::{c['meta']['chunk']}" for c in chunks}
        for c in new_chunks:
            cid = f"{c['meta']['filename']}::{c['meta']['chunk']}"
            if cid not in seen:
                chunks.append(c)
                seen.add(cid)
        chunks = chunks[:k]
        context = _build_context(chunks)

    # Step 3: Generate grounded answer
    response = _claude().messages.create(
        model=CLAUDE_MODEL,
        max_tokens=4096,
        system=SYSTEM,
        messages=[{
            "role": "user",
            "content": f"Literature excerpts:\n\n{context}\n\n---\n\nQuestion: {query}",
        }],
    )
    answer = next((b.text for b in response.content if b.type == "text"), "")

    return {
        "query":   query,
        "answer":  answer,
        "chunks":  chunks,
        "citations": [
            {
                "idx":      i + 1,
                "authors":  c["meta"]["authors"],
                "year":     c["meta"]["year"],
                "title":    c["meta"]["title"],
                "filename": c["meta"]["filename"],
            }
            for i, c in enumerate(chunks)
        ],
        "usage": {
            "input_tokens":          response.usage.input_tokens,
            "output_tokens":         response.usage.output_tokens,
            "cache_creation_tokens": response.usage.cache_creation_input_tokens,
            "cache_read_tokens":     response.usage.cache_read_input_tokens,
        },
        "agent_steps": agent_steps,
    }


def evaluate_rag(query: str, answer: str, chunks: list[dict]) -> dict:
    """
    RAGAS evaluation: faithfulness + context precision.
    Lazy-imported so the app loads without ragas if evaluation isn't triggered.
    """
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.metrics import Faithfulness, LLMContextPrecisionWithoutReference
    from ragas.llms import LangchainLLMWrapper
    from langchain_anthropic import ChatAnthropic

    llm = LangchainLLMWrapper(
        ChatAnthropic(
            model=CLAUDE_MODEL,
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        )
    )

    sample = SingleTurnSample(
        user_input=query,
        response=answer,
        retrieved_contexts=[c["text"] for c in chunks],
    )

    result = evaluate(
        dataset=EvaluationDataset(samples=[sample]),
        metrics=[
            Faithfulness(llm=llm),
            LLMContextPrecisionWithoutReference(llm=llm),
        ],
    )

    df = result.to_pandas()
    row = df.iloc[0].to_dict()
    return {
        "faithfulness":       round(float(row.get("faithfulness", 0)), 3),
        "context_precision":  round(float(row.get("llm_context_precision_without_reference", 0)), 3),
    }


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or (
        "What does the literature say about prior probability effects on perceptual decisions?"
    )
    print(f"Query: {query}\n")
    result = ask(query)
    if result["agent_steps"]:
        print(f"[agent] reformulated → \"{result['agent_steps'][0]['reformulated']}\"")
    print(result["answer"])
    print("\n--- Sources ---")
    for c in result["citations"]:
        print(f"[{c['idx']}] {c['authors']} ({c['year']}) — {c['title']}")
    u = result["usage"]
    print(f"\nTokens — input: {u['input_tokens']}, output: {u['output_tokens']}")
