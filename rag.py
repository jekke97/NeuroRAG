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


_FAITHFULNESS_SYSTEM = """\
You are an evaluation judge. Assess whether the claims in an answer are supported by the provided context.

Steps:
1. Extract every factual claim from the answer.
2. For each claim, check whether it is directly supported by the context.
3. Score = supported_claims / total_claims.

Reply with ONLY valid JSON: {"score": <float 0-1>, "supported": <int>, "total": <int>, "reason": "<one sentence>"}"""

_PRECISION_SYSTEM = """\
You are an evaluation judge. Assess whether the retrieved passages are relevant to the question.

Steps:
1. For each numbered passage, decide if it contains information useful for answering the question.
2. Score = relevant_passages / total_passages.

Reply with ONLY valid JSON: {"score": <float 0-1>, "relevant": <int>, "total": <int>, "reason": "<one sentence>"}"""


def evaluate_rag(query: str, answer: str, chunks: list[dict]) -> dict:
    """
    LLM-as-judge evaluation (RAGAS methodology) using Claude directly.
    Measures faithfulness and context precision without external dependencies.
    """
    context = _build_context(chunks)

    # Faithfulness: are the answer's claims grounded in the context?
    faith_resp = _claude().messages.create(
        model=CLAUDE_MODEL,
        max_tokens=200,
        system=_FAITHFULNESS_SYSTEM,
        messages=[{"role": "user", "content": f"Context:\n{context}\n\nAnswer:\n{answer}"}],
    )
    faith_raw = next((b.text for b in faith_resp.content if b.type == "text"), "")
    try:
        faith = json.loads(faith_raw.strip())
    except Exception:
        faith = {"score": 0.0, "reason": "parse error"}

    # Context precision: are the retrieved passages relevant to the question?
    prec_resp = _claude().messages.create(
        model=CLAUDE_MODEL,
        max_tokens=200,
        system=_PRECISION_SYSTEM,
        messages=[{"role": "user", "content": f"Question: {query}\n\nPassages:\n{context}"}],
    )
    prec_raw = next((b.text for b in prec_resp.content if b.type == "text"), "")
    try:
        prec = json.loads(prec_raw.strip())
    except Exception:
        prec = {"score": 0.0, "reason": "parse error"}

    return {
        "faithfulness":      round(float(faith.get("score", 0)), 3),
        "faithfulness_reason": faith.get("reason", ""),
        "context_precision": round(float(prec.get("score", 0)), 3),
        "context_precision_reason": prec.get("reason", ""),
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
