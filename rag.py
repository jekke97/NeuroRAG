"""
Core RAG pipeline: query → retrieve → generate with Claude.
Also runnable as CLI: python rag.py "your question here"
"""
from __future__ import annotations
import os
import sys
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

# Lazy singletons — loaded once per process
_embed_model: SentenceTransformer | None = None
_index = None
_client: anthropic.Anthropic | None      = None


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
    res = _pinecone().query(
        vector=vec,
        top_k=k,
        include_metadata=True,
    )
    return [
        {
            "text": match["metadata"].get("text", ""),
            "meta": {
                "authors":  match["metadata"].get("authors", ""),
                "year":     match["metadata"].get("year", ""),
                "title":    match["metadata"].get("title", ""),
                "filename": match["metadata"].get("filename", ""),
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


def ask(query: str, k: int = TOP_K) -> dict:
    chunks  = retrieve(query, k)
    context = _build_context(chunks)

    user_msg = (
        f"Literature excerpts:\n\n{context}\n\n"
        f"---\n\nQuestion: {query}"
    )

    response = _claude().messages.create(
        model=CLAUDE_MODEL,
        max_tokens=4096,
        system=SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )

    answer = next(
        (b.text for b in response.content if b.type == "text"), ""
    )

    return {
        "answer": answer,
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
    }


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or (
        "What does the literature say about prior probability effects on perceptual decisions?"
    )
    print(f"Query: {query}\n")
    result = ask(query)
    print(result["answer"])
    print("\n--- Sources ---")
    for c in result["citations"]:
        print(f"[{c['idx']}] {c['authors']} ({c['year']}) — {c['title']}")
    u = result["usage"]
    print(f"\nTokens — input: {u['input_tokens']}, output: {u['output_tokens']}")
