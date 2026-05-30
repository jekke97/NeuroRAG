"""
Ingest Philips MRI/fMRI papers into Pinecone namespace 'philips'.

For open-access papers (pmc_id present): fetches full text from PMC API.
For abstract-only papers: uses the abstract from philips_papers.csv.

Prerequisites:
  1. Run fetch_philips_papers.py to generate philips_papers.csv
  2. PINECONE_API_KEY set in .env

Run: python ingest_philips.py
"""
import os
import sys
import csv
import time
import xml.etree.ElementTree as ET
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from pinecone.errors.exceptions import NotFoundError as PineconeNotFoundError
from tqdm import tqdm
import requests

load_dotenv()

CSV_FILE      = "philips_papers.csv"
INDEX_NAME    = "neurorag"
NAMESPACE     = "philips"
EMBED_MODEL   = "all-MiniLM-L6-v2"
EMBED_DIM     = 384
CHUNK_SIZE    = 300
CHUNK_OVERLAP = 30
BATCH_SIZE    = 100

NCBI_EMAIL   = "ettorecerracchio@gmail.com"
NCBI_API_KEY = ""  # optional — set for higher rate limits
SLEEP        = 0.12 if NCBI_API_KEY else 0.4


def fetch_pmc_fulltext(pmc_id: str) -> str:
    """Fetch full article body text from PMC via JATS XML."""
    params = {
        "db":      "pmc",
        "id":      pmc_id,
        "retmode": "xml",
        "tool":    "NeuroRAG",
        "email":   NCBI_EMAIL,
    }
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY

    try:
        r = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            params=params,
            timeout=30,
        )
        r.raise_for_status()
        time.sleep(SLEEP)
    except Exception as e:
        print(f"  ⚠ PMC fetch failed for {pmc_id}: {e}")
        return ""

    return _parse_pmc_xml(r.text)


def _parse_pmc_xml(xml_text: str) -> str:
    """Extract body paragraph text from JATS XML, skipping figures/tables/refs."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return ""

    body = root.find(".//body")
    if body is None:
        # Some PMC records only have abstract — fall through to abstract fallback
        return ""

    paragraphs = []
    for elem in body.iter():
        # Skip figure captions, table content, and reference lists
        if elem.tag in ("fig", "table-wrap", "ref-list", "supplementary-material"):
            continue
        if elem.tag == "p" and elem.text:
            text = "".join(elem.itertext()).strip()
            if len(text) > 40:
                paragraphs.append(text)

    return "\n\n".join(paragraphs)


def chunk_text(text: str) -> list[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i : i + CHUNK_SIZE])
        if len(chunk.strip()) > 50:
            chunks.append(chunk)
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def main():
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        sys.exit("PINECONE_API_KEY not set in .env")

    if not os.path.exists(CSV_FILE):
        sys.exit(f"{CSV_FILE} not found. Run fetch_philips_papers.py first.")

    with open(CSV_FILE, encoding="utf-8") as f:
        papers = list(csv.DictReader(f))

    print(f"Loaded {len(papers)} papers from {CSV_FILE}")
    oa = sum(1 for p in papers if p["pmc_id"])
    print(f"  Open access (full text): {oa}")
    print(f"  Abstract only:           {len(papers) - oa}\n")

    print(f"Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    pc = Pinecone(api_key=api_key)
    existing = [idx.name for idx in pc.list_indexes()]
    if INDEX_NAME not in existing:
        print(f"Creating Pinecone index '{INDEX_NAME}'…")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    index = pc.Index(INDEX_NAME)

    print(f"Clearing namespace '{NAMESPACE}' for fresh ingest…")
    try:
        index.delete(delete_all=True, namespace=NAMESPACE)
    except PineconeNotFoundError:
        pass  # namespace doesn't exist yet — nothing to clear

    all_ids, all_texts, all_metas = [], [], []
    full_text_count = 0
    abstract_count  = 0

    for paper in tqdm(papers, desc="Fetching text"):
        pmid    = paper["pmid"]
        pmc_id  = paper["pmc_id"]
        title   = paper["title"]
        authors = paper["authors"].split(";")[0].strip() if paper["authors"] else ""
        year    = paper["year"]
        doi     = paper["doi"]

        # Try full text first, fall back to abstract
        if pmc_id:
            text = fetch_pmc_fulltext(pmc_id)
            if text:
                full_text_count += 1
            else:
                text = paper.get("abstract", "")
                abstract_count += 1
        else:
            text = paper.get("abstract", "")
            abstract_count += 1

        if not text.strip():
            continue

        for i, chunk in enumerate(chunk_text(text)):
            chunk_id = f"pmid{pmid}::{i}"
            all_ids.append(chunk_id)
            all_texts.append(chunk)
            all_metas.append({
                "authors":  authors,
                "year":     year,
                "title":    title,
                "filename": pmc_id if pmc_id else f"pmid_{pmid}",
                "chunk":    i,
                "text":     chunk,
                "source":   "philips",
                "doi":      doi,
                "pmid":     pmid,
            })

    print(f"\nTotal chunks: {len(all_ids)}")
    print(f"  From full text: {full_text_count} papers")
    print(f"  From abstracts: {abstract_count} papers")
    print("Embedding and indexing…\n")

    for start in tqdm(range(0, len(all_ids), BATCH_SIZE), desc="Indexing"):
        batch_ids   = all_ids  [start : start + BATCH_SIZE]
        batch_texts = all_texts[start : start + BATCH_SIZE]
        batch_metas = all_metas[start : start + BATCH_SIZE]
        embeddings  = model.encode(batch_texts, show_progress_bar=False).tolist()
        vectors = [
            {"id": vid, "values": vec, "metadata": meta}
            for vid, vec, meta in zip(batch_ids, embeddings, batch_metas)
        ]
        index.upsert(vectors=vectors, namespace=NAMESPACE)

    stats    = index.describe_index_stats()
    ns_count = stats.get("namespaces", {}).get(NAMESPACE, {}).get("vector_count", len(all_ids))
    print(f"\n✓ Done. {ns_count} vectors in namespace '{NAMESPACE}'")


if __name__ == "__main__":
    main()
