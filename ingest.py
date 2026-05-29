"""
One-time ingestion: PDFs from Google Drive Zotero folder → Pinecone.
Run this once before using the app or CLI.
"""
import os
import re
import glob
import sys
from dotenv import load_dotenv
import fitz  # pymupdf
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

load_dotenv()

DRIVE_PATH    = "/Users/ettore/Library/CloudStorage/GoogleDrive-ettorecerracchio@gmail.com/My Drive/IMCN/Zotero"
INDEX_NAME    = "neurorag"
EMBED_MODEL   = "all-MiniLM-L6-v2"
EMBED_DIM     = 384
CHUNK_SIZE    = 300   # words — fits within all-MiniLM-L6-v2's ~256-token window
CHUNK_OVERLAP = 30
BATCH_SIZE    = 100   # Pinecone upsert batch size (max 100 vectors per call)


def parse_filename(path: str) -> dict:
    """Extract authors/year/title from 'Authors_Year_Title.pdf'."""
    base = os.path.splitext(os.path.basename(path))[0]
    m = re.match(r"^(.+?)_(\d{4})_(.+)$", base)
    if m:
        return {"authors": m.group(1), "year": m.group(2), "title": m.group(3)}
    return {"authors": "", "year": "", "title": base}


def extract_text(path: str) -> str:
    try:
        doc = fitz.open(path)
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        return text
    except Exception as e:
        print(f"  ⚠ skipped {os.path.basename(path)}: {e}", file=sys.stderr)
        return ""


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

    pdf_files = sorted(glob.glob(os.path.join(DRIVE_PATH, "*.pdf")))
    if not pdf_files:
        sys.exit(f"No PDFs found in {DRIVE_PATH}")
    print(f"Found {len(pdf_files)} PDFs")

    print(f"Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    pc = Pinecone(api_key=api_key)

    # Always delete and recreate for a clean ingest
    existing = [idx.name for idx in pc.list_indexes()]
    if INDEX_NAME in existing:
        print(f"Deleting existing index '{INDEX_NAME}' for fresh ingest…")
        pc.delete_index(INDEX_NAME)
    print(f"Creating Pinecone index '{INDEX_NAME}'…")
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBED_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print("Index created.")

    index = pc.Index(INDEX_NAME)

    # Collect all chunks
    all_ids, all_texts, all_metas = [], [], []

    for pdf_path in tqdm(pdf_files, desc="Extracting text"):
        meta = parse_filename(pdf_path)
        text = extract_text(pdf_path)
        if not text.strip():
            continue
        for i, chunk in enumerate(chunk_text(text)):
            raw_id = f"{os.path.basename(pdf_path)}::{i}"
            chunk_id = raw_id.encode("ascii", errors="ignore").decode("ascii")
            all_ids.append(chunk_id)
            all_texts.append(chunk)
            all_metas.append({
                **meta,
                "filename": os.path.basename(pdf_path),
                "chunk": i,
                "text": chunk,  # store text in metadata for retrieval
            })

    print(f"\nTotal chunks: {len(all_ids)} — embedding and indexing…")

    for start in tqdm(range(0, len(all_ids), BATCH_SIZE), desc="Indexing batches"):
        batch_ids   = all_ids  [start : start + BATCH_SIZE]
        batch_texts = all_texts[start : start + BATCH_SIZE]
        batch_metas = all_metas[start : start + BATCH_SIZE]
        embeddings  = model.encode(batch_texts, show_progress_bar=False).tolist()
        vectors = [
            {"id": vid, "values": vec, "metadata": meta}
            for vid, vec, meta in zip(batch_ids, embeddings, batch_metas)
        ]
        index.upsert(vectors=vectors)

    stats = index.describe_index_stats()
    print(f"\n✓ Done. {stats['total_vector_count']} vectors in Pinecone index '{INDEX_NAME}'")


if __name__ == "__main__":
    main()
