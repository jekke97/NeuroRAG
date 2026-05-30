"""
Fetch metadata for Philips-affiliated MRI/fMRI papers from PubMed.
Outputs philips_papers.csv — titles, authors, DOIs, PMC IDs, open-access flag, abstracts.

This is a metadata-only step. Full-text download comes later.

Run: python fetch_philips_papers.py
"""

import csv
import time
import xml.etree.ElementTree as ET
import requests

# Optional but recommended: get a free NCBI API key at ncbi.nlm.nih.gov/account
# Without it: 3 requests/second limit. With it: 10 requests/second.
NCBI_API_KEY = ""
NCBI_EMAIL   = "ettorecerracchio@gmail.com"

# PubMed search query:
# - Philips[Affiliation]  → author must be affiliated with Philips (not just used a Philips scanner)
# - Title/Abstract terms  → restrict to MRI/fMRI papers
QUERY = (
    'Philips[Affiliation] AND ('
    'MRI[Title/Abstract] OR '
    'fMRI[Title/Abstract] OR '
    '"magnetic resonance imaging"[Title/Abstract] OR '
    '"functional MRI"[Title/Abstract] OR '
    '"functional magnetic resonance"[Title/Abstract]'
    ')'
)

OUTPUT_FILE = "philips_papers.csv"
BATCH_SIZE  = 200
SLEEP       = 0.12 if NCBI_API_KEY else 0.4  # stay within rate limits

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def _get(endpoint: str, params: dict) -> requests.Response:
    params["tool"]  = "NeuroRAG"
    params["email"] = NCBI_EMAIL
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
    r = requests.get(f"{BASE}/{endpoint}", params=params, timeout=30)
    r.raise_for_status()
    time.sleep(SLEEP)
    return r


def search(query: str) -> tuple[int, str, str]:
    """Return (total_count, webenv, query_key) using NCBI history server."""
    r = _get("esearch.fcgi", {
        "db":         "pubmed",
        "term":       query,
        "usehistory": "y",
        "retmax":     0,
        "retmode":    "xml",
    })
    root      = ET.fromstring(r.text)
    count     = int(root.findtext("Count", "0"))
    webenv    = root.findtext("WebEnv", "")
    query_key = root.findtext("QueryKey", "")
    return count, webenv, query_key


def fetch_batch(webenv: str, query_key: str, retstart: int) -> str:
    r = _get("efetch.fcgi", {
        "db":        "pubmed",
        "WebEnv":    webenv,
        "query_key": query_key,
        "retstart":  retstart,
        "retmax":    BATCH_SIZE,
        "retmode":   "xml",
    })
    return r.text


def parse_batch(xml_text: str) -> list[dict]:
    root    = ET.fromstring(xml_text)
    records = []

    for article in root.findall(".//PubmedArticle"):

        # PMID
        pmid = article.findtext(".//PMID", "")

        # Title
        title = article.findtext(".//ArticleTitle", "").strip()

        # Authors
        authors = []
        for a in article.findall(".//Author"):
            last  = a.findtext("LastName", "")
            first = a.findtext("ForeName", "")
            name  = f"{last}, {first}".strip(", ")
            if name:
                authors.append(name)

        # Year
        year = article.findtext(".//PubDate/Year", "")
        if not year:
            medline = article.findtext(".//MedlineDate", "")
            year    = medline[:4] if medline else ""

        # Journal
        journal = article.findtext(".//Journal/Title", "")

        # DOI and PMC ID (from ArticleIdList)
        doi    = ""
        pmc_id = ""
        for id_el in article.findall(".//ArticleId"):
            id_type = id_el.get("IdType", "")
            if id_type == "doi":
                doi    = id_el.text or ""
            elif id_type == "pmc":
                pmc_id = id_el.text or ""

        # Abstract (handles structured abstracts with labelled sections)
        parts = article.findall(".//AbstractText")
        abstract = " ".join(
            (f"{p.get('Label')}: " if p.get("Label") else "") + (p.text or "")
            for p in parts
        ).strip()

        # Affiliations (deduplicated)
        affiliations = list(dict.fromkeys(
            aff.text for aff in article.findall(".//AffiliationInfo/Affiliation")
            if aff.text
        ))

        # Philips-specific affiliation flag (true = author IS at Philips, not just used their scanner)
        philips_author = any(
            "philips" in aff.lower() for aff in affiliations
        )

        records.append({
            "pmid":           pmid,
            "title":          title,
            "authors":        "; ".join(authors),
            "year":           year,
            "journal":        journal,
            "doi":            doi,
            "doi_url":        f"https://doi.org/{doi}" if doi else "",
            "pmc_id":         pmc_id,
            "pmc_url":        f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/" if pmc_id else "",
            "open_access":    "yes" if pmc_id else "no",
            "abstract":       abstract,
            "affiliations":   " | ".join(affiliations),
            "philips_author": "yes" if philips_author else "no",
            "pubmed_url":     f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
        })

    return records


FIELDS = [
    "pmid", "title", "authors", "year", "journal",
    "doi", "doi_url", "pmc_id", "pmc_url", "open_access",
    "abstract", "affiliations", "philips_author", "pubmed_url",
]


def main():
    print(f"Query:\n  {QUERY}\n")

    count, webenv, query_key = search(QUERY)
    print(f"Total results: {count}\n")

    if count == 0:
        print("No results. Check your query or network connection.")
        return

    all_records = []
    for start in range(0, count, BATCH_SIZE):
        end = min(start + BATCH_SIZE, count)
        print(f"  Fetching {start + 1}–{end} …")
        xml_text = fetch_batch(webenv, query_key, start)
        all_records.extend(parse_batch(xml_text))

    # Sort by year descending
    all_records.sort(key=lambda r: r["year"], reverse=True)

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(all_records)

    oa    = sum(1 for r in all_records if r["open_access"] == "yes")
    total = len(all_records)
    print(f"\n✓ {total} papers saved to {OUTPUT_FILE}")
    print(f"  Open access (full text downloadable): {oa:>5}  ({100*oa//total}%)")
    print(f"  Abstract only:                        {total-oa:>5}  ({100*(total-oa)//total}%)")
    print(f"\nNext step: run ingest_philips.py to embed and index the open-access papers.")


if __name__ == "__main__":
    main()
