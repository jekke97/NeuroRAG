"""
Streamlit UI for NeuroRAG.
Run with: streamlit run app.py
"""
import streamlit as st
from rag import ask

st.set_page_config(
    page_title="NeuroRAG",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 NeuroRAG")
from rag import CLAUDE_MODEL
st.caption(f"Semantic Q&A over your neuroscience library · Pinecone + {CLAUDE_MODEL}")

with st.sidebar:
    st.header("Settings")
    k = st.slider("Chunks to retrieve", min_value=4, max_value=20, value=8)
    st.markdown("---")
    st.markdown(
        "**How it works**\n"
        "1. Your query is embedded with `all-MiniLM-L6-v2`\n"
        "2. Top-K chunks are retrieved from ChromaDB\n"
        "3. Claude synthesizes a cited answer\n\n"
        "**Corpus:** your Zotero Google Drive library"
    )

query = st.text_area(
    "Ask a question about your papers:",
    placeholder=(
        "What does the literature say about prior probability effects "
        "on perceptual decision-making?"
    ),
    height=90,
)

if st.button("Ask", type="primary") and query.strip():
    with st.spinner("Retrieving and generating…"):
        result = ask(query.strip(), k=k)

    st.markdown("## Answer")
    st.markdown(result["answer"])

    st.markdown("## Sources retrieved")
    for c in result["citations"]:
        st.markdown(
            f"**[{c['idx']}]** {c['authors']} ({c['year']}) — *{c['title']}*"
        )

    with st.expander("Token usage"):
        st.json(result["usage"])
