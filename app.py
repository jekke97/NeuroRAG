"""
Streamlit UI for NeuroRAG.
Run with: streamlit run app.py
"""
import os
from datetime import date
import streamlit as st
import streamlit.components.v1 as components

try:
    for _k, _v in st.secrets.items():
        os.environ.setdefault(_k, str(_v))
except FileNotFoundError:
    pass  # running locally — keys come from .env via load_dotenv() in rag.py

from rag import (
    retrieve, retrieve_merged, build_context, check_scope, agent_decide,
    generate_answer, evaluate_rag, CLAUDE_MODEL, TOP_K,
    NAMESPACE_NEURORAG, NAMESPACE_PHILIPS,
)

DAILY_LIMIT = 10

@st.cache_resource
def _rate_store() -> dict:
    return {}

def _client_ip() -> str:
    try:
        forwarded = st.context.headers.get("X-Forwarded-For", "")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return st.context.headers.get("X-Real-Ip", "local")
    except Exception:
        return "local"

def _rate_limit_ok() -> bool:
    store = _rate_store()
    key = f"{_client_ip()}:{date.today().isoformat()}"
    count = store.get(key, 0)
    if count >= DAILY_LIMIT:
        return False
    store[key] = count + 1
    return True

st.set_page_config(
    page_title="NeuroRAG",
    page_icon="◉",
    layout="centered",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: -apple-system, 'Inter', BlinkMacSystemFont, sans-serif !important;
}

.stApp { background: #000000 !important; }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stSidebar"] { display: none !important; }

.block-container {
    max-width: 700px !important;
    padding-top: 5rem !important;
    padding-bottom: 5rem !important;
}

h1 {
    font-size: 2.4rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.04em !important;
    color: #f5f5f7 !important;
    margin-bottom: 0.5rem !important;
    line-height: 1.1 !important;
}

.nr-description {
    color: #86868b;
    font-size: 0.9rem;
    line-height: 1.65;
    margin-bottom: 1.5rem;
    letter-spacing: -0.01em;
}
.nr-description strong { color: #a1a1a6; font-weight: 500; }

.topic-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 2rem;
}
.topic-pill {
    background: #1c1c1e;
    color: #a1a1a6;
    border: 1px solid #3a3a3c;
    border-radius: 980px;
    padding: 4px 14px;
    font-size: 0.78rem;
    font-family: -apple-system, 'Inter', BlinkMacSystemFont, sans-serif;
    letter-spacing: -0.01em;
    white-space: nowrap;
}

textarea {
    border-radius: 14px !important;
    border: 1.5px solid #3a3a3c !important;
    background: #1c1c1e !important;
    font-size: 1rem !important;
    line-height: 1.6 !important;
    color: #f5f5f7 !important;
    resize: none !important;
    transition: border-color 0.15s ease !important;
}
textarea:focus {
    border-color: #86868b !important;
    box-shadow: none !important;
    outline: none !important;
}
textarea::placeholder { color: #48484a !important; }

[data-testid="stFormSubmitButton"] > button {
    background: #f5f5f7 !important;
    color: #000000 !important;
    border: none !important;
    border-radius: 980px !important;
    padding: 0.6rem 2rem !important;
    font-size: 0.95rem !important;
    font-weight: 500 !important;
    letter-spacing: -0.01em !important;
    width: 100% !important;
    margin-top: 0.5rem !important;
    transition: opacity 0.15s ease !important;
    cursor: pointer !important;
}
[data-testid="stFormSubmitButton"] > button:hover { opacity: 0.75 !important; }
[data-testid="stFormSubmitButton"] > button:active { opacity: 0.5 !important; }

[data-testid="stSpinner"] p { color: #86868b !important; font-size: 0.9rem !important; }

h2 {
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em !important;
    color: #f5f5f7 !important;
    margin-top: 2.5rem !important;
    margin-bottom: 0.6rem !important;
    padding-bottom: 0.5rem !important;
    border-bottom: 1px solid #2c2c2e !important;
}

.stMarkdown p { line-height: 1.8; color: #e5e5ea; font-size: 0.97rem; }

/* Out-of-scope answer box */
.out-of-scope-box {
    border: 1.5px solid #ff453a;
    border-radius: 14px;
    background: rgba(255, 69, 58, 0.07);
    padding: 18px 22px;
    margin-top: 1.5rem;
}
.out-of-scope-warning {
    color: #ff6961;
    font-size: 0.8rem;
    font-weight: 500;
    letter-spacing: 0.02em;
    text-transform: uppercase;
    margin-bottom: 10px;
}
.out-of-scope-answer {
    color: #e5e5ea;
    font-size: 0.97rem;
    line-height: 1.8;
}

[data-testid="stExpander"] {
    border: 1px solid #3a3a3c !important;
    border-radius: 12px !important;
    background: #1c1c1e !important;
    margin-top: 1rem !important;
}
[data-testid="stExpander"] summary {
    font-size: 0.82rem !important;
    color: #86868b !important;
    font-weight: 500 !important;
}

hr { border: none !important; border-top: 1px solid #2c2c2e !important; }

/* Corpus selector — segmented control */
[data-testid="stRadio"] > div {
    display: flex !important;
    flex-direction: row !important;
    gap: 4px !important;
    background: #1c1c1e !important;
    border: 1px solid #3a3a3c !important;
    border-radius: 12px !important;
    padding: 4px !important;
    width: fit-content !important;
    margin-bottom: 1.5rem !important;
}
[data-testid="stRadio"] label {
    border-radius: 8px !important;
    padding: 6px 16px !important;
    font-size: 0.83rem !important;
    font-weight: 500 !important;
    color: #86868b !important;
    cursor: pointer !important;
    transition: background 0.15s ease, color 0.15s ease !important;
    white-space: nowrap !important;
}
[data-testid="stRadio"] label:has(input:checked) {
    background: #3a3a3c !important;
    color: #f5f5f7 !important;
}
[data-testid="stRadio"] label input { display: none !important; }
[data-testid="stRadio"] p { margin: 0 !important; }

/* Source badge in citations */
.src-badge {
    display: inline-block;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    padding: 1px 8px;
    border-radius: 980px;
    margin-left: 6px;
    vertical-align: middle;
}
.src-neurorag { background: #1c3a5e; color: #5ac8fa; }
.src-philips  { background: #1a3a2a; color: #30d158; }

/* LLM-as-judge evaluation box */
.eval-box {
    border: 1.5px solid #30d158;
    border-radius: 14px;
    background: rgba(48, 209, 88, 0.07);
    padding: 18px 22px;
    margin-top: 1.5rem;
}
.eval-label {
    color: #30d158;
    font-size: 0.8rem;
    font-weight: 500;
    letter-spacing: 0.02em;
    text-transform: uppercase;
    margin-bottom: 12px;
}
.eval-scores {
    display: flex;
    gap: 2rem;
    margin-bottom: 10px;
}
.eval-score-item { }
.eval-score-name {
    color: #86868b;
    font-size: 0.75rem;
    letter-spacing: 0.01em;
    text-transform: uppercase;
    margin-bottom: 2px;
}
.eval-score-value {
    color: #f5f5f7;
    font-size: 1.4rem;
    font-weight: 600;
    letter-spacing: -0.02em;
}
.eval-reasons {
    color: #86868b;
    font-size: 0.82rem;
    line-height: 1.6;
    border-top: 1px solid rgba(48, 209, 88, 0.2);
    padding-top: 10px;
    margin-top: 4px;
}
</style>
""", unsafe_allow_html=True)

components.html("""
<script>
(function () {
    var doc = window.parent.document;
    function attachEnter() {
        doc.querySelectorAll("textarea").forEach(function (ta) {
            if (ta._nrEnter) return;
            ta._nrEnter = true;
            ta.addEventListener("keydown", function (e) {
                if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    var btn = doc.querySelector("[data-testid='stFormSubmitButton'] button");
                    if (btn) btn.click();
                }
            });
        });
    }
    attachEnter();
    new MutationObserver(attachEnter).observe(doc.body, { childList: true, subtree: true });
})();
</script>
""", height=0)

st.title("NeuroRAG")

st.markdown(
    '<div class="nr-description">'
    '<strong>Proof of concept</strong> · end-to-end RAG over a real neuroscience PhD library.<br>'
    'Ask a question — an agent retrieves the most relevant excerpts from 951 research papers '
    'and generates a grounded, cited answer. Off-topic questions are answered from general '
    'knowledge and flagged. Queries are reformulated automatically when retrieval falls short.'
    '</div>',
    unsafe_allow_html=True,
)

TOPICS = [
    "Attention", "Expectation", "Reward", "fMRI", "Decision-making",
    "Perception", "Dopamine", "Working memory", "Bayesian inference",
    "Neural coding", "EEG / MEG", "Predictive coding", "Consciousness",
    "Learning", "Eye movements",
]
st.markdown(
    '<div class="topic-pills">' +
    "".join(f'<span class="topic-pill">{t}</span>' for t in TOPICS) +
    "</div>",
    unsafe_allow_html=True,
)

_CORPUS_OPTIONS = {
    "My library": NAMESPACE_NEURORAG,
    "Philips research": NAMESPACE_PHILIPS,
    "Both": "both",
}
_CORPUS_LABELS = ["My library", "Philips research", "Both"]

selected_corpus = st.radio(
    "",
    _CORPUS_LABELS,
    horizontal=True,
    label_visibility="collapsed",
)
namespace = _CORPUS_OPTIONS[selected_corpus]

if "result" not in st.session_state:
    st.session_state.result = None
if "ragas_scores" not in st.session_state:
    st.session_state.ragas_scores = None

with st.form("query_form"):
    query = st.text_area(
        "",
        placeholder="Ask anything — e.g. What is the role of dopamine in reward learning?",
        height=120,
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button("Ask", use_container_width=True)

if submitted and query.strip():
    if not _rate_limit_ok():
        st.error(f"Daily limit of {DAILY_LIMIT} queries reached. Come back tomorrow.")
    else:
        st.session_state.ragas_scores = None
        q = query.strip()
        agent_steps = []

        _corpus_label = selected_corpus
        with st.status("Agent working…", expanded=True) as status:
            st.write(f"Corpus: **{_corpus_label}** — checking whether the question is within scope…")
            scope = check_scope(q, namespace)

            if not scope.get("in_scope", True):
                st.write("Question is outside the corpus — answering from general knowledge.")
                status.update(label="Answered from general knowledge", state="complete")
                st.session_state.result = {
                    "query": q, "answer": scope.get("response", ""),
                    "chunks": [], "citations": [], "usage": {},
                    "agent_steps": [], "out_of_scope": True,
                }
            else:
                st.write("Question is within scope. Retrieving relevant excerpts…")
                if namespace == "both":
                    chunks = retrieve_merged(q, TOP_K)
                else:
                    chunks = retrieve(q, TOP_K, namespace)
                context = build_context(chunks)

                st.write("Evaluating context quality…")
                decision = agent_decide(q, context[:2000])

                if decision.get("action") == "reformulate" and decision.get("query"):
                    ref_q = decision["query"]
                    st.write(f"Context insufficient — reformulating query to: *\"{ref_q}\"*")
                    agent_steps.append({"original": q, "reformulated": ref_q})
                    if namespace == "both":
                        new_chunks = retrieve_merged(ref_q, TOP_K)
                    else:
                        new_chunks = retrieve(ref_q, TOP_K, namespace)
                    seen = {f"{c['meta']['filename']}::{c['meta']['chunk']}" for c in chunks}
                    for c in new_chunks:
                        cid = f"{c['meta']['filename']}::{c['meta']['chunk']}"
                        if cid not in seen:
                            chunks.append(c)
                            seen.add(cid)
                    chunks = chunks[:TOP_K]
                else:
                    st.write("Context quality sufficient.")

                st.write("Generating answer…")
                gen = generate_answer(q, chunks)

                st.write("Validating answer quality (LLM-as-judge)…")
                try:
                    scores = evaluate_rag(q, gen["answer"], chunks)
                except Exception:
                    scores = None

                status.update(label="Done", state="complete")

                st.session_state.ragas_scores = scores
                st.session_state.result = {
                    "query": q, "answer": gen["answer"], "chunks": chunks,
                    "citations": [
                        {"idx": i+1, "authors": c["meta"]["authors"],
                         "year": c["meta"]["year"], "title": c["meta"]["title"],
                         "filename": c["meta"]["filename"],
                         "source": c["meta"].get("source", namespace),
                         "doi": c["meta"].get("doi", "")}
                        for i, c in enumerate(chunks)
                    ],
                    "usage": gen["usage"],
                    "agent_steps": agent_steps,
                    "out_of_scope": False,
                    "namespace": namespace,
                }

if st.session_state.result:
    result = st.session_state.result

    if result.get("out_of_scope"):
        st.markdown(
            f'<div class="out-of-scope-box">'
            f'<div class="out-of-scope-warning">⚠ Outside corpus — answered from general knowledge</div>'
            f'<div class="out-of-scope-answer">{result["answer"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown("## Answer")
        st.markdown(result["answer"])

        st.markdown("## Sources")
        show_badge = result.get("namespace") == "both"
        for c in result["citations"]:
            badge = ""
            if show_badge:
                src = c.get("source", "")
                if src == "philips":
                    badge = '<span class="src-badge src-philips">Philips</span>'
                else:
                    badge = '<span class="src-badge src-neurorag">Library</span>'
            doi_link = f' · <a href="https://doi.org/{c["doi"]}" target="_blank" style="color:#86868b;font-size:0.82rem;">DOI</a>' if c.get("doi") else ""
            st.markdown(
                f'<p style="margin:0.3rem 0;color:#e5e5ea;font-size:0.95rem;">'
                f'<strong>[{c["idx"]}]</strong> {c["authors"]} ({c["year"]}) — <em>{c["title"]}</em>'
                f'{badge}{doi_link}</p>',
                unsafe_allow_html=True,
            )

        if st.session_state.ragas_scores:
            scores = st.session_state.ragas_scores
            st.markdown(
                f'<div class="eval-box">'
                f'<div class="eval-label">✓ LLM-as-judge validation</div>'
                f'<div class="eval-scores">'
                f'  <div class="eval-score-item">'
                f'    <div class="eval-score-name">Faithfulness</div>'
                f'    <div class="eval-score-value">{scores["faithfulness"]:.2f}<span style="font-size:0.9rem;color:#86868b;font-weight:400"> / 1.0</span></div>'
                f'  </div>'
                f'  <div class="eval-score-item">'
                f'    <div class="eval-score-name">Context precision</div>'
                f'    <div class="eval-score-value">{scores["context_precision"]:.2f}<span style="font-size:0.9rem;color:#86868b;font-weight:400"> / 1.0</span></div>'
                f'  </div>'
                f'</div>'
                f'<div class="eval-reasons">'
                f'{scores.get("faithfulness_reason", "")} &nbsp;·&nbsp; {scores.get("context_precision_reason", "")}'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

