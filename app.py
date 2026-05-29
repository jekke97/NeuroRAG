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

from rag import ask, evaluate_rag, CLAUDE_MODEL

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

.block-container {
    max-width: 700px !important;
    padding-top: 5rem !important;
    padding-bottom: 5rem !important;
}

/* Title */
h1 {
    font-size: 2.4rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.04em !important;
    color: #f5f5f7 !important;
    margin-bottom: 0.15rem !important;
    line-height: 1.1 !important;
}

/* Caption */
[data-testid="stCaptionContainer"] p {
    color: #86868b !important;
    font-size: 0.9rem !important;
    letter-spacing: -0.01em !important;
    margin-bottom: 0.75rem !important;
}

/* Topic pills */
.topic-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 2rem;
    margin-top: 0.25rem;
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

/* Textarea */
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

/* Form submit button — white pill */
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

/* Spinner */
[data-testid="stSpinner"] p { color: #86868b !important; font-size: 0.9rem !important; }

/* Section headings */
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

/* Answer + citation text */
.stMarkdown p { line-height: 1.8; color: #e5e5ea; font-size: 0.97rem; }

/* Expander */
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

/* Sidebar */
[data-testid="stSidebar"] {
    background: #1c1c1e !important;
    border-right: 1px solid #3a3a3c !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] li,
[data-testid="stSidebar"] label {
    color: #86868b !important;
    font-size: 0.83rem !important;
    line-height: 1.6 !important;
}
[data-testid="stSidebar"] h2 {
    color: #f5f5f7 !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    border: none !important;
    padding: 0 !important;
    margin-top: 0 !important;
}
[data-testid="stSidebar"] hr { border-color: #3a3a3c !important; }

/* Slider */
[data-testid="stSlider"] [role="slider"] {
    background: #f5f5f7 !important;
    border-color: #f5f5f7 !important;
}

/* Divider */
hr { border: none !important; border-top: 1px solid #2c2c2e !important; }
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
    "Proof of concept · end-to-end RAG over a real neuroscience PhD library — "
    "semantic retrieval + Claude-generated cited answers, built for clinical AI applications.",
    unsafe_allow_html=False,
)
st.caption(f"Semantic search over 951 neuroscience papers · {CLAUDE_MODEL}")

TOPICS = [
    "Attention", "Expectation", "Reward", "fMRI", "Decision-making",
    "Perception", "Dopamine", "Working memory", "Bayesian inference",
    "Neural coding", "EEG / MEG", "Predictive coding", "Consciousness",
    "Learning", "Eye movements",
]
pills_html = '<div class="topic-pills">' + "".join(
    f'<span class="topic-pill">{t}</span>' for t in TOPICS
) + "</div>"
st.markdown(pills_html, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("#### Settings")
    k = st.slider(
        "Sources to consult",
        min_value=4,
        max_value=20,
        value=8,
        help="Higher = broader context, slower response",
    )
    st.markdown("---")
    st.markdown(
        "**How it works**\n\n"
        "1. Your question is embedded into a vector\n"
        "2. The most relevant paper excerpts are retrieved\n"
        "3. Claude synthesises a cited answer\n\n"
        "**Corpus:** your Zotero library (951 papers, 46 K excerpts)\n\n"
        "*Shift + Enter for a new line.*"
    )

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
        with st.spinner("Searching and generating…"):
            st.session_state.result = ask(query.strip(), k=k)

if st.session_state.result:
    result = st.session_state.result

    # Agent trace — show if the query was reformulated
    if result.get("agent_steps"):
        step = result["agent_steps"][0]
        st.markdown(
            f'<p style="color:#86868b;font-size:0.83rem;margin-bottom:1.5rem;">'
            f'↻ Query reformulated to: <em>"{step["reformulated"]}"</em></p>',
            unsafe_allow_html=True,
        )

    st.markdown("## Answer")
    st.markdown(result["answer"])

    st.markdown("## Sources")
    for c in result["citations"]:
        st.markdown(
            f"**[{c['idx']}]** {c['authors']} ({c['year']}) — *{c['title']}*"
        )

    st.markdown("## Quality evaluation")
    if st.session_state.ragas_scores is None:
        if st.button("Run RAGAS evaluation", use_container_width=True):
            with st.spinner("Evaluating faithfulness and context precision — ~30 s…"):
                try:
                    st.session_state.ragas_scores = evaluate_rag(
                        query=result["query"],
                        answer=result["answer"],
                        chunks=result["chunks"],
                    )
                except Exception as e:
                    st.error(f"Evaluation failed: {e}")

    if st.session_state.ragas_scores:
        scores = st.session_state.ragas_scores
        col1, col2 = st.columns(2)
        col1.metric(
            "Faithfulness",
            f"{scores['faithfulness']:.2f} / 1.0",
            help="Are the answer's claims supported by the retrieved excerpts?",
        )
        col2.metric(
            "Context precision",
            f"{scores['context_precision']:.2f} / 1.0",
            help="Are the retrieved excerpts relevant to the question?",
        )
        if scores.get("faithfulness_reason") or scores.get("context_precision_reason"):
            st.markdown(
                f'<p style="color:#86868b;font-size:0.82rem;margin-top:0.5rem;">'
                f'Faithfulness: {scores["faithfulness_reason"]} &nbsp;·&nbsp; '
                f'Precision: {scores["context_precision_reason"]}</p>',
                unsafe_allow_html=True,
            )

    with st.expander("Token usage"):
        st.json(result["usage"])
