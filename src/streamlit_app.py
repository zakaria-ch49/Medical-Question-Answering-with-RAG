import os
import sys
import json
import html as html_module
import requests
import streamlit as st
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

# ─── Path & env setup (avant tout autre import) ───────────────────────────────
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(APP_DIR, "..")
sys.path.insert(0, APP_DIR)
load_dotenv(os.path.join(ROOT_DIR, ".env"))

# ─── Configuration de la page (doit être le 1er appel Streamlit) ──────────────
st.set_page_config(
    page_title="Medical RAG Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "Medical RAG Assistant — PubMed + FAISS + OpenRouter\n\n⚠️ For educational use only.",
    },
)

# ─── Imports des modules du projet ────────────────────────────────────────────
from download_pubmed import download_articles
from bio_clinical_embeddings import (
    load_documents_from_articles,
    create_vector_store,
    search_with_score,
)
from open_router import query_openrouter, stream_openrouter, OPENROUTER_API_KEY

# ─── Constantes ───────────────────────────────────────────────────────────────
DEFAULT_RETMAX    = 50
DEFAULT_TOP_K     = 5
SCORE_THRESHOLD   = 0.30   # Seuil strict : distance BGE ≤ 0.30 (hors-sujet filtrés)
MODEL_NAME        = "qwen/qwen3-vl-235b-a22b-thinking"
EMBEDDING_MODEL   = "BAAI/bge-base-en"

# ─── CSS professionnel ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ══ Reset & base ══════════════════════════════════════════════════════════ */
html, body { font-family: 'Inter', sans-serif; }
* { font-family: 'Inter', sans-serif; }

/* ══ Fond principal dark ═══════════════════════════════════════════════════ */
[data-testid="stAppViewContainer"]         { background: #0d1117 !important; }
[data-testid="stAppViewContainer"] > div  { background: #0d1117 !important; }
.main .block-container                     { background: #0d1117 !important; }
[data-testid="stMainBlockContainer"]       { background: #0d1117 !important; }

/* ══ Sidebar dark ══════════════════════════════════════════════════════════ */
[data-testid="stSidebar"]           { background: #080d14 !important; border-right: 1px solid #1e293b; }
[data-testid="stSidebar"] *         { color: #cbd5e1 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3        { color: #f1f5f9 !important; }
[data-testid="stSidebar"] .stDivider { border-color: #1e293b !important; }

/* ══ Texte global clair ════════════════════════════════════════════════════ */
p, span, div, label, li { color: #cbd5e1; }
h1, h2, h3, h4, h5      { color: #f1f5f9; }

/* ══ Header principal ══════════════════════════════════════════════════════ */
.pro-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f4c81 100%);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    display: flex;
    align-items: center;
    gap: 2rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.5);
    position: relative;
    overflow: hidden;
}
.pro-header::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: rgba(255,255,255,0.04);
    border-radius: 50%;
}
.pro-header::after {
    content: '';
    position: absolute;
    bottom: -80px; left: 30%;
    width: 300px; height: 300px;
    background: rgba(255,255,255,0.03);
    border-radius: 50%;
}
.pro-header-text h1 {
    color: #ffffff !important;
    font-size: 2rem;
    font-weight: 700;
    margin: 0 0 0.35rem 0;
    letter-spacing: -0.5px;
}
.pro-header-text p {
    color: #93c5fd !important;
    font-size: 0.95rem;
    margin: 0;
    font-weight: 400;
}
.pro-badge {
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.18);
    color: #e2e8f0 !important;
    border-radius: 20px;
    padding: 0.3rem 0.9rem;
    font-size: 0.75rem;
    font-weight: 500;
    text-align: center;
}

/* ══ Disclaimer ════════════════════════════════════════════════════════════ */
.pro-disclaimer {
    background: #1c1a09;
    border: 1px solid #44370a;
    border-left: 4px solid #ca8a04;
    border-radius: 10px;
    padding: 1rem 1.4rem;
    margin-bottom: 1.5rem;
}
.pro-disclaimer-title {
    font-weight: 600;
    color: #fde68a !important;
    font-size: 0.88rem;
    margin-bottom: 0.4rem;
}
.pro-disclaimer ul { margin: 0; padding-left: 1.2rem; color: #fef3c7 !important; font-size: 0.82rem; line-height: 1.7; }

/* ══ Search card ═══════════════════════════════════════════════════════════ */
.search-card {
    background: #161b22;
    border-radius: 14px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    border: 1px solid #30363d;
}
.search-card-title {
    font-size: 1rem;
    font-weight: 600;
    color: #e6edf3 !important;
    margin-bottom: 1rem;
}

/* ══ Chat input bar ══════════════════════════════════════════════════════════ */
/* Kill ALL default Streamlit red/orange focus rings globally */
*:focus, *:focus-visible, *:focus-within {
    outline: none !important;
    box-shadow: none !important;
}
[data-testid="stBottom"] {
    background: #0d1117 !important;
    padding: 0.75rem 1.5rem 1rem 1.5rem !important;
    border-top: 1px solid #1e293b !important;
}
[data-testid="stChatInput"] {
    background: #1c2333 !important;
    border-radius: 30px !important;
    border: 1.5px solid #30363d !important;
    padding: 4px 8px 4px 18px !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.35) !important;
    display: flex !important;
    align-items: center !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
    outline: none !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #2563eb !important;
    box-shadow: 0 4px 24px rgba(37,99,235,0.22) !important;
    outline: none !important;
}
[data-testid="stChatInput"] *,
[data-testid="stChatInput"] *:focus,
[data-testid="stChatInput"] *:focus-visible {
    outline: none !important;
    box-shadow: none !important;
    border-color: transparent !important;
}
[data-testid="stChatInputTextArea"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
    color: #e6edf3 !important;
    font-size: 0.97rem !important;
    font-family: 'Inter', sans-serif !important;
    caret-color: #2563eb !important;
    resize: none !important;
    padding: 10px 0 10px 4px !important;
}
[data-testid="stChatInputTextArea"]:focus,
[data-testid="stChatInputTextArea"]:focus-visible {
    outline: none !important;
    box-shadow: none !important;
    border: none !important;
}
[data-testid="stChatInputTextArea"]::placeholder {
    color: #64748b !important;
    font-style: normal !important;
}
/* Send button — blue circle */
[data-testid="stChatInputSubmitButton"] {
    background: #2563eb !important;
    border-radius: 50% !important;
    width: 42px !important;
    height: 42px !important;
    min-width: 42px !important;
    min-height: 42px !important;
    border: none !important;
    outline: none !important;
    box-shadow: 0 2px 10px rgba(37,99,235,0.45) !important;
    transition: background 0.2s, transform 0.15s, box-shadow 0.2s !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    flex-shrink: 0 !important;
}
[data-testid="stChatInputSubmitButton"]:hover {
    background: #1d4ed8 !important;
    transform: scale(1.08) !important;
    box-shadow: 0 4px 18px rgba(37,99,235,0.6) !important;
}
[data-testid="stChatInputSubmitButton"] svg {
    fill: #ffffff !important;
    width: 18px !important;
    height: 18px !important;
}

/* ══ Bouton principal ══════════════════════════════════════════════════════ */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.7rem 1.5rem !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 8px rgba(37,99,235,0.4) !important;
}
div[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, #1e40af, #1d4ed8) !important;
    box-shadow: 0 4px 16px rgba(37,99,235,0.6) !important;
    transform: translateY(-1px) !important;
}

/* ══ Section cards ═════════════════════════════════════════════════════════ */
.section-card {
    background: #161b22;
    border-radius: 14px;
    padding: 1.6rem 2rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    border: 1px solid #30363d;
}
.section-title {
    font-size: 1rem;
    font-weight: 700;
    color: #e6edf3 !important;
    margin-bottom: 1.2rem;
    padding-bottom: 0.7rem;
    border-bottom: 2px solid #1e3a5f;
}

/* ══ Metric pills ══════════════════════════════════════════════════════════ */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 0.75rem;
    margin-top: 0.5rem;
}
.metric-pill {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 0.9rem 0.75rem;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-pill:hover { border-color: #2563eb; }
.metric-pill .mp-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #60a5fa !important;
    line-height: 1;
}
.metric-pill .mp-label {
    font-size: 0.72rem;
    color: #64748b !important;
    margin-top: 0.3rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

/* ══ Document cards ════════════════════════════════════════════════════════ */
.doc-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 1rem 1.3rem;
    margin-bottom: 0.8rem;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.doc-card:hover { border-color: #2563eb; box-shadow: 0 4px 20px rgba(37,99,235,0.15); }
.doc-card-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 1rem;
    margin-bottom: 0.6rem;
}
.doc-number {
    background: #1d4ed8;
    color: #ffffff !important;
    border-radius: 6px;
    padding: 0.2rem 0.55rem;
    font-size: 0.75rem;
    font-weight: 700;
    flex-shrink: 0;
    margin-top: 0.1rem;
}
.doc-title    { font-size: 0.9rem; font-weight: 600; color: #e6edf3 !important; flex: 1; line-height: 1.4; }
.doc-pmid     { font-size: 0.75rem; color: #64748b !important; margin-bottom: 0.5rem; }
.doc-pmid a   { color: #60a5fa !important; text-decoration: none; font-weight: 500; }
.doc-abstract { font-size: 0.82rem; color: #8b949e !important; line-height: 1.65; }
.badge-green  { background:#052e16; color:#86efac !important; border:1px solid #16a34a; border-radius:20px; padding:0.2rem 0.65rem; font-size:0.72rem; font-weight:600; }
.badge-yellow { background:#1c1500; color:#fde68a !important; border:1px solid #ca8a04; border-radius:20px; padding:0.2rem 0.65rem; font-size:0.72rem; font-weight:600; }
.badge-red    { background:#1c0a0a; color:#fca5a5 !important; border:1px solid #dc2626; border-radius:20px; padding:0.2rem 0.65rem; font-size:0.72rem; font-weight:600; }

/* ══ Query chips ═══════════════════════════════════════════════════════════ */
.query-chip {
    display: inline-block;
    background: #0d1f3c;
    border: 1px solid #1e40af;
    color: #93c5fd !important;
    border-radius: 8px;
    padding: 0.35rem 0.9rem;
    font-size: 0.82rem;
    font-weight: 600;
    font-family: 'Courier New', monospace;
}

/* ══ Sidebar items ═════════════════════════════════════════════════════════ */
.sidebar-hist-item {
    background: #161b22;
    border-radius: 8px;
    padding: 0.5rem 0.8rem;
    margin: 0.3rem 0;
    font-size: 0.8rem;
    color: #94a3b8 !important;
    border-left: 3px solid #2563eb;
}

/* ══ Inputs dark ═══════════════════════════════════════════════════════════ */
div[data-testid="stTextArea"] textarea {
    background: #161b22 !important;
    color: #e6edf3 !important;
    border-radius: 10px !important;
    border: 1.5px solid #30363d !important;
    font-size: 0.92rem !important;
}
div[data-testid="stTextArea"] textarea:focus { border-color: #2563eb !important; }
div[data-testid="stTextInput"] input {
    background: #161b22 !important;
    color: #e6edf3 !important;
    border-radius: 10px !important;
    border: 1.5px solid #30363d !important;
}

/* ══ Divers overrides ══════════════════════════════════════════════════════ */
/* Slider : piste grise, poignée bleue, sans bande de fond */
[data-testid="stSlider"] > div > div > div { background: #30363d !important; }
[data-testid="stSlider"] > div > div > div > div { background: #2563eb !important; }
[data-testid="stSlider"] [role="slider"] { background: #2563eb !important; border: 2px solid #60a5fa !important; box-shadow: none !important; }
.stAlert { border-radius: 10px !important; }
hr { border-color: #30363d !important; margin: 1.5rem 0 !important; }
[data-testid="stExpander"]        { background: #161b22 !important; border: 1px solid #30363d !important; border-radius: 10px !important; }
[data-testid="stExpander"] summary { color: #e6edf3 !important; }
</style>
""", unsafe_allow_html=True)

# ─── Initialisation session state ─────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "current_result" not in st.session_state:
    st.session_state.current_result = None
if "last_question" not in st.session_state:
    st.session_state.last_question = ""

# ─── Fonctions utilitaires ────────────────────────────────────────────────────

def get_active_api_key():
    """Retourne la clé API depuis le fichier .env."""
    return OPENROUTER_API_KEY


def translate_to_english(question: str, hint: str, api_key: str) -> str:
    """Génère une requête PubMed précise en anglais (MeSH terms)."""
    try:
        prompt = (
            f"The user asked this medical question: '{question}'\n"
            f"The user suggested this PubMed keyword: '{hint}'\n"
            "Generate a concise and precise PubMed search query in English using MeSH terms when possible "
            "(2-5 words, standard medical terminology). Reply with ONLY the query, nothing else."
        )
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            data=json.dumps({
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": "You are a biomedical search expert."},
                    {"role": "user",   "content": prompt},
                ],
            }),
            timeout=30,
        )
        if resp.status_code == 200:
            data = resp.json()
            content = data["choices"][0]["message"].get("content", "").strip()
            if not content:
                for d in data["choices"][0]["message"].get("reasoning_details", []):
                    if d.get("type") == "reasoning.text":
                        content = d.get("text", "").strip().splitlines()[0]
                        break
            if content:
                return content
    except Exception:
        pass
    return hint


def extract_response_content(response: dict):
    """Extrait le texte et les stats d'usage d'une réponse OpenRouter."""
    if not response:
        return None, {}
    choices = response.get("choices", [])
    if not choices:
        return None, {}
    message = choices[0].get("message", {})
    content = message.get("content", "").strip()
    if not content:
        for detail in message.get("reasoning_details", []):
            if detail.get("type") == "reasoning.text":
                content = detail.get("text", "").strip()
                break
    return content or None, response.get("usage", {})


def score_label(score: float):
    """Returns (prefix, badge_class, label) based on the BGE score."""
    if score < 0.2:
        return "", "badge-green", "Very relevant"
    elif score < 0.35:
        return "", "badge-yellow", "Relevant"
    else:
        return "", "badge-red", "Less relevant"


# ─── Pipeline RAG ─────────────────────────────────────────────────────────────

def run_rag_pipeline(user_question: str, pubmed_hint: str, retmax: int, top_k: int):
    """
    Exécute le pipeline RAG complet et retourne un dict de résultats.
    Affiche la progression avec st.status à chaque étape.
    """
    api_key = get_active_api_key()
    result = {
        "question":        user_question,
        "pubmed_hint":     pubmed_hint,
        "pubmed_query_en": pubmed_hint,
        "question_en":     user_question,
        "articles_count":  0,
        "documents":       [],
        "response_content": None,
        "usage":           {},
        "error":           None,
    }

    # ── Step 0: Translation (parallel) ────────────────────────────────────
    with st.status("Step 1/4 — Translating and optimizing queries…", expanded=True) as status:
        with ThreadPoolExecutor(max_workers=2) as executor:
            f_pubmed = executor.submit(translate_to_english, user_question, pubmed_hint, api_key)
            f_faiss  = executor.submit(translate_to_english, user_question, user_question, api_key)
            pubmed_query_en = f_pubmed.result()
            question_en     = f_faiss.result()
        result["pubmed_query_en"] = pubmed_query_en
        result["question_en"]     = question_en

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**PubMed Query:** `{pubmed_hint}` → **`{pubmed_query_en}`**")
        with col2:
            st.markdown(f"**FAISS Question:** `{question_en}`")
        status.update(label="Step 1/4 — Translation complete", state="complete", expanded=False)

    # ── Step 1: PubMed Download ──────────────────────────────────────────────
    with st.status(f"Step 2/4 — Downloading {retmax} articles from PubMed…", expanded=True) as status:
        articles = download_articles(pubmed_query_en, retmax=retmax)
        if not articles:
            result["error"] = f"No articles found for '{pubmed_query_en}' on PubMed."
            status.update(label="No articles found", state="error", expanded=True)
            return result

        result["articles_count"] = len(articles)
        st.markdown(f"**{len(articles)}** articles downloaded from PubMed")
        status.update(label=f"Step 2/4 — {len(articles)} articles downloaded", state="complete", expanded=False)

    # ── Step 2: FAISS Vector Index ────────────────────────────────────────────
    with st.status("Step 3/4 — Building FAISS vector index…", expanded=True) as status:
        documents    = load_documents_from_articles(articles)
        vector_store = create_vector_store(documents)
        st.markdown(f"FAISS index built with **{len(documents)}** documents")
        status.update(label=f"Step 3/4 — Index built ({len(documents)} documents)", state="complete", expanded=False)

    # ── Step 3: Semantic Search ───────────────────────────────────────────────
    with st.status(f"Step 4/4 — Searching for the {top_k} most relevant documents…", expanded=True) as status:
        results_with_scores = search_with_score(
            vector_store, question_en, k=top_k * 2, score_threshold=SCORE_THRESHOLD, min_results=top_k
        )
        top_results = results_with_scores[:top_k]
        result["documents"] = top_results
        st.markdown(f"**{len(top_results)}** relevant documents selected")
        status.update(label=f"Step 4/4 — {len(top_results)} documents found", state="complete", expanded=False)

    return result


# ─── Fonctions d'affichage réutilisables ──────────────────────────────────────

def _show_metrics(result):
    """Affiche les métriques en haut des résultats."""
    usage = result.get("usage", {})
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("📥 Articles téléchargés",  result["articles_count"])
    m2.metric("📄 Documents analysés",    len(result["documents"]))
    m3.metric("🔤 Prompt (tokens)",       usage.get("prompt_tokens",     "—"))
    m4.metric("💬 Réponse (tokens)",      usage.get("completion_tokens", "—"))
    m5.metric("📦 Total (tokens)",        usage.get("total_tokens",      "—"))


def _show_queries(result):
    """Affiche le détail des requêtes générées."""
    with st.expander("🔍 Détail des requêtes générées", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Requête PubMed (anglais MeSH) :**")
            st.code(result["pubmed_query_en"], language=None)
        with c2:
            st.markdown("**Question traduite pour FAISS :**")
            st.code(result["question_en"], language=None)


def _show_documents(result):
    """Display relevant PubMed documents."""
    st.markdown(f"### Relevant PubMed Documents ({len(result['documents'])} found)")
    for i, (doc, score) in enumerate(result["documents"], 1):
        _, badge_cls, label = score_label(score)
        pmid     = doc.metadata.get("source", "N/A")
        title    = html_module.escape(doc.metadata.get("title", "No title"))
        abstract = doc.page_content
        abstract_preview = html_module.escape(abstract[:380]) + ("…" if len(abstract) > 380 else "")
        st.markdown(f"""
        <div class="doc-card">
            <div class="doc-card-header">
                <span class="doc-number">#{i}</span>
                <span class="doc-title">{title}</span>
                <span class="{badge_cls}">{label}</span>
            </div>
            <div class="doc-pmid">PMID: <a href="https://pubmed.ncbi.nlm.nih.gov/{pmid}/" target="_blank">{pmid}</a> · BGE Score: <strong>{score:.3f}</strong></div>
            <div class="doc-abstract">{abstract_preview}</div>
        </div>
        """, unsafe_allow_html=True)
        if len(abstract) > 380:
            with st.expander(f"Full Abstract #{i}", expanded=False):
                st.markdown(abstract)


def _show_disclaimer_download(result):
    """Affiche le disclaimer médical et le bouton de téléchargement."""
    st.warning(
        "Important reminder: This response is automatically generated from PubMed articles. "
        "It does not replace professional medical advice. "
        "Verify each reference directly on [PubMed](https://pubmed.ncbi.nlm.nih.gov/) "
        "before any clinical use."
    )
    export_text = (
        f"QUESTION: {result['question']}\n"
        f"PubMed Query: {result['pubmed_query_en']}\n"
        f"{'─'*60}\n\nDOCUMENTS FOUND ({len(result['documents'])}):\n"
    )
    for i, (doc, score) in enumerate(result["documents"], 1):
        _, _, label = score_label(score)
        export_text += (
            f"\n[{i}] PMID {doc.metadata.get('source','N/A')} — {label} (score: {score:.3f})\n"
            f"    Title: {doc.metadata.get('title','')}\n"
            f"    Abstract: {doc.page_content[:300]}…\n"
        )
    export_text += (
        f"\n{'─'*60}\n\nAI RESPONSE:\n{result.get('response_content','')}\n\n"
        f"{'─'*60}\n"
        f"DISCLAIMER: Response automatically generated. Does not replace medical advice.\n"
    )
    st.download_button(
        label="Download Report (TXT)",
        data=export_text,
        file_name=f"medical_rag_report_{result['pubmed_query_en'].replace(' ', '_')}.txt",
        mime="text/plain",
        use_container_width=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ─── INTERFACE STREAMLIT ──────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

# ══ Sidebar ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding:1.2rem 0 0.5rem 0; text-align:center;">
        <div style="font-size:2.4rem;">🏥</div>
        <div style="font-size:1rem; font-weight:700; color:#f1f5f9; margin-top:0.3rem;">MedRAG</div>
        <div style="font-size:0.72rem; color:#64748b; margin-top:0.1rem;">Intelligent Medical Assistant</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown('<p style="font-size:0.75rem; font-weight:600; text-transform:uppercase; letter-spacing:0.08em; color:#475569; margin-bottom:0.8rem;">Parameters</p>', unsafe_allow_html=True)
    retmax = st.slider("PubMed Articles", min_value=10, max_value=200, value=DEFAULT_RETMAX, step=10,
                       help="Number of articles downloaded from PubMed")
    top_k  = st.slider("Top-K Documents", min_value=2, max_value=10, value=DEFAULT_TOP_K, step=1,
                       help="Documents sent to the LLM to generate the answer")

    st.divider()

    st.markdown('<p style="font-size:0.75rem; font-weight:600; text-transform:uppercase; letter-spacing:0.08em; color:#475569; margin-bottom:0.6rem;">Models</p>', unsafe_allow_html=True)
    st.markdown(f'<div style="background:#1e293b; border-radius:8px; padding:0.6rem 0.8rem; font-size:0.75rem; color:#94a3b8; margin-bottom:0.4rem;"><span style="color:#60a5fa; font-weight:600;">LLM</span><br>{MODEL_NAME.split("/")[-1]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="background:#1e293b; border-radius:8px; padding:0.6rem 0.8rem; font-size:0.75rem; color:#94a3b8;"><span style="color:#60a5fa; font-weight:600;">Embeddings</span><br>{EMBEDDING_MODEL}</div>', unsafe_allow_html=True)

pubmed_hint = ""


# ══ Header ════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="pro-header">
    <div class="pro-header-text">
        <h1>Medical RAG Assistant</h1>
        <p>PubMed Search · FAISS Semantic Analysis · Evidence-Based AI Responses</p>
        <div style="display:flex; gap:0.5rem; margin-top:0.9rem; flex-wrap:wrap;">
            <span class="pro-badge">PubMed NCBI</span>
            <span class="pro-badge">FAISS + BGE</span>
            <span class="pro-badge">OpenRouter AI</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ══ Disclaimer ════════════════════════════════════════════════════════════════
st.markdown("""
<div class="pro-disclaimer">
    <div class="pro-disclaimer-title">Medical Disclaimer — For Educational Use Only</div>
    <ul>
        <li>Responses are generated by AI from scientific articles and <strong>do not replace professional medical advice</strong>.</li>
        <li>Consult a healthcare professional before any treatment. Verify dosages and contraindications.</li>
        <li>Emergency in Morocco: <strong>150 (SAMU)</strong> · <strong>15 (Medical Emergencies)</strong> · <strong>177 (Civil Protection)</strong></li>
    </ul>
</div>
""", unsafe_allow_html=True)

user_question = st.chat_input("Ask your medical question here…", key="question_input")

# ══ Exécution du pipeline ══════════════════════════════════════════════════════
if user_question:
    if not (get_active_api_key() and get_active_api_key() != "<OPENROUTER_API_KEY>"):
        st.error("❌ OpenRouter API key not found. Check your `.env` file.")
    else:
        hint = user_question.strip()

        st.markdown("---")
        st.markdown('<div class="section-card"><div class="section-title">RAG Pipeline Running…</div></div>', unsafe_allow_html=True)

        result = run_rag_pipeline(
            user_question=user_question.strip(),
            pubmed_hint=hint,
            retmax=retmax,
            top_k=top_k,
        )

        if result.get("error"):
            st.error(result['error'])
        else:
            # ── Métriques ────────────────────────────────────────────────────
            st.markdown("---")
            st.markdown(f"""
            <div class="section-card">
                <div class="section-title">Search Summary</div>
                <div class="metric-grid">
                    <div class="metric-pill"><div class="mp-value">{result["articles_count"]}</div><div class="mp-label">PubMed Articles</div></div>
                    <div class="metric-pill"><div class="mp-value">{len(result["documents"])}</div><div class="mp-label">Selected Docs</div></div>
                    <div class="metric-pill"><div class="mp-value">{retmax}</div><div class="mp-label">Requested</div></div>
                    <div class="metric-pill"><div class="mp-value">{top_k}</div><div class="mp-label">FAISS Top-K</div></div>
                    <div class="metric-pill"><div class="mp-value">{SCORE_THRESHOLD}</div><div class="mp-label">Score Threshold</div></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Requêtes générées ─────────────────────────────────────────────
            with st.expander("Generated Queries", expanded=False):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**PubMed Query (MeSH):**")
                    st.markdown(f'<span class="query-chip">{result["pubmed_query_en"]}</span>', unsafe_allow_html=True)
                with c2:
                    st.markdown("**FAISS Question:**")
                    st.markdown(f'<span class="query-chip">{result["question_en"]}</span>', unsafe_allow_html=True)

            # ── PubMed Documents ──────────────────────────────────────────────
            st.markdown("---")
            st.markdown(f'<div class="section-card"><div class="section-title">Selected PubMed Documents ({len(result["documents"])} / {top_k})</div></div>', unsafe_allow_html=True)

            for i, (doc, score) in enumerate(result["documents"], 1):
                _, badge_cls, label = score_label(score)
                pmid     = doc.metadata.get("source", "N/A")
                title    = html_module.escape(doc.metadata.get("title", "No title"))
                abstract = doc.page_content
                abstract_preview = html_module.escape(abstract[:380]) + ("…" if len(abstract) > 380 else "")

                st.markdown(f"""
                <div class="doc-card">
                    <div class="doc-card-header">
                        <span class="doc-number">#{i}</span>
                        <span class="doc-title">{title}</span>
                        <span class="{badge_cls}">{label}</span>
                    </div>
                    <div class="doc-pmid">
                        PMID: <a href="https://pubmed.ncbi.nlm.nih.gov/{pmid}/" target="_blank">{pmid}</a>
                        &nbsp;·&nbsp; BGE Score: <strong>{score:.3f}</strong>
                    </div>
                    <div class="doc-abstract">{abstract_preview}</div>
                </div>
                """, unsafe_allow_html=True)

                if len(abstract) > 380:
                    with st.expander(f"Read full abstract — #{i}", expanded=False):
                        st.markdown(abstract)

            # ── AI Response (streaming) ───────────────────────────────────────
            st.markdown("---")
            st.markdown('<div class="section-card"><div class="section-title">AI Model Response</div></div>', unsafe_allow_html=True)

            docs_only = [doc for doc, _ in result["documents"]]

            # Indicateur de génération : spinner affiché jusqu'au 1er token
            spinner_slot = st.empty()
            spinner_slot.markdown("""
            <div style="
                background:#161b22;
                border:1px solid #30363d;
                border-left:4px solid #2563eb;
                border-radius:12px;
                padding:1.4rem 1.8rem;
                display:flex;
                align-items:center;
                gap:1rem;
                color:#93c5fd;
                font-size:0.92rem;
                font-weight:500;
            ">
                <div style="
                    width:18px; height:18px;
                    border:3px solid #1e40af;
                    border-top:3px solid #60a5fa;
                    border-radius:50%;
                    animation:spin 0.8s linear infinite;
                    flex-shrink:0;
                "></div>
                Generating response…
            </div>
            <style>
            @keyframes spin { to { transform: rotate(360deg); } }
            </style>
            """, unsafe_allow_html=True)

            def stream_with_clear():
                first = True
                for chunk in stream_openrouter(docs_only, user_question.strip()):
                    if first:
                        spinner_slot.empty()
                        first = False
                    yield chunk

            with st.container(border=True):
                full_response = st.write_stream(stream_with_clear())

            result["response_content"] = full_response
            st.session_state.current_result = result
            st.session_state.last_question  = user_question.strip()
            st.session_state.history.append({"question": user_question.strip(), "result": result})

            # ── Disclaimer + export ────────────────────────────────────────────
            st.markdown("---")
            _show_disclaimer_download(result)


# ══ Résultat précédent (rechargement sans clic) ════════════════════════════════
elif st.session_state.get("current_result"):
    result = st.session_state.current_result
    if result.get("error"):
        st.error(f"❌ {result['error']}")
    else:
        st.markdown("---")
        usage = result.get("usage", {})
        st.markdown(f"""
        <div class="section-card">
            <div class="section-title">📊 Search Summary</div>
            <div class="metric-grid">
                <div class="metric-pill"><div class="mp-value">{result["articles_count"]}</div><div class="mp-label">PubMed Articles</div></div>
                <div class="metric-pill"><div class="mp-value">{len(result["documents"])}</div><div class="mp-label">Selected Docs</div></div>
                <div class="metric-pill"><div class="mp-value">{usage.get("prompt_tokens","—")}</div><div class="mp-label">Prompt Tokens</div></div>
                <div class="metric-pill"><div class="mp-value">{usage.get("completion_tokens","—")}</div><div class="mp-label">Response Tokens</div></div>
                <div class="metric-pill"><div class="mp-value">{usage.get("total_tokens","—")}</div><div class="mp-label">Total Tokens</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("Generated Queries", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**PubMed MeSH:**")
                st.markdown(f'<span class="query-chip">{result["pubmed_query_en"]}</span>', unsafe_allow_html=True)
            with c2:
                st.markdown("**FAISS:**")
                st.markdown(f'<span class="query-chip">{result["question_en"]}</span>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(f'<div class="section-card"><div class="section-title">PubMed Documents ({len(result["documents"])} found)</div></div>', unsafe_allow_html=True)

        for i, (doc, score) in enumerate(result["documents"], 1):
            _, badge_cls, label = score_label(score)
            pmid     = doc.metadata.get("source", "N/A")
            title    = html_module.escape(doc.metadata.get("title", "No title"))
            abstract = doc.page_content
            abstract_preview = html_module.escape(abstract[:380]) + ("…" if len(abstract) > 380 else "")
            st.markdown(f"""
            <div class="doc-card">
                <div class="doc-card-header">
                    <span class="doc-number">#{i}</span>
                    <span class="doc-title">{title}</span>
                    <span class="{badge_cls}">{label}</span>
                </div>
                <div class="doc-pmid">PMID: <a href="https://pubmed.ncbi.nlm.nih.gov/{pmid}/" target="_blank">{pmid}</a> · BGE Score: <strong>{score:.3f}</strong></div>
                <div class="doc-abstract">{abstract_preview}</div>
            </div>
            """, unsafe_allow_html=True)
            if len(abstract) > 380:
                with st.expander(f"Full Abstract #{i}", expanded=False):
                    st.markdown(abstract)

        st.markdown("---")
        st.markdown('<div class="section-card"><div class="section-title">AI Model Response</div></div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown(result["response_content"])

        st.markdown("---")
        _show_disclaimer_download(result)