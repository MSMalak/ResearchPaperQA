import os
import shutil
from pathlib import Path

import streamlit as st
from rag_chatbot.generator import RAGChatbot

st.set_page_config(page_title="ResearchPaperQA", page_icon="📄", layout="wide")

# ----------------------------
# Persistent dirs for UI demo
# ----------------------------
UI_WORKDIR = Path(".ui_cache")
UI_DOCS_DIR = UI_WORKDIR / "papers"
UI_INDEX_DIR = UI_WORKDIR / "index"


def ensure_dirs():
    UI_WORKDIR.mkdir(parents=True, exist_ok=True)
    UI_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    UI_INDEX_DIR.mkdir(parents=True, exist_ok=True)


def save_uploads(uploaded_files) -> int:
    """Save uploaded PDFs into UI_DOCS_DIR (persistent). Returns count saved."""
    ensure_dirs()
    n = 0
    for f in uploaded_files:
        if not f.name.lower().endswith(".pdf"):
            continue
        out = UI_DOCS_DIR / f.name
        with open(out, "wb") as w:
            w.write(f.getbuffer())
        n += 1
    return n


def clear_index():
    """Remove FAISS index directory contents."""
    if UI_INDEX_DIR.exists():
        shutil.rmtree(UI_INDEX_DIR, ignore_errors=True)
    UI_INDEX_DIR.mkdir(parents=True, exist_ok=True)


def clear_papers():
    """Remove uploaded papers directory contents."""
    if UI_DOCS_DIR.exists():
        shutil.rmtree(UI_DOCS_DIR, ignore_errors=True)
    UI_DOCS_DIR.mkdir(parents=True, exist_ok=True)


@st.cache_resource(show_spinner=False)
def build_bot_cached(documents_dir: str, index_dir: str, generator: str) -> RAGChatbot:
    """
    Cached bot builder. Cache key depends on (documents_dir, index_dir, generator).
    If you rebuild the index, call st.cache_resource.clear() so it re-instantiates.
    """
    bot = RAGChatbot(
        documents_path=documents_dir,
        index_path=index_dir,
        generator_type=generator,
    )
    # Force recreate is handled outside (by clearing index dir + clearing cache)
    bot.setup(force_recreate_index=False)
    return bot


def init_state():
    st.session_state.setdefault("ready", False)
    st.session_state.setdefault("generator", "local")
    st.session_state.setdefault("last_answer", None)


init_state()
ensure_dirs()

st.title("📄 ResearchPaperQA")
st.caption("RAG over PDFs (Chunking → Embeddings → FAISS → Retrieval → Answer)")

# ----------------------------
# Sidebar controls
# ----------------------------
with st.sidebar:
    st.header("Settings")

    generator = st.selectbox("Generator backend", ["local", "openai"], index=0)
    st.session_state.generator = generator

    st.markdown("---")
    st.subheader("OpenAI (optional)")
    openai_key = st.text_input("OPENAI_API_KEY", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key

    st.markdown("---")
    st.subheader("Papers")

    uploaded = st.file_uploader("Upload PDF papers", type=["pdf"], accept_multiple_files=True)

    col_a, col_b = st.columns(2)
    with col_a:
        build_btn = st.button("📚 Build / Load index", type="primary")
    with col_b:
        rebuild_btn = st.button("♻️ Rebuild index", help="Clears the FAISS index and rebuilds it from current PDFs.")

    st.markdown("---")
    st.subheader("Maintenance")
    m1, m2 = st.columns(2)
    with m1:
        clear_papers_btn = st.button("🧹 Clear papers")
    with m2:
        clear_index_btn = st.button("🗑️ Clear index")

# ----------------------------
# Maintenance actions
# ----------------------------
if clear_papers_btn:
    clear_papers()
    st.session_state.ready = False
    st.cache_resource.clear()
    st.success("Cleared uploaded papers.")

if clear_index_btn:
    clear_index()
    st.session_state.ready = False
    st.cache_resource.clear()
    st.success("Cleared FAISS index.")

# ----------------------------
# Main layout
# ----------------------------
left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("1) Papers in workspace")
    pdfs = sorted([p.name for p in UI_DOCS_DIR.glob("*.pdf")])
    if pdfs:
        st.write(f"Stored PDFs: **{len(pdfs)}**")
        for name in pdfs[:15]:
            st.write(f"- {name}")
        if len(pdfs) > 15:
            st.write(f"...and {len(pdfs) - 15} more")
    else:
        st.info("No PDFs stored yet. Upload from the sidebar.")

with right:
    st.subheader("2) Index status")
    faiss_file = UI_INDEX_DIR / "index.faiss"
    pkl_file = UI_INDEX_DIR / "index.pkl"
    if st.session_state.ready and faiss_file.exists() and pkl_file.exists():
        st.success("Index ready ✅ You can ask questions below.")
        st.write(f"Docs dir: `{UI_DOCS_DIR}`")
        st.write(f"Index dir: `{UI_INDEX_DIR}`")
    else:
        st.warning("Index not built yet.")

# ----------------------------
# Build / Load index logic
# ----------------------------
def build_or_load(force_rebuild: bool):
    if uploaded:
        n_saved = save_uploads(uploaded)
        if n_saved == 0:
            st.error("No valid PDF uploaded.")
            return

    # If forcing rebuild: clear index dir and clear Streamlit cache (so bot rebuilds)
    if force_rebuild:
        clear_index()
        st.cache_resource.clear()

    # If no PDFs exist, cannot build
    pdfs_now = list(UI_DOCS_DIR.glob("*.pdf"))
    if not pdfs_now:
        st.error("No PDFs available. Upload at least one PDF first.")
        return

    with st.spinner("Initializing chatbot + building/loading FAISS index..."):
        # build_bot_cached will load existing index if present,
        # or create it depending on your internal logic.
        # If you want guaranteed rebuild, we already cleared index + cleared cache.
        bot = build_bot_cached(str(UI_DOCS_DIR), str(UI_INDEX_DIR), st.session_state.generator)
        st.session_state.ready = True
        st.session_state.last_answer = None

    st.success("Done ✅")


if build_btn:
    build_or_load(force_rebuild=False)

if rebuild_btn:
    build_or_load(force_rebuild=True)

st.markdown("---")
st.subheader("3) Ask a question")

# ----------------------------
# Ask form: Enter OR click Ask
# ----------------------------
with st.form("ask_form", clear_on_submit=False):
    query = st.text_input(
        "Your question",
        placeholder="e.g., What is the main contribution of the paper?",
    )
    submitted = st.form_submit_button("🤖 Ask")

if submitted:
    if not st.session_state.ready:
        st.error("Build / Load the index first (sidebar).")
    elif not query.strip():
        st.error("Please type a question.")
    else:
        bot = build_bot_cached(str(UI_DOCS_DIR), str(UI_INDEX_DIR), st.session_state.generator)

        with st.spinner("Answering..."):
            out = bot.ask(query)

        st.session_state.last_answer = out

# ----------------------------
# Display last answer (persists across reruns)
# ----------------------------
out = st.session_state.last_answer
if out:
    answer = out.get("answer", "")
    sources = out.get("source_documents", []) or []
    model = out.get("model", out.get("generator", "unknown"))

    st.markdown("### Answer")
    st.write(answer)

    st.markdown("### Sources (top retrieved chunks)")
    st.caption(f"Generator: **{model}** • Retrieved chunks: **{len(sources)}**")

    if not sources:
        st.info("No source chunks returned.")
    else:
        for i, doc in enumerate(sources[:5], start=1):
            meta = getattr(doc, "metadata", {}) or {}
            src = meta.get("source_file", meta.get("source", "unknown"))
            page = meta.get("page", meta.get("page_number", ""))
            header = f"**#{i}** — {src}"
            if page != "":
                header += f" (page {page})"
            with st.expander(header, expanded=(i == 1)):
                st.write(doc.page_content)

