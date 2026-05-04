"""
app/streamlit_app.py
Streamlit chat interface for the RAG Banking Chatbot.

Features:
  - Conversational chat UI with message history
  - Source citations shown below each answer
  - Sidebar: manual "Update Knowledge Base" button
  - Sidebar: last update timestamp + index stats
  - Background scheduler for daily auto-updates
"""

import sys
import os
from pathlib import Path
import streamlit as st

_APP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_APP_ROOT))

# Load .env for local development
try:
    from dotenv import load_dotenv
    load_dotenv(_APP_ROOT / ".env", override=True)
except ImportError:
    pass

# Streamlit Cloud: inject st.secrets into env vars before any module loads
try:
    if hasattr(st, "secrets"):
        for _k in ["GROQ_API_KEY", "HUGGINGFACEHUB_API_TOKEN"]:
            if _k in st.secrets and not os.getenv(_k):
                os.environ[_k] = st.secrets[_k]
except Exception:
    pass

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Banking RAG Chatbot",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

from pipeline.rag_chain import get_chain, query_chain, reset_chain
from pipeline.updater import run_ingestion_pipeline, load_run_log, start_scheduler
from ingestion.embedder import index_exists, get_index_stats

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .source-box {
        background: #f0f4f8;
        border-left: 3px solid #1f77b4;
        border-radius: 4px;
        padding: 8px 12px;
        font-size: 0.82em;
        color: #444;
        margin-top: 4px;
    }
    .source-title { font-weight: 600; color: #1f77b4; }
    .stChatMessage { margin-bottom: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ── Session state init ────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chain_bundle" not in st.session_state:
    st.session_state.chain_bundle = None
if "scheduler_started" not in st.session_state:
    st.session_state.scheduler_started = False

# ── Start background scheduler once per session ───────────────────────────────
if not st.session_state.scheduler_started:
    try:
        start_scheduler(hour=6, minute=0, blocking=False)
        st.session_state.scheduler_started = True
    except Exception:
        pass  # Scheduler already running or env doesn't support threads

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏦 Banking RAG Chatbot")
    st.markdown("Powered by RBI circulars, SBI, HDFC & ICICI FAQs.")
    st.divider()

    # Knowledge base stats
    st.subheader("📚 Knowledge Base")
    run_log = load_run_log()
    stats = get_index_stats()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Chunks", stats.get("doc_count", 0))
    with col2:
        last_run = run_log.get("last_run", "Never")
        if last_run != "Never":
            last_run = last_run[:10]
        st.metric("Last Updated", last_run)

    new_docs = run_log.get("new_documents_ingested", 0)
    if new_docs:
        st.caption(f"Last run added {new_docs} new document(s).")

    st.divider()

    # Manual update button
    st.subheader("🔄 Manual Update")
    if st.button("Update Knowledge Base", use_container_width=True, type="primary"):
        with st.spinner("Scraping and indexing new documents..."):
            result = run_ingestion_pipeline(max_rbi_pdfs=5)
        reset_chain()
        st.session_state.chain_bundle = None
        if result["new_documents"] > 0:
            st.success(
                f"Added {result['new_documents']} new doc(s), "
                f"{result['new_chunks']} chunk(s)."
            )
        else:
            st.info("Knowledge base is already up to date.")

    st.divider()

    # Clear chat
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chain_bundle = None
        reset_chain()
        st.rerun()

    st.divider()
    st.caption(
        "Data sources: RBI.org.in · SBI · HDFC · ICICI\n"
        "Model: Llama 3.1 8B (Groq) · Embeddings: all-MiniLM-L6-v2"
    )

# ── Main chat area ────────────────────────────────────────────────────────────
st.header("💬 Ask me anything about banking in India")
st.caption(
    "I can help with RBI policies, loan FAQs, interest rates, and more. "
    "Answers are grounded in official documents — I'll tell you when I don't know."
)

# Guard: no index yet
if not index_exists():
    st.warning(
        "⚠️ No knowledge base found. Click **Update Knowledge Base** in the sidebar "
        "to ingest documents before chatting."
    )
    st.stop()

# Load chain bundle (cached in session state)
if st.session_state.chain_bundle is None:
    with st.spinner("Loading AI model..."):
        try:
            st.session_state.chain_bundle = get_chain()
        except EnvironmentError as e:
            st.error(f"❌ {e}")
            st.info("👉 Get a FREE Groq key at https://console.groq.com → add GROQ_API_KEY to your .env or Streamlit secrets.")
            st.stop()
        except Exception as e:
            st.error(f"❌ Failed to load model: {e}")
            st.stop()

# Display message history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📎 Sources", expanded=False):
                for src in msg["sources"]:
                    st.markdown(
                        f'<div class="source-box">'
                        f'<span class="source-title">[{src["doc_type"].upper()}]</span> '
                        f'{src["source"]}<br>'
                        f'<small>{src["content"][:200]}...</small>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

# Chat input
if prompt := st.chat_input("Ask about RBI policies, loans, interest rates..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        answer = ""
        sources = []
        with st.spinner("Thinking..."):
            try:
                result = query_chain(st.session_state.chain_bundle, prompt)
                answer = result["answer"]
                sources = result["sources"]
            except Exception as e:
                answer = f"Sorry, I encountered an error: {e}"
                sources = []

        st.markdown(answer)

        if sources:
            with st.expander("📎 Sources", expanded=False):
                for src in sources:
                    st.markdown(
                        f'<div class="source-box">'
                        f'<span class="source-title">[{src["doc_type"].upper()}]</span> '
                        f'{src["source"]}<br>'
                        f'<small>{src["content"][:200]}...</small>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
