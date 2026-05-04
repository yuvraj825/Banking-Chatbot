"""
pipeline/rag_chain.py
Conversational RAG chain built with LCEL (LangChain Expression Language).

Compatible with langchain >= 1.0, langchain-community >= 0.4

LLM: Groq API — Llama 3.1 8B (free, fast, no credit card)
     Fallback: HuggingFace Inference API (Zephyr-7B-Beta)
Memory: InMemoryChatMessageHistory via RunnableWithMessageHistory
"""

import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# Load .env for local dev
try:
    from dotenv import load_dotenv
    load_dotenv(_PROJECT_ROOT / ".env", override=True)
except ImportError:
    pass

# Streamlit Cloud: inject secrets before anything else loads
try:
    import streamlit as st
    if hasattr(st, "secrets"):
        for _key in ["GROQ_API_KEY", "HUGGINGFACEHUB_API_TOKEN"]:
            if _key in st.secrets and not os.getenv(_key):
                os.environ[_key] = st.secrets[_key]
except Exception:
    pass

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from retrieval.retriever import get_retriever
from ingestion.embedder import index_exists

# In-memory session store
_session_store: dict = {}
SESSION_ID = "banking_chat"
MAX_HISTORY = 10


def _get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in _session_store:
        _session_store[session_id] = InMemoryChatMessageHistory()
    history = _session_store[session_id]
    # Trim to last MAX_HISTORY messages to control context size
    if len(history.messages) > MAX_HISTORY:
        history.messages = history.messages[-MAX_HISTORY:]
    return history


# LLM builders

def _build_groq_llm():
    from langchain_groq import ChatGroq
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set.")
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=api_key,
        temperature=0.2,
        max_tokens=512,
    )
    print("[RAGChain] Using Groq — Llama 3.1 8B Instant.")
    return llm


def _build_hf_llm():
    from langchain_huggingface import HuggingFaceEndpoint
    api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
    if not api_token:
        raise EnvironmentError("HUGGINGFACEHUB_API_TOKEN not set.")
    llm = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        huggingfacehub_api_token=api_token,
        task="text-generation",
        temperature=0.2,
        max_new_tokens=512,
        repetition_penalty=1.1,
    )
    print("[RAGChain] Using HuggingFace — Zephyr-7B-Beta.")
    return llm


def _build_llm():
    groq_key = os.getenv("GROQ_API_KEY", "")
    hf_key = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")

    if groq_key:
        try:
            return _build_groq_llm()
        except Exception as e:
            print(f"[RAGChain] Groq failed ({e}), trying HuggingFace...")

    if hf_key:
        return _build_hf_llm()

    raise EnvironmentError(
        "No LLM API key found!\n\n"
        "RECOMMENDED (Free, fast, no credit card):\n"
        "  1. Go to https://console.groq.com and sign up\n"
        "  2. Create an API key\n"
        "  3. Add to .env: GROQ_API_KEY=your_key_here\n\n"
        "ALTERNATIVE (HuggingFace free):\n"
        "  1. Go to https://huggingface.co/settings/tokens\n"
        "  2. Create a READ token\n"
        "  3. Add to .env: HUGGINGFACEHUB_API_TOKEN=your_token"
    )


# Prompt

SYSTEM_PROMPT = """You are a knowledgeable and professional banking assistant for Indian customers.
You have access to information from RBI circulars, bank FAQs (SBI, HDFC, ICICI), and curated financial data.

STRICT RULES:
1. Answer ONLY based on the provided context. Do not use outside knowledge.
2. If the answer is not in the context, say:
   "I don't have that information in my current knowledge base. Please check rbi.org.in or your bank's official portal."
3. Be concise, accurate, and helpful.
4. When citing figures (rates, fees, limits), mention the source document.
5. Never give personalised financial advice. Recommend consulting a certified financial advisor.
6. Keep responses under 300 words unless the question requires more detail.

CONTEXT:
{context}"""


def _build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])


def _format_docs(docs: list) -> str:
    if not docs:
        return "No relevant documents found."
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        doc_type = doc.metadata.get("doc_type", "doc")
        parts.append(f"[{i}] [{doc_type.upper()}] {source}\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(parts)


# Chain builder

def build_rag_chain():
    """
    Build a stateful LCEL RAG chain with conversation history.
    Returns a bundle dict with 'chain' and 'state' for source attribution.
    """
    if not index_exists():
        raise FileNotFoundError(
            "FAISS index not found. Build the knowledge base first:\n"
            "  python pipeline/updater.py --run-now"
        )

    llm = _build_llm()
    retriever = get_retriever(search_type="mmr", k=4)
    prompt = _build_prompt()

    # Mutable state to pass retrieved docs out of the chain for source display
    _chain_state: dict = {"last_docs": []}

    def retrieve_and_store(inputs: dict) -> dict:
        question = inputs["question"]
        docs = retriever.invoke(question)
        _chain_state["last_docs"] = docs
        return {
            "context": _format_docs(docs),
            "question": question,
        }

    core_chain = (
        RunnableLambda(retrieve_and_store)
        | (lambda x: {"context": x["context"], "question": x["question"], "history": x.get("history", [])})
        | prompt
        | llm
        | StrOutputParser()
    )

    chain_with_history = RunnableWithMessageHistory(
        core_chain,
        get_session_history=_get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )

    return {"chain": chain_with_history, "state": _chain_state}


def query_chain(chain_bundle: dict, question: str) -> dict:
    """
    Run a query and return answer + sources.

    Args:
        chain_bundle: Output of build_rag_chain().
        question: User's natural language question.

    Returns:
        {"answer": str, "sources": list[dict]}
    """
    chain = chain_bundle["chain"]
    state = chain_bundle["state"]

    answer = chain.invoke(
        {"question": question},
        config={"configurable": {"session_id": SESSION_ID}},
    )

    # Clean LLM echo artefacts
    for marker in ["Answer:", "Assistant:", "Human:", "\nQuestion:"]:
        if marker in answer:
            answer = answer.split(marker)[-1].strip()

    seen, sources = set(), []
    for doc in state.get("last_docs", []):
        src = doc.metadata.get("source", "Unknown")
        if src not in seen:
            seen.add(src)
            sources.append({
                "content": doc.page_content[:300],
                "source": src,
                "doc_type": doc.metadata.get("doc_type", "unknown"),
            })

    return {"answer": answer, "sources": sources}


# Singleton

_chain_instance = None


def get_chain() -> dict:
    global _chain_instance
    if _chain_instance is None:
        _chain_instance = build_rag_chain()
    return _chain_instance


def reset_chain() -> None:
    global _chain_instance, _session_store
    _chain_instance = None
    _session_store.clear()
    print("[RAGChain] Chain and history reset.")


if __name__ == "__main__":
    bundle = get_chain()
    response = query_chain(bundle, "What is the current repo rate set by RBI?")
    print("\nAnswer:", response["answer"])
    print("\nSources:")
    for s in response["sources"]:
        print(f"  [{s['doc_type']}] {s['source']}")
