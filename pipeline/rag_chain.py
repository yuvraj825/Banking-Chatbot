"""
pipeline/rag_chain.py
Conversational RAG chain using LangChain LCEL.
Manages conversation history manually to avoid version-specific import issues.
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

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage

from retrieval.retriever import get_retriever
from ingestion.embedder import index_exists

# In-memory conversation history — no external dependency
_conversation_history = []
MAX_HISTORY = 10

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
        "  3. Add to Streamlit secrets: GROQ_API_KEY = \"your_key_here\"\n\n"
        "ALTERNATIVE (HuggingFace free):\n"
        "  1. Go to https://huggingface.co/settings/tokens\n"
        "  2. Create a READ token\n"
        "  3. Add to Streamlit secrets: HUGGINGFACEHUB_API_TOKEN = \"your_token\""
    )


def _format_docs(docs):
    if not docs:
        return "No relevant documents found."
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        doc_type = doc.metadata.get("doc_type", "doc")
        parts.append(f"[{i}] [{doc_type.upper()}] {source}\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(parts)


def _build_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])


def build_rag_chain():
    if not index_exists():
        raise FileNotFoundError(
            "FAISS index not found. Build the knowledge base first:\n"
            "  Click 'Update Knowledge Base' in the sidebar."
        )

    llm = _build_llm()
    retriever = get_retriever(search_type="mmr", k=4)
    prompt = _build_prompt()

    # Mutable state to expose retrieved docs for source attribution
    _state = {"last_docs": []}

    def retrieve_and_format(inputs):
        docs = retriever.invoke(inputs["question"])
        _state["last_docs"] = docs
        return {
            "context": _format_docs(docs),
            "question": inputs["question"],
            "history": inputs.get("history", []),
        }

    chain = (
        RunnableLambda(retrieve_and_format)
        | prompt
        | llm
        | StrOutputParser()
    )

    print("[RAGChain] Chain assembled successfully.")
    return {"chain": chain, "state": _state}


def query_chain(chain_bundle, question):
    global _conversation_history

    chain = chain_bundle["chain"]
    state = chain_bundle["state"]

    # Trim history to last MAX_HISTORY messages
    history = _conversation_history[-MAX_HISTORY:]

    answer = chain.invoke({
        "question": question,
        "history": history,
    })

    # Clean LLM echo artefacts
    for marker in ["Answer:", "Assistant:", "Human:", "\nQuestion:"]:
        if marker in answer:
            answer = answer.split(marker)[-1].strip()

    # Update history
    _conversation_history.append(HumanMessage(content=question))
    _conversation_history.append(AIMessage(content=answer))

    # Build sources list
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


def get_chain():
    global _chain_instance
    if _chain_instance is None:
        _chain_instance = build_rag_chain()
    return _chain_instance


def reset_chain():
    global _chain_instance, _conversation_history
    _chain_instance = None
    _conversation_history = []
    print("[RAGChain] Chain and history reset.")


if __name__ == "__main__":
    bundle = get_chain()
    response = query_chain(bundle, "What is the current repo rate set by RBI?")
    print("\nAnswer:", response["answer"])
    print("\nSources:")
    for s in response["sources"]:
        print(f"  [{s['doc_type']}] {s['source']}")
