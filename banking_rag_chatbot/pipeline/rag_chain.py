"""
pipeline/rag_chain.py
Assembles the full conversational RAG chain with memory.

LLM: Groq API — Llama 3.1 8B (free tier, no credit card needed)
     Fallback: HuggingFace Inference API (Zephyr-7B-Beta)
Memory: ConversationBufferWindowMemory (last 5 exchanges)

Why Groq?
  - 100% free, 14,400 requests/day on free tier
  - No credit card required
  - Sub-second latency
  - Get free key at: https://console.groq.com
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Always load .env from the project root, regardless of working directory
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env", override=True)

# Streamlit Cloud: read secrets from st.secrets if available
try:
    import streamlit as st
    if hasattr(st, "secrets"):
        for _key in ["GROQ_API_KEY", "HUGGINGFACEHUB_API_TOKEN"]:
            if _key in st.secrets and not os.getenv(_key):
                os.environ[_key] = st.secrets[_key]
except Exception:
    pass  # Not running in Streamlit context

sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from langchain.memory import ConversationBufferWindowMemory
except ImportError:
    from langchain_community.memory import ConversationBufferWindowMemory

try:
    from langchain.chains import ConversationalRetrievalChain
except ImportError:
    from langchain_community.chains import ConversationalRetrievalChain

from retrieval.retriever import get_retriever
from retrieval.prompt_template import get_chat_prompt, get_condense_prompt
from ingestion.embedder import index_exists

MEMORY_WINDOW = 5


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
        "RECOMMENDED (Free, fast):\n"
        "  1. Go to https://console.groq.com and sign up free\n"
        "  2. Create an API key\n"
        "  3. Add to your .env file: GROQ_API_KEY=your_key_here\n\n"
        "ALTERNATIVE (HuggingFace free):\n"
        "  1. Go to https://huggingface.co/settings/tokens\n"
        "  2. Create a READ token\n"
        "  3. Add to your .env file: HUGGINGFACEHUB_API_TOKEN=your_token"
    )


def _build_memory():
    return ConversationBufferWindowMemory(
        k=MEMORY_WINDOW,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
        input_key="question",
    )


def build_rag_chain():
    if not index_exists():
        raise FileNotFoundError(
            "FAISS index not found. Build the knowledge base first:\n"
            "  python pipeline/updater.py --run-now"
        )

    llm = _build_llm()
    retriever = get_retriever(search_type="mmr", k=4)
    memory = _build_memory()

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        condense_question_prompt=get_condense_prompt(),
        combine_docs_chain_kwargs={"prompt": get_chat_prompt()},
        return_source_documents=True,
        verbose=False,
        # get_chat_history removed in langchain 0.2
    )

    print("[RAGChain] Chain assembled successfully.")
    return chain


def query_chain(chain, question: str) -> dict:
    result = chain.invoke({"question": question})
    answer = result.get("answer", "No answer generated.")

    # Clean LLM prompt echo artefacts
    for marker in ["Answer:", "Assistant:", "Human:", "\nQuestion:"]:
        if marker in answer:
            answer = answer.split(marker)[-1].strip()

    seen, sources = set(), []
    for doc in result.get("source_documents", []):
        src = doc.metadata.get("source", "Unknown")
        if src not in seen:
            seen.add(src)
            sources.append({
                "content": doc.page_content[:300],
                "source": src,
                "doc_type": doc.metadata.get("doc_type", "unknown"),
            })

    return {"answer": answer, "sources": sources}


_chain_instance = None


def get_chain():
    global _chain_instance
    if _chain_instance is None:
        _chain_instance = build_rag_chain()
    return _chain_instance


def reset_chain():
    global _chain_instance
    _chain_instance = None
    print("[RAGChain] Chain reset.")


if __name__ == "__main__":
    chain = get_chain()
    response = query_chain(chain, "What is the current repo rate set by RBI?")
    print("\nAnswer:", response["answer"])
