# Banking RAG Chatbot

An AI-powered chatbot that answers questions about Indian banking — RBI policies, loan FAQs, interest rates, and more. Built with LangChain, FAISS, and Groq (free LLM).

## Quick Start (Local)

### 1. Create conda environment
```bash
conda create -n banking-rag python=3.11 -y
conda activate banking-rag
pip install -r requirements.txt
```

### 2. Set up API key
Get a **free** Groq key at https://console.groq.com (no credit card needed).

```bash
cp .env.example .env
# Edit .env and add: GROQ_API_KEY=your_key_here
```

### 3. Build knowledge base
```bash
python pipeline/updater.py --run-now
```

### 4. Run the app
```bash
streamlit run app/streamlit_app.py
```

---

## Deploy on Streamlit Cloud (Free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo
3. Set **Main file path** to `app/streamlit_app.py`
4. Under **Advanced settings → Secrets**, add:
   ```
   GROQ_API_KEY = "gsk_your_key_here"
   ```
5. Click **Deploy**

> **Note:** After deploying, click the app's menu → **Rerun** once to trigger the first knowledge base build via the sidebar's "Update Knowledge Base" button.

---

## Architecture

```
app/streamlit_app.py        ← Streamlit UI
pipeline/
  rag_chain.py              ← LangChain RAG chain (Groq LLM + FAISS retriever)
  updater.py                ← Ingestion pipeline + APScheduler
ingestion/
  scraper.py                ← RBI + bank FAQ scraper
  loader.py                 ← PDF/HTML/CSV → LangChain Documents
  chunker.py                ← RecursiveCharacterTextSplitter
  embedder.py               ← all-MiniLM-L6-v2 + FAISS index
  deduplicator.py           ← SHA-256 dedup
retrieval/
  retriever.py              ← MMR retriever
  prompt_template.py        ← System + condense prompts
```

## Data Sources
- RBI circulars (rbi.org.in)
- SBI, HDFC, ICICI loan FAQs
- Any PDFs/CSVs you drop into `data/raw/`
