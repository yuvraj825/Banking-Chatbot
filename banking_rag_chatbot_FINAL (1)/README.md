# 🏦 RAG Banking Chatbot with Auto-Updating Knowledge Base

A production-ready financial advisory chatbot that answers queries about RBI policies, bank FAQs, and loan products — and keeps itself updated automatically.

**Stack:** Python · LangChain 1.x (LCEL) · FAISS · HuggingFace · Groq · BeautifulSoup · APScheduler · Streamlit

---

## ✨ Features

- **RAG pipeline** grounded in RBI circulars, SBI/HDFC/ICICI FAQs, and loan documents
- **Auto-updating** — daily scraper keeps the knowledge base current with new RBI circulars
- **SHA-256 deduplication** — only new documents are re-embedded, no full rebuilds
- **Incremental FAISS updates** — new vectors upserted without downtime
- **Source citations** — every answer shows which document it came from
- **Conversation memory** — remembers the last 5 exchanges per session
- **Free LLM** — uses Groq (Llama 3.1 8B, free tier, no credit card needed)

---

## 🚀 Quick Start

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/banking-chatbot.git
cd banking-chatbot/banking_rag_chatbot
pip install -r requirements.txt
```

### 2. Set up your API key (free)

Get a free Groq key at [console.groq.com](https://console.groq.com) — no credit card required.

```bash
cp .env.example .env
# Edit .env and add your key:
# GROQ_API_KEY=your_key_here
```

### 3. Build the knowledge base

```bash
python pipeline/updater.py --run-now
```

This scrapes RBI circulars and bank FAQ pages, embeds them, and saves the FAISS index locally. Takes ~2–5 minutes on first run (downloads the embedding model).

### 4. Run the chatbot

```bash
streamlit run app/streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501)

---

## 📁 Project Structure

```
banking_rag_chatbot/
├── app/
│   └── streamlit_app.py        # Streamlit UI
├── ingestion/
│   ├── scraper.py              # Scrapes RBI + bank FAQ pages
│   ├── loader.py               # PDF / HTML / CSV → LangChain Documents
│   ├── chunker.py              # Splits docs into chunks
│   ├── embedder.py             # Embeds + saves to FAISS
│   └── deduplicator.py         # SHA-256 hash dedup
├── retrieval/
│   ├── retriever.py            # FAISS retriever
│   └── prompt_template.py      # System + RAG prompts
├── pipeline/
│   ├── rag_chain.py            # LCEL RAG chain with memory
│   └── updater.py              # APScheduler orchestrator
├── evaluation/
│   └── ragas_eval.py           # RAGAS evaluation script
├── tests/
│   └── test_queries.py         # Smoke test queries
├── data/
│   ├── raw/                    # Downloaded PDFs + scraped files
│   └── hashes.json             # Dedup hash store
├── vectorstore/
│   └── faiss_index/            # Saved FAISS index (auto-generated)
├── .env.example
├── requirements.txt
└── README.md
```

---

## 🌐 Deploy to Streamlit Cloud

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Set **Main file path** to `banking_rag_chatbot/app/streamlit_app.py`
4. In **Advanced settings → Secrets**, add:
   ```toml
   GROQ_API_KEY = "your_key_here"
   ```
5. Click Deploy

> **Note:** On Streamlit Cloud, the FAISS index must be pre-built and committed to the repo, or rebuilt on first run via the sidebar "Update Knowledge Base" button.

---

## 🔄 Auto-Update Schedule

The scheduler runs daily at 06:00 UTC. To trigger manually:

```bash
# CLI
python pipeline/updater.py --run-now

# Or click "Update Knowledge Base" in the Streamlit sidebar
```

---

## 📊 Evaluation

```bash
pip install ragas datasets
python evaluation/ragas_eval.py
```

Results saved to `evaluation/results.json`.

---

## 🔑 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | ✅ Recommended | Free at [console.groq.com](https://console.groq.com) |
| `HUGGINGFACEHUB_API_TOKEN` | ⬜ Fallback | Free at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |

---

## 📚 Data Sources

| Source | Type |
|---|---|
| RBI circular index | PDF (auto-scraped) |
| SBI home loan FAQ | HTML (auto-scraped) |
| HDFC loan FAQ | HTML (auto-scraped) |
| ICICI loan FAQ | HTML (auto-scraped) |
| Custom files in `data/raw/` | PDF / CSV / TXT (drop manually) |
