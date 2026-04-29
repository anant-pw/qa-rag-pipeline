# 🔍 QA Knowledge Bot — RAG Pipeline

A production-grade RAG (Retrieval-Augmented Generation) system that lets you ask questions about your QA docs — test cases, bug reports, SRS — and get grounded answers with source attribution.

Built from scratch to understand every layer of RAG. No black boxes.

---

## 🧠 What is RAG?

LLMs have frozen knowledge (training cutoff). RAG gives them **your** documents at query time.

```
your docs → chunk → embed → store in vector DB   (indexing, runs once)
user question → embed → search → retrieve → LLM  (querying, runs every time)
```

The LLM never guesses. It reads your chunks and answers from them.

---

## 🏗️ Architecture

```
INDEXING PHASE (ingest.py)
──────────────────────────────────────────────────────
docs/*.txt
    ↓ LangChain RecursiveCharacterTextSplitter
8–20 chunks (~300 chars, 50 overlap)
    ↓ sentence-transformers (all-MiniLM-L6-v2)
8–20 × 384 float32 vectors
    ↓
Qdrant Cloud ← stored here permanently

QUERYING PHASE (query.py / app.py)
──────────────────────────────────────────────────────
user question (no chunking — embeds as single string)
    ↓ same embedding model (all-MiniLM-L6-v2)
1 × 384 query vector
    ↓ Qdrant cosine similarity search
top 3 matching chunks + source filenames + scores
    ↓ Groq (llama-3.1-8b-instant)
grounded answer ("Not found in QA docs" if absent)
```

---

## ⚙️ Stack

| Component | Tool | Why |
|---|---|---|
| Chunking | LangChain `RecursiveCharacterTextSplitter` | Splits on paragraphs → sentences → words (smarter than fixed slicing) |
| Embedding | `sentence-transformers` / `all-MiniLM-L6-v2` | Free, runs on CPU, 384-dim vectors |
| Vector DB | Qdrant Cloud | Real cloud vector DB, free tier, production-grade |
| LLM | Groq + `llama-3.1-8b-instant` | Free tier, ~1s response, no GPU needed |
| Web UI | Streamlit | Browser interface in ~10 lines of Python |

---

## 🚀 Setup (15 minutes)

### 1. Clone repo

```bash
git clone https://github.com/anant-pw/qa-rag-pipeline.git
cd qa-rag-pipeline
```

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies

```bash
pip install langchain langchain-community langchain-groq sentence-transformers qdrant-client streamlit python-dotenv groq
```

### 4. Get free API keys

| Service | URL | What to copy |
|---|---|---|
| Qdrant Cloud | cloud.qdrant.io | Cluster URL + API Key |
| Groq | console.groq.com | API Key |

### 5. Create `.env` file

```
QDRANT_URL=https://xxxx.qdrant.io
QDRANT_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
```

### 6. Add your QA docs

Drop any `.txt` files into the `docs/` folder. A sample `docs/sample_qa.txt` is included with test cases and bug reports.

---

## ▶️ Run

### Step 1 — Index your docs (run once, or when docs change)

```bash
python ingest.py
```

Output:
```
Loaded 1 document(s)
Created 18 chunks
Embedded 18 chunks → shape (18, 384)
Pushed 18 points to Qdrant Cloud ✅
```

### Step 2a — CLI mode

```bash
python query.py
```

```
🔍 QA RAG — CLI Mode
You: What happens after 3 failed login attempts?

── Retrieved chunks ──────────────────────
  [1] score: 0.8821 | sample_qa.txt
  [2] score: 0.6103 | sample_qa.txt

── Answer ────────────────────────────────
After 3 failed attempts, the account is locked. (TC002)
```

### Step 2b — Web UI mode

```bash
streamlit run app.py
```

Opens at `http://localhost:8501` — chat interface with source expander and session history.

---

## 💡 Key RAG concepts this project teaches

| Concept | Where it happens |
|---|---|
| Chunking strategy | `ingest.py` → `RecursiveCharacterTextSplitter` |
| Why same embedding model for index + query | `ingest.py` + `query.py` both use `all-MiniLM-L6-v2` |
| Vector DB vs regular DB | Qdrant cosine search vs SQL WHERE |
| Cosine similarity score | Logged per chunk in CLI output |
| Hallucination prevention | Prompt rule: "answer ONLY from context" |
| Source attribution | Filename + score returned with every answer |
| Re-indexing on doc update | Re-index button in Streamlit sidebar |

---

## 🏢 Production equivalent stack

```
this project          production
────────────────      ──────────────────────────────
LangChain splitter  = LangChain / LlamaIndex
all-MiniLM          = OpenAI text-embedding-3-large
Qdrant Cloud        = Qdrant Cloud / Pinecone
Groq + llama3       = GPT-4o / Claude
Streamlit           = React + FastAPI
ingest.py           = scheduled nightly pipeline
```

Concepts identical. Tools scale up. That's it.

---

## 📁 Project structure

```
qa-rag-pipeline/
├── docs/
│   └── sample_qa.txt     ← your QA documents go here
├── ingest.py             ← chunk + embed + push to Qdrant
├── query.py              ← CLI query engine
├── app.py                ← Streamlit web UI
├── .env                  ← API keys (never committed)
└── README.md
```

---

## 🎯 Interview answers this project unlocks

> **"What is RAG?"** — LLMs have frozen knowledge. RAG retrieves relevant chunks from your documents at query time and injects them into the prompt, grounding the answer in your data.

> **"How do you prevent hallucination in RAG?"** — The prompt explicitly instructs the LLM to answer only from retrieved context and say "not found" otherwise. Without this instruction, the LLM falls back to training memory.

> **"Why can't you use PostgreSQL as a vector store?"** — SQL does exact/range queries. Nearest neighbor search in 384-dimensional space requires specialized indexes — Qdrant uses HNSW optimized for cosine similarity at scale.

> **"What breaks if you use different embedding models for indexing and querying?"** — Vectors live in different spaces. Cosine distances become meaningless. Retrieval returns garbage.

---

Built to understand RAG deeply — not just use it.
