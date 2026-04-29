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

---

## 🎯 Full RAG Interview Q&A

### Tier 1 — Fundamentals

**Q: What is RAG and why does it exist?**
> LLMs are trained on frozen data — they don't know your private documents or recent events. RAG fixes this by retrieving relevant chunks from your own data at query time and injecting them into the prompt. The LLM answers from your content, not from training memory.

**Q: What is an embedding?**
> An embedding converts text into a fixed-size vector of numbers where semantic similarity maps to geometric closeness. "How does RAG work?" and "Explain retrieval augmented generation" share almost no words but produce very similar vectors because they mean the same thing.

**Q: What is a vector database?**
> A database optimized for nearest neighbor search in high-dimensional space. Instead of exact keyword match (SQL WHERE), it finds the most geometrically similar vectors to a query vector using cosine similarity or L2 distance.

**Q: Why can't you use a regular database for vectors?**
> SQL does exact or range queries. Finding the 3 most similar vectors across 384 dimensions requires specialized indexes (HNSW, IVF) that SQL engines don't have. A WHERE clause cannot express "find me the closest vector to this query vector."

**Q: What is cosine similarity?**
> A measure of the angle between two vectors. Score of 1.0 = identical direction = same meaning. Score of 0.0 = perpendicular = unrelated. Score of -1.0 = opposite. RAG retrieval ranks chunks by cosine similarity to the query vector and returns the top-k.

**Q: What is chunking and why does it matter?**
> Documents are too large to embed whole — embedding models have token limits, and large chunks dilute retrieval precision. Chunking splits documents into smaller pieces. Each chunk should contain one complete idea — enough context to answer a question on its own.

**Q: Why must the same embedding model be used for indexing and querying?**
> Each embedding model creates its own vector space. If you embed chunks with model A and the query with model B, the vectors live in different spaces. Cosine distances between them are meaningless — retrieval returns garbage.

**Q: What is top-k retrieval?**
> After embedding the query, FAISS or Qdrant returns the k most similar chunks. k=3 means you get the 3 closest chunks. Too small = miss relevant context. Too large = add noise that confuses the LLM. Typical production value: 3–5.

**Q: What is the difference between indexing and querying in RAG?**
> Indexing is the offline phase — chunk documents, embed chunks, store vectors in the vector DB. Runs once (or when docs change). Querying is the online phase — embed the user question, search the vector DB, retrieve top-k chunks, generate answer with LLM. Runs on every user request.

**Q: How does RAG prevent hallucination?**
> RAG alone doesn't — the prompt does. You explicitly instruct the LLM to answer only from the retrieved context and say "I don't know" if the answer isn't there. Without that instruction the LLM falls back to training memory, defeating the purpose of retrieval.

**Q: What happens when the answer is not in the docs?**
> FAISS and Qdrant always return top-k results — they have no concept of "no match." The retrieved chunks will be the closest available, even if unrelated. The prompt instruction ("say not found if answer absent") is the only mechanism that handles this case correctly.

**Q: What is a prompt and what role does it play in RAG?**
> The prompt is the full text sent to the LLM — it combines retrieved chunks (context) + user question + behavioral instructions. It is your only control panel over LLM behavior. Every constraint you want (stay on topic, don't hallucinate, answer briefly) must be explicitly written in the prompt.

---

### Tier 2 — Design Decisions

**Q: How do you decide chunk size?**
> It depends on document type and query nature. Smaller chunks give precise retrieval but lose context. Larger chunks give more context but reduce retrieval precision. Overlap prevents meaning loss at boundaries. Start at 500 tokens with 50-token overlap and tune based on answer quality.

**Q: What is chunk overlap and why does it exist?**
> Overlap means the next chunk starts N characters before the previous one ended. Without overlap, a sentence split across two chunks loses meaning at the boundary — neither chunk contains the complete thought. Overlap ensures boundary content appears in at least one complete chunk.

**Q: What breaks if chunk size is too small? Too large?**
> Too small: retrieved chunk lacks enough context for the LLM to answer — it gets a sentence fragment with no surrounding meaning. Too large: retrieved chunk contains too many topics — the LLM gets noise alongside the relevant content, reducing answer precision.

**Q: What is the difference between FAISS and Qdrant?**
> FAISS is an in-process library — vectors live in RAM on one machine, lost when the process stops. Qdrant is a persistent cloud service — vectors survive restarts, queryable over HTTP from anywhere, support metadata filtering, and scale horizontally. Retrieval concept is identical — nearest neighbor search — but Qdrant is production infrastructure, not a local index.

**Q: What is L2 distance vs cosine similarity?**
> L2 (Euclidean) measures straight-line distance between vector tips — sensitive to vector magnitude. Cosine measures the angle between vectors — ignores magnitude, only cares about direction. For text embeddings, cosine is generally preferred because two sentences can have different lengths but same meaning — cosine handles this correctly, L2 does not.

**Q: What is top-k and how do you tune it?**
> top-k controls how many chunks are retrieved per query. Too small (k=1) = miss relevant context, answer incomplete. Too large (k=10) = noisy context, LLM gets confused, also costs more tokens sent to the LLM API. Tune by testing answer quality vs cost. Most production systems use k=3 to 5.

**Q: How do you handle a query that is too long to embed?**
> Embedding models have token limits (all-MiniLM = 256 tokens). For long queries, either truncate (accept loss), summarize using an LLM first then embed the summary, or extract the core intent with an LLM and embed that. Short queries (under 100 tokens) rarely hit this limit.

**Q: How do you update a RAG system when documents change?**
> Re-run the ingestion pipeline — re-chunk, re-embed, re-upload to the vector DB. For Qdrant, use upsert so existing points are overwritten. In production this runs as a scheduled job (nightly) or triggered on document update events. The Streamlit UI in this project has a Re-index button that does exactly this.

**Q: What is metadata filtering in retrieval?**
> Attaching structured data (filename, date, category, author) to each vector as payload. At query time you can filter by metadata before running similarity search — e.g., "search only in test cases, not bug reports" or "only docs from this month." Qdrant supports this natively via payload filters.

**Q: What is source attribution and why does it matter?**
> Returning the filename, page number, or URL of the chunk that produced the answer. Lets users verify the answer against the original document. Critical for trust in production systems — especially legal, medical, or compliance domains. In this project, filename and cosine score are returned with every answer.
