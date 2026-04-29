import os
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from groq import Groq

load_dotenv()

# ── config ────────────────────────────────────────────────────────────
QDRANT_URL     = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
COLLECTION     = "qa_docs"
EMBED_MODEL    = "all-MiniLM-L6-v2"
TOP_K          = 3

# ── cache model so it loads once, not on every query ──────────────────
@st.cache_resource
def load_model():
    return SentenceTransformer(EMBED_MODEL)

@st.cache_resource
def load_clients():
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    groq   = Groq(api_key=GROQ_API_KEY)
    return qdrant, groq

# ── retrieve from Qdrant ──────────────────────────────────────────────
def retrieve(question, qdrant_client, model):
    vector  = model.encode(question).tolist()
    results = qdrant_client.query_points(
        collection_name=COLLECTION,
        query=vector,
        limit=TOP_K,
        with_payload=True
    )
    chunks = []
    for r in results.points:
        payload = r.payload or {}
        chunks.append({
            "text": payload.get("text", ""),
            "filename": payload.get("filename", "unknown"),
            "score": round(r.score, 4)
        })
    return chunks

# ── generate from Groq ────────────────────────────────────────────────
def generate(question, chunks, groq_client):
    context = "\n\n".join([
        f"[Source: {c['filename']} | score: {c['score']}]\n{c['text']}"
        for c in chunks
    ])
    prompt = f"""You are a QA assistant. Answer using ONLY the context below.
If the answer is not in the context, say "Not found in QA docs."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# ── UI ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="QA Knowledge Bot", page_icon="🔍")
st.title("🔍 QA Knowledge Bot")
st.caption("Ask anything about your test cases and bug reports.")

# load once
model                = load_model()
qdrant_client, groq_client = load_clients()

# chat history — persists across questions in same session
if "history" not in st.session_state:
    st.session_state.history = []

# input box
question = st.chat_input("Ask a question about your QA docs...")

if question:
    # show user message
    with st.chat_message("user"):
        st.write(question)

    # retrieve + generate
    with st.spinner("Searching QA docs..."):
        chunks = retrieve(question, qdrant_client, model)
        answer = generate(question, chunks, groq_client)

    # show answer
    with st.chat_message("assistant"):
        st.write(answer)

        # show sources in expander — collapsed by default
        with st.expander("📄 Sources retrieved"):
            for i, c in enumerate(chunks):
                st.markdown(f"**[{i+1}] {c['filename']} — score: `{c['score']}`**")
                st.caption(c['text'])

    # save to history
    st.session_state.history.append({
        "question": question,
        "answer":   answer
    })

# show history in sidebar
with st.sidebar:
    st.header("💬 Session History")
    if st.session_state.history:
        for i, h in enumerate(reversed(st.session_state.history)):
            with st.expander(f"Q{len(st.session_state.history)-i}: {h['question'][:40]}..."):
                st.write(h["answer"])
    else:
        st.caption("No questions yet.")

    # re-index button
    st.divider()
    if st.button("🔄 Re-index docs"):
        os.system("python ingest.py")
        st.success("Re-indexed successfully!")