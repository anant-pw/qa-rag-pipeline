import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from groq import Groq

load_dotenv()

# ── config ────────────────────────────────────────────────────────────
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
COLLECTION = "qa_docs"
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 3

model = SentenceTransformer(EMBED_MODEL)

# ── step 4: embed query + search Qdrant Cloud ─────────────────────────
def retrieve(question):
    vector = model.encode(question).tolist()

    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=60
    )

    results = client.query_points(
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

# ── step 5: build prompt + generate with Groq ─────────────────────────
def generate(question, chunks):
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

    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a precise QA assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return response.choices[0].message.content

# ── run: interactive CLI loop ─────────────────────────────────────────
if __name__ == "__main__":
    print("\n🔍 QA RAG — CLI Mode")
    print("Type your question. Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() == "exit":
            break
        if not question:
            continue

        chunks = retrieve(question)

        print("\n── Retrieved chunks ──────────────────────────")
        for i, c in enumerate(chunks):
            print(f"  [{i+1}] score: {c['score']} | {c['filename']}")
            print(f"      {c['text'][:100]}...")

        answer = generate(question, chunks)
        print(f"\n── Answer ────────────────────────────────────")
        print(f"{answer}\n")
        print("─" * 50)