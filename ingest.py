import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

load_dotenv()

# ── config ────────────────────────────────────────────────────────────
QDRANT_URL     = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION     = "qa_docs"
EMBED_MODEL    = "all-MiniLM-L6-v2"
DOCS_FOLDER    = "docs"

# ── step 1: load all docs from folder ────────────────────────────────
def load_documents(folder):
    docs = []
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            path = os.path.join(folder, filename)
            with open(path, "r", encoding="utf-8") as f:
                docs.append({"filename": filename, "text": f.read()})
    print(f"Loaded {len(docs)} document(s)")
    return docs

# ── step 2: chunk with LangChain ─────────────────────────────────────
def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]  # tries these in order
    )
    chunks = []
    for doc in docs:
        splits = splitter.split_text(doc["text"])
        for i, split in enumerate(splits):
            chunks.append({
                "text":     split,
                "filename": doc["filename"],
                "chunk_id": i
            })
    print(f"Created {len(chunks)} chunks")
    return chunks

# ── step 3: embed ─────────────────────────────────────────────────────
def embed_chunks(chunks):
    model  = SentenceTransformer(EMBED_MODEL)
    texts  = [c["text"] for c in chunks]
    vectors = model.encode(texts, show_progress_bar=True)
    print(f"Embedded {len(vectors)} chunks → shape {vectors.shape}")
    return vectors

# ── step 4: push to Qdrant Cloud ──────────────────────────────────────
def push_to_qdrant(chunks, vectors):
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    # create collection if not exists
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION in existing:
        client.delete_collection(COLLECTION)
        print(f"Deleted existing collection: {COLLECTION}")

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(
            size=384,        # all-MiniLM-L6-v2 dimension
            distance=Distance.COSINE
        )
    )
    print(f"Created collection: {COLLECTION}")

    # build points — each point = vector + payload (metadata)
    points = []
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        points.append(PointStruct(
            id=i,
            vector=vector.tolist(),
            payload={
                "text":     chunk["text"],
                "filename": chunk["filename"],
                "chunk_id": chunk["chunk_id"]
            }
        ))

    # upload in one batch
    client.upsert(collection_name=COLLECTION, points=points)
    print(f"Pushed {len(points)} points to Qdrant Cloud ✅")

    # verify
    info = client.get_collection(COLLECTION)
    print(f"Qdrant collection size: {info.points_count} vectors")

# ── run ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    docs    = load_documents(DOCS_FOLDER)
    chunks  = chunk_documents(docs)
    vectors = embed_chunks(chunks)
    push_to_qdrant(chunks, vectors)