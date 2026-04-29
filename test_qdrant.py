import os
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION = "qa_docs"

print(f"QDRANT_URL: {QDRANT_URL}")
print(f"QDRANT_API_KEY: {'set' if QDRANT_API_KEY else 'MISSING'}")

try:
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30)
    print("✅ Client created")
    
    collections = client.get_collections()
    print(f"✅ Collections: {[c.name for c in collections.collections]}")
    
    if COLLECTION in [c.name for c in collections.collections]:
        print(f"✅ {COLLECTION} exists")
    else:
        print(f"❌ {COLLECTION} missing - run ingest.py first")
        
except Exception as e:
    print(f"❌ Error: {e}")