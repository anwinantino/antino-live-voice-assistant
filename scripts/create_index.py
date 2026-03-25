"""
Create the Pinecone serverless index for Antino RAG.
Run once: python scripts/create_index.py
"""
import os
import time
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "antino-rag")
EMBED_DIM = 384
METRIC = "cosine"
CLOUD = "aws"
REGION = "us-east-1"


def create_index():
    from pinecone import Pinecone, ServerlessSpec

    pc = Pinecone(api_key=PINECONE_API_KEY)

    existing = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX_NAME in existing:
        print(f"[OK] Index '{PINECONE_INDEX_NAME}' already exists. Skipping creation.")
        return

    print(f"Creating Pinecone index: {PINECONE_INDEX_NAME} (dim={EMBED_DIM}, metric={METRIC})")
    t0 = time.time()
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=EMBED_DIM,
        metric=METRIC,
        spec=ServerlessSpec(cloud=CLOUD, region=REGION),
    )

    # Wait for index to be ready
    while True:
        status = pc.describe_index(PINECONE_INDEX_NAME).status
        if status.get("ready", False):
            break
        print("  Waiting for index to be ready...")
        time.sleep(2)

    elapsed = time.time() - t0
    print(f"[DONE] Index '{PINECONE_INDEX_NAME}' created and ready in {elapsed:.1f}s")


if __name__ == "__main__":
    create_index()
