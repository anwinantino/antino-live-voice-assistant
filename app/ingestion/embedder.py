"""
HuggingFace Embeddings + Pinecone vector store.
Handles embedding generation and batch upsert.
"""
import os
import time
import logging
import uuid
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "antino-rag")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBED_DIM = 384
BATCH_SIZE = 100

_embed_model = None
_pinecone_index = None


def get_embed_model():
    global _embed_model
    if _embed_model is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        t0 = time.time()
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info(f"Embedding model loaded in {time.time()-t0:.2f}s")
    return _embed_model


def get_pinecone_index():
    global _pinecone_index
    if _pinecone_index is None:
        from pinecone import Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)
        _pinecone_index = pc.Index(PINECONE_INDEX_NAME)
        logger.info(f"Connected to Pinecone index: {PINECONE_INDEX_NAME}")
    return _pinecone_index


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for a list of texts."""
    t0 = time.time()
    model = get_embed_model()
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=False).tolist()
    elapsed = time.time() - t0
    logger.info(f"[Embedder] Embedded {len(texts)} texts in {elapsed:.2f}s")
    return embeddings


def upsert_chunks(chunks: List[Dict], task_id: str = None, tracker=None) -> int:
    """
    Embed and upsert chunks to Pinecone.
    chunks: list of {"text": ..., "source": ..., "type": ...}
    Returns total upserted count.
    """
    index = get_pinecone_index()
    total = len(chunks)
    upserted = 0

    for i in range(0, total, BATCH_SIZE):
        batch = chunks[i: i + BATCH_SIZE]
        texts = [c["text"] for c in batch]

        t0 = time.time()
        embeddings = embed_texts(texts)

        vectors = [
            {
                "id": str(uuid.uuid4()),
                "values": emb,
                "metadata": {
                    "text": chunk["text"][:1000],  # Pinecone metadata limit
                    "source": chunk.get("source", ""),
                    "type": chunk.get("type", "web"),
                },
            }
            for chunk, emb in zip(batch, embeddings)
        ]

        index.upsert(vectors=vectors)
        upserted += len(vectors)
        elapsed = time.time() - t0

        progress = int((i + len(batch)) / total * 100)
        logger.info(f"[Embedder] Upserted batch {i//BATCH_SIZE+1} ({upserted}/{total}) in {elapsed:.2f}s")

        if tracker and task_id:
            tracker.update(task_id, progress,
                           "running", f"Indexed {upserted}/{total} chunks...")

    return upserted


def query_pinecone(query_text: str, top_k: int = 4) -> tuple[str, float]:
    """
    Embed a query and retrieve top-k context chunks.
    Returns: (combined_context_str, elapsed_seconds)
    """
    t0 = time.time()
    index = get_pinecone_index()
    embedder = get_embed_model()

    q_emb = embedder.encode([query_text]).tolist()[0]
    results = index.query(vector=q_emb, top_k=top_k, include_metadata=True)

    context_parts = []
    for match in results.get("matches", []):
        meta = match.get("metadata", {})
        text = meta.get("text", "")
        source = meta.get("source", "")
        if text:
            context_parts.append(f"[Source: {source}]\n{text}")

    context = "\n\n---\n\n".join(context_parts)
    elapsed = time.time() - t0
    logger.info(f"[Embedder] Pinecone query returned {len(context_parts)} chunks in {elapsed:.2f}s")
    return context, elapsed
