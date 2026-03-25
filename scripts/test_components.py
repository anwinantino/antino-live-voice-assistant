"""
Component Benchmark Test Suite
================================
Tests and benchmarks EVERY component individually with precise timing.
Run: python scripts/test_components.py

All components are tested in isolation with clear pass/fail + timing.
"""
import sys
import os
import time
import json

# Fix Windows Unicode output
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

PASS = "✅ PASS"
FAIL = "❌ FAIL"
WARN = "⚠️  WARN"
SEP = "─" * 60


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def result(name, ok, elapsed, note=""):
    status = PASS if ok else FAIL
    timing = f"{elapsed*1000:.0f}ms" if elapsed < 1 else f"{elapsed:.2f}s"
    print(f"  {status}  {name:<35} [{timing}] {note}")


# ─────────────────────────────────────────────────────────────────
# 1. Environment Variables
# ─────────────────────────────────────────────────────────────────
section("1. Environment Variables")
required_vars = ["PINECONE_API_KEY", "PINECONE_INDEX_NAME", "OLLAMA_BASE_URL", "EMBEDDING_MODEL"]
all_ok = True
for var in required_vars:
    val = os.getenv(var, "")
    ok = bool(val)
    if not ok:
        all_ok = False
    print(f"  {'✅' if ok else '❌'}  {var} = {'***' + val[-4:] if val else 'NOT SET'}")


# ─────────────────────────────────────────────────────────────────
# 2. Pinecone Connection & Index
# ─────────────────────────────────────────────────────────────────
section("2. Pinecone Connection")
try:
    t0 = time.time()
    from pinecone import Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    indexes = [i.name for i in pc.list_indexes()]
    elapsed = time.time() - t0
    index_name = os.getenv("PINECONE_INDEX_NAME", "antino-rag")
    ok = index_name in indexes
    result("Pinecone client init + list_indexes", ok, elapsed,
           f"Indexes: {indexes}")

    if ok:
        t0 = time.time()
        idx = pc.Index(index_name)
        stats = idx.describe_index_stats()
        elapsed = time.time() - t0
        vector_count = stats.get("total_vector_count", 0)
        result("Describe index stats", True, elapsed,
               f"Vectors in index: {vector_count}")
    else:
        print(f"  {FAIL}  Index '{index_name}' not found. Run: python scripts/create_index.py")
except Exception as e:
    result("Pinecone connection", False, 0, f"Error: {e}")


# ─────────────────────────────────────────────────────────────────
# 3. Sentence Transformers (Embeddings)
# ─────────────────────────────────────────────────────────────────
section("3. Embedding Model (SentenceTransformer)")
try:
    t0 = time.time()
    from sentence_transformers import SentenceTransformer
    model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    model = SentenceTransformer(model_name)
    load_elapsed = time.time() - t0
    result(f"Load {model_name}", True, load_elapsed)

    # Single encode
    t0 = time.time()
    emb = model.encode(["What services does Antino offer?"])[0]
    single_elapsed = time.time() - t0
    result("Encode single query", True, single_elapsed,
           f"dim={len(emb)}")

    # Batch encode
    t0 = time.time()
    texts = [f"Test document chunk number {i}" for i in range(50)]
    embs = model.encode(texts)
    batch_elapsed = time.time() - t0
    result("Encode batch (50 texts)", True, batch_elapsed,
           f"{batch_elapsed/50*1000:.1f}ms/chunk")
except Exception as e:
    result("Embedding model", False, 0, f"Error: {e}")


# ─────────────────────────────────────────────────────────────────
# 4. Pinecone Query (RAG Retrieval)
# ─────────────────────────────────────────────────────────────────
section("4. Pinecone Query (Retrieval)")
try:
    from app.ingestion.embedder import query_pinecone

    test_queries = [
        "What services does Antino offer?",
        "Who are the clients of Antino?",
        "What is Antino known for?",
    ]
    for q in test_queries:
        t0 = time.time()
        context, elapsed = query_pinecone(q, top_k=4)
        ctx_len = len(context)
        result(f"Query: '{q[:30]}...'", ctx_len > 0, elapsed,
               f"{ctx_len} chars retrieved")
except Exception as e:
    result("Pinecone query", False, 0, f"Error: {e}")


# ─────────────────────────────────────────────────────────────────
# 5. BeautifulSoup Scraper
# ─────────────────────────────────────────────────────────────────
section("5. Web Scraper (BeautifulSoup)")
try:
    from app.ingestion.scraper import scrape_url

    t0 = time.time()
    data = scrape_url("https://www.antino.com/")
    elapsed = time.time() - t0
    ok = bool(data.get("text"))
    result("Scrape https://www.antino.com/", ok, elapsed,
           f"{len(data.get('text',''))} chars, {len(data.get('tables',[]))} tables")
except Exception as e:
    result("Web scraper", False, 0, f"Error: {e}")


# ─────────────────────────────────────────────────────────────────
# 6. Text Processor (Chunking)
# ─────────────────────────────────────────────────────────────────
section("6. Text Processor (Chunking)")
try:
    from app.ingestion.processor import chunk_text, clean_text

    sample_text = " ".join([f"This is sentence number {i} about Antino software development." for i in range(200)])
    t0 = time.time()
    cleaned = clean_text(sample_text)
    chunks = chunk_text(cleaned, source="test", doc_type="test")
    elapsed = time.time() - t0
    result("Chunk 200-sentence text", len(chunks) > 0, elapsed,
           f"{len(chunks)} chunks from {len(sample_text)} chars")
except Exception as e:
    result("Text processor", False, 0, f"Error: {e}")


# ─────────────────────────────────────────────────────────────────
# 7. Whisper STT
# ─────────────────────────────────────────────────────────────────
section("7. Whisper STT (faster-whisper)")
try:
    from app.models.whisper import get_whisper_model, transcribe

    t0 = time.time()
    model = get_whisper_model()
    load_elapsed = time.time() - t0
    result("Load Whisper model (tiny)", True, load_elapsed)

    # Generate a small synthetic audio (silence, 1s, 16kHz WAV)
    import struct, wave, io
    sample_rate = 16000
    duration = 1.0
    samples = [0] * int(sample_rate * duration)
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f'<{len(samples)}h', *samples))
    wav_bytes = wav_buf.getvalue()

    t0 = time.time()
    text, _ = transcribe(wav_bytes)
    elapsed = time.time() - t0
    result("Transcribe 1s silent audio", True, elapsed,
           f"Output: '{text}' (expected empty/short)")
except Exception as e:
    result("Whisper STT", False, 0, f"Error: {e}")


# ─────────────────────────────────────────────────────────────────
# 8. gTTS (Text-to-Speech)
# ─────────────────────────────────────────────────────────────────
section("8. gTTS (Text-to-Speech)")
try:
    from app.models.tts import text_to_audio_bytes

    sentences = [
        "Hello! I am the Antino virtual assistant.",
        "We specialize in building world-class software products.",
        "Please visit our website to learn more about our services.",
    ]
    for sentence in sentences:
        t0 = time.time()
        audio_bytes, elapsed = text_to_audio_bytes(sentence)
        ok = len(audio_bytes) > 0
        result(f"TTS: '{sentence[:40]}...'", ok, elapsed,
               f"{len(audio_bytes)/1024:.1f}KB audio")
except Exception as e:
    result("gTTS TTS", False, 0, f"Error: {e}")


# ─────────────────────────────────────────────────────────────────
# 9. Ollama / Mistral LLM
# ─────────────────────────────────────────────────────────────────
section("9. Ollama LLM (Mistral)")
try:
    import httpx
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # Check if Ollama is running
    t0 = time.time()
    try:
        r = httpx.get(f"{base_url}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        elapsed = time.time() - t0
        ok = any("mistral" in m for m in models)
        result("Ollama server reachable", True, elapsed,
               f"Models: {models}")
        if not ok:
            print(f"  {WARN}  Mistral not found. Run: ollama pull mistral")
    except Exception as e:
        result("Ollama server reachable", False, 0, f"Error: {e}")
        print(f"  {WARN}  Ensure Ollama is running: ollama serve")
        raise

    # Test streaming
    from app.models.ollama import stream_response
    context = "Antino is a software development company based in India founded in 2012."
    query = "When was Antino founded?"

    t0 = time.time()
    tokens = []
    first_token_time = None
    for token, _ in stream_response(context, query):
        tokens.append(token)
        if first_token_time is None:
            first_token_time = time.time() - t0

    total_elapsed = time.time() - t0
    full_response = "".join(tokens)

    result("Ollama TTFT (time to first token)", first_token_time is not None,
           first_token_time or 0)
    result("Ollama full response generation", len(full_response) > 0,
           total_elapsed, f"{len(tokens)} tokens, {len(full_response)} chars")
    print(f"\n  Response preview: '{full_response[:100].strip()}...'")
except Exception as e:
    if "Ollama server" not in str(e):
        result("Ollama LLM", False, 0, f"Error: {e}")


# ─────────────────────────────────────────────────────────────────
# 10. Full RAG Stream (End-to-End, no voice)
# ─────────────────────────────────────────────────────────────────
section("10. Full RAG Stream Pipeline (Text → SSE Chunks)")
try:
    from app.rag.stream import stream_rag_response
    import json as _json

    query = "What does Antino do?"
    t0 = time.time()
    chunks_received = []
    text_chunks = []
    audio_chunks = []
    meta_info = {}
    first_text_time = None

    for sse_chunk in stream_rag_response(query):
        # Parse SSE data line
        for line in sse_chunk.split("\n"):
            if line.startswith("data: "):
                try:
                    data = _json.loads(line[6:])
                    chunks_received.append(data)
                    if data["type"] == "text" and first_text_time is None:
                        first_text_time = time.time() - t0
                    elif data["type"] == "text":
                        text_chunks.append(data["content"])
                    elif data["type"] == "audio":
                        audio_chunks.append(data["content"])
                    elif data["type"] == "meta":
                        meta_info.update(data)
                except Exception:
                    pass

    total_elapsed = time.time() - t0

    result("Time to first text chunk (TTFT)", first_text_time is not None,
           first_text_time or 0)
    result("Total text chunks received", len(text_chunks) > 0, total_elapsed,
           f"{len(text_chunks)} text + {len(audio_chunks)} audio chunks")

    if meta_info.get("retrieval_time"):
        print(f"\n  📊 Pipeline Breakdown:")
        print(f"     Pinecone retrieval:    {meta_info.get('retrieval_time', '?'):.3f}s")
        print(f"     Time to first token:   {meta_info.get('ttft', '?'):.3f}s" if meta_info.get('ttft') else "")
        print(f"     Total pipeline time:   {total_elapsed:.2f}s")

except Exception as e:
    result("Full RAG Stream", False, 0, f"Error: {e}")


# ─────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  Component Benchmark Test Complete")
print(f"{'='*60}\n")
