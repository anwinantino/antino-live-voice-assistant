# Antino Live Voice Assistant

A production-grade, voice-first AI chatbot for the Antino website — **Gemini Live style**. Uses FastAPI, LangGraph, Pinecone, Ollama (Mistral), faster-whisper STT, gTTS, and BeautifulSoup.

---

## 🚀 Quick Setup

### 1. Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- Pull the Mistral model: `ollama pull mistral`

### 2. Install Dependencies

```bash
cd antino-live-voice-assistant
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### 3. Environment Variables

The `.env` file is pre-configured. Edit if needed:

| Variable | Default |
|---|---|
| `PINECONE_API_KEY` | (pre-set) |
| `PINECONE_INDEX_NAME` | `antino-rag` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` |
| `WHISPER_MODEL` | `tiny` |

### 4. Create Pinecone Index

```bash
python scripts/create_index.py
```

### 5. Seed the Knowledge Base (Antino Website)

```bash
python scripts/ingest_antino.py
```

This crawls and indexes the entire Antino website. Takes ~5-10 minutes on first run.

### 6. Run Component Benchmark Tests

```bash
python scripts/test_components.py
```

This tests and benchmarks every component individually (whisper, embeddings, Pinecone, Ollama, TTS).

### 7. Start the Backend

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8010 --reload
```

### 8. Open the Frontend

Visit: **http://localhost:8010**

---

## 🧠 Architecture

```
VOICE: Audio → Whisper STT → Pinecone Retrieval → Ollama/Mistral (stream) → gTTS → Browser
TEXT:  Text  →              → Pinecone Retrieval → Ollama/Mistral (stream) → gTTS → Browser
```

**LangGraph 4-node pipeline:**
1. `input_node` — transcribes audio or passes text through
2. `retrieval_node` — Pinecone top-4 similarity search
3. `llm_node` — Mistral streaming with strict context grounding
4. `tts_node` — sentence-by-sentence audio generation

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Health check |
| `POST` | `/api/text-chat-stream` | Text query → SSE stream |
| `POST` | `/api/voice-chat-stream` | Audio file → SSE stream |
| `POST` | `/api/ingest` | Ingest URL or document (background) |
| `GET` | `/api/ingestion-status/{task_id}` | Poll ingestion progress |

---

## 🎨 Frontend Features

- Gemini-style animated orb (Idle / Thinking / Speaking states)
- Hold-to-record mic button
- Streaming text display (tokens appear in real-time)
- Base64 audio playback queue
- Sidebar panel: ingest URLs + upload PDF/TXT + progress bar

---

## ⚡ Performance Notes

- **Whisper tiny**: ~0.5-1s transcription
- **Embeddings**: ~50ms per query
- **Pinecone**: ~100ms retrieval
- **Mistral first token**: ~3-8s (CPU) / ~0.5-1s (GPU)
- **TTS per sentence**: ~0.5-1s

To speed up inference, switch to a cloud LLM (Groq/OpenAI) or run Ollama on a GPU.
