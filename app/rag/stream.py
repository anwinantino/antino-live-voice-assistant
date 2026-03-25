"""
Sentence-by-sentence SSE streaming generator.
Streams LLM tokens, buffers them per sentence, then:
  1. Yields a {type:"text"} SSE chunk
  2. Calls gTTS and yields a {type:"audio", content: base64} SSE chunk

This creates the Gemini Live "text appears then audio follows" feel.
"""
import re
import json
import time
import base64
import logging
from typing import Generator
from app.ingestion.embedder import query_pinecone
from app.models.ollama import stream_response
from app.models.tts import text_to_audio_bytes

logger = logging.getLogger(__name__)

SENTENCE_END = re.compile(r'(?<=[.!?])\s')


def _split_sentence(buffer: str) -> tuple[str, str]:
    """
    Split buffer at the last sentence boundary.
    Returns (complete_sentence, remainder).
    """
    parts = SENTENCE_END.split(buffer, maxsplit=1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1]
    return "", buffer


def _sse_event(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


def stream_rag_response(query: str) -> Generator[str, None, None]:
    """
    Full RAG streaming pipeline:
      1. Pinecone retrieval
      2. Stream Mistral token-by-token
      3. Buffer per sentence → yield text SSE + audio SSE

    Yields SSE-formatted strings.
    """
    total_start = time.time()

    # ── Step 1: Retrieval ───────────────────────────────────────
    context, retrieval_time = query_pinecone(query, top_k=4)
    logger.info(f"[Stream] Retrieval: {retrieval_time:.2f}s")
    yield _sse_event({"type": "meta", "retrieval_time": round(retrieval_time, 3)})

    if not context.strip():
        fallback = "I don't have that information in my knowledge base."
        yield _sse_event({"type": "text", "content": fallback, "done": False})
        audio_bytes, _ = text_to_audio_bytes(fallback)
        if audio_bytes:
            yield _sse_event({
                "type": "audio",
                "content": base64.b64encode(audio_bytes).decode(),
            })
        yield _sse_event({"type": "done"})
        return

    # ── Step 2: LLM Streaming ───────────────────────────────────
    token_buffer = ""
    full_response = ""
    first_token_time = None
    sentence_count = 0

    for token, _ in stream_response(context, query):
        if first_token_time is None:
            first_token_time = time.time()
            ttft = first_token_time - total_start
            logger.info(f"[Stream] TTFT: {ttft:.2f}s")
            yield _sse_event({"type": "meta", "ttft": round(ttft, 3)})

        token_buffer += token
        full_response += token

        # Check for sentence boundary
        sentence, token_buffer = _split_sentence(token_buffer)

        if sentence:
            sentence_count += 1
            logger.info(f"[Stream] Sentence {sentence_count}: '{sentence[:40]}...'")

            # Yield text chunk immediately
            yield _sse_event({"type": "text", "content": sentence + " ", "done": False})

            # Generate TTS for this sentence
            t_tts = time.time()
            audio_bytes, tts_time = text_to_audio_bytes(sentence)
            logger.info(f"[Stream] TTS sentence {sentence_count}: {tts_time:.2f}s")

            if audio_bytes:
                yield _sse_event({
                    "type": "audio",
                    "content": base64.b64encode(audio_bytes).decode(),
                    "tts_time": round(tts_time, 3),
                })

    # Flush remaining buffer
    remainder = token_buffer.strip()
    if remainder:
        yield _sse_event({"type": "text", "content": remainder, "done": False})
        audio_bytes, _ = text_to_audio_bytes(remainder)
        if audio_bytes:
            yield _sse_event({
                "type": "audio",
                "content": base64.b64encode(audio_bytes).decode(),
            })

    total_elapsed = time.time() - total_start
    logger.info(f"[Stream] Complete pipeline: {total_elapsed:.2f}s | Response: {len(full_response)} chars")
    yield _sse_event({
        "type": "done",
        "total_time": round(total_elapsed, 3),
        "response_length": len(full_response),
    })
