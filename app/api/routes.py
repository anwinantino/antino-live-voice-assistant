"""
FastAPI routes: text-chat, voice-chat, ingest, ingestion-status, health.
"""
import asyncio
import logging
import base64
from typing import Optional

from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.rag.stream import stream_rag_response
from app.models import whisper as whisper_model
from app.ingestion import scraper, processor, embedder
from app.utils.helpers import tracker

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Request Models ──────────────────────────────────────────────────────────

class TextChatRequest(BaseModel):
    query: str


class IngestRequest(BaseModel):
    url: Optional[str] = None


# ── Health ──────────────────────────────────────────────────────────────────

@router.get("/health")
async def health():
    return {"status": "ok", "service": "Antino Voice RAG"}


# ── Text Chat Stream ────────────────────────────────────────────────────────

@router.post("/text-chat-stream")
async def text_chat_stream(request: TextChatRequest):
    """
    Accept a text query and return SSE stream with text + audio chunks.
    """
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    logger.info(f"[API] text-chat-stream: '{query}'")

    async def event_generator():
        # Run sync generator in thread pool to avoid blocking event loop
        loop = asyncio.get_event_loop()
        gen = stream_rag_response(query)
        for chunk in gen:
            yield chunk
            await asyncio.sleep(0)  # yield control to event loop

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── Voice Chat Stream ───────────────────────────────────────────────────────

@router.post("/voice-chat-stream")
async def voice_chat_stream(audio: UploadFile = File(...)):
    """
    Accept audio file, transcribe with Whisper, return SSE stream.
    """
    audio_bytes = await audio.read()
    logger.info(f"[API] voice-chat-stream: {len(audio_bytes)} bytes")

    if not audio_bytes:
        raise HTTPException(status_code=400, detail="No audio received.")

    # Transcribe (sync, fast — tiny model ~0.5-1s)
    try:
        query, stt_time = await asyncio.get_event_loop().run_in_executor(
            None, whisper_model.transcribe, audio_bytes
        )
        logger.info(f"[API] Transcribed: '{query}' in {stt_time:.2f}s")
    except Exception as e:
        logger.error(f"[API] Whisper error: {e}")
        raise HTTPException(status_code=500, detail=f"STT error: {e}")

    if not query.strip():
        raise HTTPException(status_code=400, detail="Could not transcribe audio.")

    import json

    async def event_generator():
        # First yield the transcription so frontend can display it
        yield f"data: {json.dumps({'type': 'transcription', 'content': query, 'stt_time': round(stt_time, 3)})}\n\n"
        await asyncio.sleep(0)

        gen = stream_rag_response(query)
        for chunk in gen:
            yield chunk
            await asyncio.sleep(0)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── Ingestion ───────────────────────────────────────────────────────────────

def _run_url_ingestion(task_id: str, url: str):
    """Background task: crawl URL(s), chunk, embed, upsert."""
    try:
        tracker.update(task_id, 5, "running", f"Scraping {url}...")
        pages = scraper.crawl_site(url, max_pages=20)

        tracker.update(task_id, 30, "running", f"Processing {len(pages)} pages...")
        chunks = processor.process_scraped_pages(pages)

        if not chunks:
            tracker.update(task_id, 100, "error", "No content found at URL.")
            return

        tracker.update(task_id, 50, "running", f"Indexing {len(chunks)} chunks...")
        upserted = embedder.upsert_chunks(chunks, task_id=task_id, tracker=tracker)

        tracker.update(task_id, 100, "done", f"Done! Indexed {upserted} chunks from {url}.")
    except Exception as e:
        logger.error(f"[Ingest URL] Error: {e}")
        tracker.update(task_id, 100, "error", f"Error: {str(e)}")


def _run_doc_ingestion(task_id: str, file_bytes: bytes, filename: str, content_type: str):
    """Background task: parse document, chunk, embed, upsert."""
    try:
        tracker.update(task_id, 10, "running", f"Parsing {filename}...")

        if "pdf" in content_type or filename.endswith(".pdf"):
            chunks = processor.process_pdf(file_bytes, filename)
        else:
            chunks = processor.process_txt(file_bytes, filename)

        if not chunks:
            tracker.update(task_id, 100, "error", "No content extracted from document.")
            return

        tracker.update(task_id, 40, "running", f"Indexing {len(chunks)} chunks...")
        upserted = embedder.upsert_chunks(chunks, task_id=task_id, tracker=tracker)

        tracker.update(task_id, 100, "done", f"Done! Indexed {upserted} chunks from {filename}.")
    except Exception as e:
        logger.error(f"[Ingest Doc] Error: {e}")
        tracker.update(task_id, 100, "error", f"Error: {str(e)}")


@router.post("/ingest")
async def ingest(
    background_tasks: BackgroundTasks,
    url: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(default=None),
):
    """
    Ingest a URL or uploaded document into Pinecone.
    Returns task_id for polling progress.
    Note: Runs ONLY in background — does not block query endpoints.
    """
    task_id = tracker.create_task()

    if file and file.filename:
        # Document upload
        file_bytes = await file.read()
        background_tasks.add_task(
            _run_doc_ingestion,
            task_id,
            file_bytes,
            file.filename,
            file.content_type or "",
        )
        return {"task_id": task_id, "message": f"Ingesting document: {file.filename}"}

    elif url:
        # URL crawl
        background_tasks.add_task(_run_url_ingestion, task_id, url)
        return {"task_id": task_id, "message": f"Ingesting URL: {url}"}

    else:
        raise HTTPException(status_code=400, detail="Provide either 'url' or a file upload.")


# ── Ingestion Status ────────────────────────────────────────────────────────

@router.get("/ingestion-status/{task_id}")
async def ingestion_status(task_id: str):
    """Poll the progress of a background ingestion task."""
    status = tracker.to_dict(task_id)
    if "error" in status and status.get("error") == "Task not found":
        raise HTTPException(status_code=404, detail="Task not found.")
    return status
