"""
faster-whisper STT model wrapper.
Uses 'tiny' model by default for speed (~0.5-1s on CPU).
"""
import os
import io
import time
import logging
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL", "tiny")

_whisper_model = None


def get_whisper_model():
    """Lazy-load and cache the Whisper model singleton."""
    global _whisper_model
    if _whisper_model is None:
        logger.info(f"Loading faster-whisper model: {WHISPER_MODEL_SIZE}")
        t0 = time.time()
        from faster_whisper import WhisperModel
        _whisper_model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device="cpu",
            compute_type="int8",
        )
        logger.info(f"Whisper model loaded in {time.time()-t0:.2f}s")
    return _whisper_model


def transcribe(audio_bytes: bytes) -> tuple[str, float]:
    """
    Transcribe audio bytes to text.
    Returns: (text, elapsed_seconds)
    """
    t0 = time.time()
    model = get_whisper_model()

    audio_io = io.BytesIO(audio_bytes)
    segments, info = model.transcribe(audio_io, beam_size=1, language="en")

    text = " ".join(seg.text.strip() for seg in segments).strip()
    elapsed = time.time() - t0

    logger.info(f"[Whisper] Transcribed in {elapsed:.2f}s → '{text[:60]}...'")
    return text, elapsed
