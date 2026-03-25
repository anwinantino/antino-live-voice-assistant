"""
gTTS Text-to-Speech wrapper.
Converts text to MP3 bytes in-memory.
"""
import io
import time
import logging
from gtts import gTTS

logger = logging.getLogger(__name__)


def text_to_audio_bytes(text: str, lang: str = "en") -> tuple[bytes, float]:
    """
    Convert text to MP3 audio bytes using gTTS.
    Returns: (mp3_bytes, elapsed_seconds)
    """
    t0 = time.time()
    if not text or not text.strip():
        return b"", 0.0

    tts = gTTS(text=text.strip(), lang=lang, slow=False)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    audio_bytes = buf.read()
    elapsed = time.time() - t0

    logger.info(f"[TTS] Converted {len(text)} chars in {elapsed:.2f}s → {len(audio_bytes)} bytes")
    return audio_bytes, elapsed
