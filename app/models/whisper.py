"""
Silero VAD + faster-whisper STT pipeline.

Pipeline:
  raw_audio_bytes (webm / wav / any)
    → convert to 16kHz mono PCM WAV  (pydub, requires ffmpeg)
        IF pydub/ffmpeg not available:
            → pass raw bytes directly to faster-whisper (it handles any format)
    → [optional] Silero VAD → speech-only audio
    → faster-whisper transcription
    → text
"""
import io
import wave
import time
import struct
import logging
import os
from typing import Tuple

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL", "tiny")
SAMPLE_RATE        = 16000
SILERO_THRESHOLD   = 0.4   # speech probability threshold (0–1)
MIN_SPEECH_MS      = 250   # ignore speech segments shorter than this
MIN_SILENCE_MS     = 100   # min silence gap between speech segments

_whisper_model = None
_silero_model  = None
_silero_utils  = None


# ── Model Loading ────────────────────────────────────────────────────────────

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        logger.info(f"Loading faster-whisper ({WHISPER_MODEL_SIZE})...")
        t0 = time.time()
        from faster_whisper import WhisperModel
        _whisper_model = WhisperModel(
            WHISPER_MODEL_SIZE, device="cpu", compute_type="int8"
        )
        logger.info(f"Whisper loaded in {time.time()-t0:.2f}s")
    return _whisper_model


def get_silero_vad():
    """Lazy-load Silero VAD from torch hub (cached after first download)."""
    global _silero_model, _silero_utils
    if _silero_model is None:
        logger.info("Loading Silero VAD...")
        t0 = time.time()
        import torch
        torch.set_num_threads(1)
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        _silero_model = model
        # utils is a 5-tuple: (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks)
        _silero_utils = utils
        logger.info(f"Silero VAD loaded in {time.time()-t0:.2f}s")
    return _silero_model, _silero_utils


# ── Audio Conversion ─────────────────────────────────────────────────────────

def _to_16khz_wav(audio_bytes: bytes):
    """
    Convert any audio format to 16kHz mono 16-bit PCM WAV using pydub.
    Requires ffmpeg to be installed for webm/ogg/mp4 formats.

    Returns (wav_bytes: bytes, success: bool)
    """
    try:
        from pydub import AudioSegment
        seg = AudioSegment.from_file(io.BytesIO(audio_bytes))
        seg = seg.set_frame_rate(SAMPLE_RATE).set_channels(1).set_sample_width(2)
        out = io.BytesIO()
        seg.export(out, format="wav")
        result = out.getvalue()
        logger.info(f"[STT] Converted: {len(audio_bytes)}B → {len(result)}B (16kHz WAV)")
        return result, True
    except Exception as e:
        logger.warning(f"[STT] pydub conversion failed ({e}). Will pass raw bytes to Whisper.")
        return audio_bytes, False


def _is_valid_wav(audio_bytes: bytes) -> bool:
    """Return True if the bytes start with a valid RIFF WAV header."""
    return audio_bytes[:4] == b"RIFF" and audio_bytes[8:12] == b"WAVE"


# ── Silero VAD ───────────────────────────────────────────────────────────────

def _silero_filter(wav_bytes: bytes) -> bytes:
    """
    Run Silero VAD on 16kHz mono PCM WAV bytes.
    Returns WAV bytes containing only the speech segments.
    If no speech is detected, returns the original wav_bytes.
    """
    import torch

    model, utils = get_silero_vad()
    get_speech_timestamps = utils[0]
    # NOTE: utils[2] (read_audio) needs torchcodec on torchaudio >= 2.9 — we load WAV manually instead.

    t0 = time.time()

    # Load WAV directly: int16 PCM → float32 tensor normalized to [-1, 1]
    try:
        with wave.open(io.BytesIO(wav_bytes)) as wf:
            n   = wf.getnframes()
            raw = wf.readframes(n)
    except Exception as e:
        logger.warning(f"[VAD] Could not open WAV for VAD ({e}). Skipping VAD.")
        return wav_bytes

    samples = struct.unpack(f"<{len(raw)//2}h", raw)
    audio_tensor = torch.tensor(samples, dtype=torch.float32) / 32768.0

    # Get speech timestamps
    speech_timestamps = get_speech_timestamps(
        audio_tensor,
        model,
        sampling_rate=SAMPLE_RATE,
        threshold=SILERO_THRESHOLD,
        min_speech_duration_ms=MIN_SPEECH_MS,
        min_silence_duration_ms=MIN_SILENCE_MS,
        return_seconds=False,
    )

    elapsed_ms = (time.time() - t0) * 1000
    total_samples = audio_tensor.shape[0]

    if not speech_timestamps:
        logger.warning(f"[VAD] No speech detected ({elapsed_ms:.0f}ms). Using full audio.")
        return wav_bytes

    speech_samples = sum(ts["end"] - ts["start"] for ts in speech_timestamps)
    ratio = speech_samples / max(total_samples, 1)
    logger.info(f"[VAD] {len(speech_timestamps)} segment(s), {ratio:.1%} speech ({elapsed_ms:.0f}ms)")

    # Concatenate speech segments
    segments = [audio_tensor[ts["start"]: ts["end"]] for ts in speech_timestamps]
    speech_tensor = torch.cat(segments)

    # float32 → int16 PCM
    speech_pcm_np = (speech_tensor.numpy() * 32768).clip(-32768, 32767).astype("int16")
    speech_pcm = speech_pcm_np.tobytes()

    # Re-wrap in WAV container
    out = io.BytesIO()
    with wave.open(out, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(speech_pcm)
    return out.getvalue()


# ── Public API ────────────────────────────────────────────────────────────────

def transcribe(audio_bytes: bytes) -> Tuple[str, float]:
    """
    Transcribe audio bytes to text using Silero VAD + faster-whisper.

    If ffmpeg is available (via pydub):
        audio → 16kHz WAV → Silero VAD → Whisper
    If ffmpeg is NOT available:
        audio → Whisper directly (whisper handles webm/wav/mp4 natively)

    Returns: (text, total_elapsed_seconds)
    """
    t_total = time.time()
    model = get_whisper_model()

    # Step 1: Try to convert to clean 16kHz WAV
    t = time.time()
    wav, converted = _to_16khz_wav(audio_bytes)
    logger.info(f"[STT] Conversion: {(time.time()-t)*1000:.0f}ms (success={converted})")

    if converted and _is_valid_wav(wav):
        # Step 2: Silero VAD (only when we have valid WAV)
        t = time.time()
        speech_wav = _silero_filter(wav)
        logger.info(f"[STT] VAD+slice: {(time.time()-t)*1000:.0f}ms")
        audio_for_whisper = io.BytesIO(speech_wav)
    else:
        # pydub/ffmpeg not available — pass raw bytes to Whisper directly
        # faster-whisper uses its own ffmpeg internally to decode any format
        logger.info("[STT] Skipping VAD — passing raw audio to Whisper")
        audio_for_whisper = io.BytesIO(audio_bytes)

    # Step 3: Whisper transcription
    t = time.time()
    segments, _ = model.transcribe(
        audio_for_whisper,
        beam_size=1,
        language="en",
    )
    text = " ".join(seg.text.strip() for seg in segments).strip()
    logger.info(f"[STT] Whisper: {(time.time()-t)*1000:.0f}ms → '{text[:80]}'")

    elapsed = time.time() - t_total
    logger.info(f"[STT] Total: {elapsed:.2f}s")
    return text, elapsed
