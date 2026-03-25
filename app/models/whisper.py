"""
Silero VAD + faster-whisper STT pipeline.

Pipeline:
  raw_audio_bytes
    → convert to 16kHz mono PCM WAV  (pydub)
    → Silero VAD  (PyTorch, snakers4/silero-vad)
        → returns timestamped speech segments
    → slice + concatenate speech-only audio
    → faster-whisper transcription
    → text

Silero VAD is a production-grade neural VAD (LSTM) trained by
snakers4. It runs in ~10-30ms on CPU and achieves near-perfect
speech/silence discrimination, removing background noise before
Whisper sees the audio — dramatically improving accuracy.
"""
import io
import wave
import time
import struct
import logging
import os
from typing import List, Tuple
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL", "tiny")
SAMPLE_RATE        = 16000
SILERO_THRESHOLD   = 0.4      # speech probability threshold (0–1)
MIN_SPEECH_MS      = 250      # ignore speech segments shorter than this
MIN_SILENCE_MS     = 100      # min silence gap between speech segments

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
    """
    Lazy-load Silero VAD model + get_speech_timestamps function.
    Silero v5 returns utils as a plain 5-tuple:
      (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks)
    """
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

def _to_16khz_wav(audio_bytes: bytes) -> bytes:
    """Convert any audio format → 16kHz mono 16-bit PCM WAV."""
    try:
        from pydub import AudioSegment
        seg = AudioSegment.from_file(io.BytesIO(audio_bytes))
        seg = seg.set_frame_rate(SAMPLE_RATE).set_channels(1).set_sample_width(2)
        out = io.BytesIO()
        seg.export(out, format="wav")
        result = out.getvalue()
        logger.info(f"[STT] Converted: {len(audio_bytes)}B → {len(result)}B (16kHz WAV)")
        return result
    except Exception as e:
        logger.warning(f"[STT] pydub failed ({e}), using raw audio")
        return audio_bytes


# ── Silero VAD ───────────────────────────────────────────────────────────────

def _silero_filter(wav_bytes: bytes) -> bytes:
    """
    Run Silero VAD on 16kHz WAV audio.
    Returns WAV bytes containing only the speech segments.
    """
    import torch

    model, utils = get_silero_vad()
    # Unpack by position: (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks)
    get_speech_timestamps = utils[0]
    # NOTE: utils[2] (read_audio) is broken on torchaudio >= 2.9 (requires torchcodec).
    # We load the WAV→tensor directly via the wave module instead.

    t0 = time.time()

    # Load WAV directly: int16 PCM → float32 tensor normalised to [-1, 1]
    with wave.open(io.BytesIO(wav_bytes)) as wf:
        sr     = wf.getframerate()
        n      = wf.getnframes()
        raw    = wf.readframes(n)
    samples = struct.unpack(f"<{len(raw)//2}h", raw)
    audio_tensor = torch.tensor(samples, dtype=torch.float32) / 32768.0

    # Get speech timestamps from Silero
    speech_timestamps = get_speech_timestamps(
        audio_tensor,
        model,
        sampling_rate=SAMPLE_RATE,
        threshold=SILERO_THRESHOLD,
        min_speech_duration_ms=MIN_SPEECH_MS,
        min_silence_duration_ms=MIN_SILENCE_MS,
        return_seconds=False,   # sample indices
    )

    elapsed_ms = (time.time() - t0) * 1000
    total_samples = audio_tensor.shape[0]

    if not speech_timestamps:
        logger.warning(f"[VAD] No speech detected in {len(wav_bytes)}B audio ({elapsed_ms:.0f}ms). Using full audio.")
        return wav_bytes

    speech_samples = sum(ts["end"] - ts["start"] for ts in speech_timestamps)
    ratio = speech_samples / max(total_samples, 1)
    logger.info(
        f"[VAD] {len(speech_timestamps)} segment(s), {ratio:.1%} speech "
        f"({elapsed_ms:.0f}ms VAD)"
    )

    # Concatenate speech segments into a single tensor
    segments = [audio_tensor[ts["start"]: ts["end"]] for ts in speech_timestamps]
    speech_tensor = torch.cat(segments)

    # Convert float32 → int16 PCM
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
    Silero VAD → faster-whisper transcription pipeline.

    Steps:
      1. Convert to 16kHz mono PCM WAV        (pydub)
      2. Silero VAD → extract speech segments  (torch, ~10-30ms)
      3. Whisper transcribe speech-only audio  (faster-whisper)

    Returns: (text, total_elapsed_seconds)
    """
    t_total = time.time()
    model = get_whisper_model()

    # Step 1: Convert
    t = time.time()
    wav = _to_16khz_wav(audio_bytes)
    logger.info(f"[STT] Conversion: {(time.time()-t)*1000:.0f}ms")

    # Step 2: Silero VAD
    t = time.time()
    speech_wav = _silero_filter(wav)
    logger.info(f"[STT] VAD+slice: {(time.time()-t)*1000:.0f}ms")

    # Step 3: Whisper
    t = time.time()
    segments, _ = model.transcribe(
        io.BytesIO(speech_wav),
        beam_size=1,
        language="en",
    )
    text = " ".join(seg.text.strip() for seg in segments).strip()
    logger.info(f"[STT] Whisper: {(time.time()-t)*1000:.0f}ms → '{text[:80]}'")

    elapsed = time.time() - t_total
    logger.info(f"[STT] Total pipeline: {elapsed:.2f}s")
    return text, elapsed
