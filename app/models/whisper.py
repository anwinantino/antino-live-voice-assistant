import os
import io
import time
import logging
from typing import Tuple
from faster_whisper import WhisperModel
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Constants from environment or defaults
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL", "tiny")

class STTModel:
    def __init__(self, model_size=WHISPER_MODEL_SIZE):
        # Determine device and compute type
        self.device = "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "int8"
        
        logger.info(f"Initializing Whisper ({model_size}) on {self.device}...")
        t0 = time.time()
        self.model = WhisperModel(
            model_size_or_path=model_size, 
            device=self.device, 
            compute_type=self.compute_type
        )
        logger.info(f"Whisper initialized in {time.time()-t0:.2f}s")

    def transcribe_bytes(self, audio_bytes: bytes, confidence_threshold: float = 0.4) -> Tuple[str, float]:
        """
        Converts audio bytes to text using Whisper.
        Uses built-in Silero VAD filter.
        """
        t0 = time.time()
        
        # faster-whisper handles decoding (WAV, WebM, etc.) internally via av/ffmpeg
        audio_stream = io.BytesIO(audio_bytes)
        
        segments, info = self.model.transcribe(
            audio_stream,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            language="en"
        )
        
        valid_texts = []
        for segment in segments:
            # no_speech_prob represents probability of no speech; 1 - prob = confidence
            confidence = 1 - segment.no_speech_prob
            if confidence >= confidence_threshold:
                valid_texts.append(segment.text)
                
        text = " ".join(valid_texts).strip()
        elapsed = time.time() - t0
        
        logger.info(f"[STT] Transcribed in {elapsed:.2f}s: '{text[:60]}...'")
        return text, elapsed

# Singleton instance
_stt_instance = None

def get_stt_model():
    global _stt_instance
    if _stt_instance is None:
        _stt_instance = STTModel()
    return _stt_instance

def transcribe(audio_bytes: bytes) -> Tuple[str, float]:
    """
    Entry point used by api/routes.py
    """
    model = get_stt_model()
    return model.transcribe_bytes(audio_bytes)
