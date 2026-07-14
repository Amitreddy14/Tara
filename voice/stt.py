"""
voice/stt.py — Speech-to-Text using OpenAI Whisper (local).

Accepts a raw PCM bytes segment from the VAD, runs Whisper, and returns
the transcribed string.

We use the `faster-whisper` library (CTranslate2 backend) which is
significantly faster than the original `openai-whisper` on CPU.

Usage:
    stt = STT()
    text = stt.transcribe(audio_bytes)   # blocking, ~0.3-2s depending on model
"""

import io
import logging
import time

import numpy as np

import config

log = logging.getLogger(__name__)


class STT:
    def __init__(self):
        log.info("Loading Whisper model '%s' on %s ...",
                 config.WHISPER_MODEL, config.WHISPER_DEVICE)
        t0 = time.time()

        # faster-whisper is a drop-in replacement with ~4x speed on CPU
        from faster_whisper import WhisperModel
        self._model = WhisperModel(
            config.WHISPER_MODEL,
            device=config.WHISPER_DEVICE,
            compute_type="int8",      # int8 = fastest on CPU, negligible accuracy loss
        )
        log.info("Whisper ready in %.1f s", time.time() - t0)

    def transcribe(self, audio_bytes: bytes) -> str:
        """
        Transcribe raw int16 PCM bytes → text string.

        Returns an empty string if nothing intelligible was detected.
        """
        if not audio_bytes:
            return ""

        # Convert int16 PCM → float32 numpy array (Whisper expects float32)
        audio_np = (
            np.frombuffer(audio_bytes, dtype=np.int16)
            .astype(np.float32) / 32768.0
        )

        t0 = time.time()
        segments, info = self._model.transcribe(
            audio_np,
            language=config.WHISPER_LANG,
            beam_size=5,
            vad_filter=False,      # we already have our own VAD
            condition_on_previous_text=False,
        )

        text = " ".join(seg.text.strip() for seg in segments).strip()
        elapsed = time.time() - t0

        # Filter garbage transcriptions
        text = _filter_garbage(text)

        if text:
            log.info("STT (%.2fs): %r", elapsed, text)
        else:
            log.debug("STT produced empty/garbage result (%.2fs)", elapsed)

        return text


def _filter_garbage(text: str) -> str:
    """
    Discard STT results that are clearly noise, not speech.
    Returns empty string if the text should be ignored.
    """
    import re
    if not text:
        return ""
    # Strip punctuation-only results like ". ." or "..." or "，"
    stripped = re.sub(r"[^\w\s]", "", text).strip()
    if not stripped:
        return ""
    # Too short to be meaningful (single char or noise word)
    words = stripped.split()
    if len(words) == 1 and len(words[0]) <= 2:
        return ""
    # Common Whisper hallucinations on silence
    hallucinations = {
        "thank you", "thanks", "you", "the", "uh", "um",
        "hmm", "hm", "ah", "oh", "okay", "ok",
    }
    if stripped.lower() in hallucinations:
        return ""
    return text