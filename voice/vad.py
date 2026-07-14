"""
voice/vad.py — Voice Activity Detection using Silero VAD.

Consumes raw PCM chunks from a Microphone instance and emits complete
speech segments (as bytes) via a callback.  Runs in its own thread so
it never blocks the main loop.

The state machine inside:
    SILENCE → (confidence > threshold) → SPEECH
    SPEECH  → (silence_duration > SPEECH_PAD_MS) → emit segment → SILENCE

Usage:
    def on_speech(audio_bytes: bytes):
        ...  # called with a complete utterance ready for STT

    vad = VAD(microphone, on_speech_callback=on_speech)
    vad.start()
    ...
    vad.stop()
"""

import logging
import threading
import time
from collections import deque
from typing import Callable

import numpy as np
import torch

import config

log = logging.getLogger(__name__)

# Silero VAD model is downloaded once and cached by torch.hub
_model_cache: dict = {}


def _load_model():
    if "model" not in _model_cache:
        log.info("Loading Silero VAD model (first run may download ~2 MB)...")
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        _model_cache["model"] = model
        _model_cache["utils"] = utils
        log.info("Silero VAD ready.")
    return _model_cache["model"]


class VAD:
    def __init__(self, microphone, on_speech_callback: Callable[[bytes], None], tts=None):
        self._mic = microphone
        self._on_speech = on_speech_callback
        self._tts = tts   # if set, VAD ignores audio while tts.is_speaking
        self._model = _load_model()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    # ── public API ───────────────────────────────────────────────

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name="vad")
        self._thread.start()
        log.info("VAD started.")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)
        log.info("VAD stopped.")

    # ── internal ─────────────────────────────────────────────────

    def _run(self) -> None:
        """
        Main VAD loop.  Reads chunks from mic, runs Silero, accumulates
        speech frames, and emits complete segments.
        """
        pad_chunks   = max(1, config.SPEECH_PAD_MS // config.CHUNK_MS)
        min_chunks   = max(1, config.MIN_SPEECH_MS  // config.CHUNK_MS)

        in_speech        = False
        speech_frames    = []
        silence_counter  = 0

        while not self._stop_event.is_set():
            chunk = self._mic.read(timeout=0.1)
            if chunk is None:
                continue

            # Ignore mic input while Tara is speaking — prevents self-transcription
            if self._tts and self._tts.is_speaking:
                speech_frames   = []
                in_speech       = False
                silence_counter = 0
                self._post_speech_cooldown = 20  # ~640ms of silence after TTS ends
                continue

            # Post-speech cooldown — ignore audio briefly after TTS finishes
            if hasattr(self, '_post_speech_cooldown') and self._post_speech_cooldown > 0:
                self._post_speech_cooldown -= 1
                speech_frames   = []
                in_speech       = False
                silence_counter = 0
                continue

            # Convert raw int16 PCM → float32 tensor in [-1, 1]
            audio_np = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
            audio_t  = torch.from_numpy(audio_np)

            confidence: float = self._model(audio_t, config.SAMPLE_RATE).item()

            if confidence >= config.VAD_THRESHOLD:
                if not in_speech:
                    log.debug("Speech start detected (confidence=%.2f)", confidence)
                    in_speech = True
                    silence_counter = 0
                speech_frames.append(chunk)

            else:
                if in_speech:
                    silence_counter += 1
                    speech_frames.append(chunk)   # keep trailing silence for natural endings

                    if silence_counter >= pad_chunks:
                        # Enough silence — decide if segment is worth emitting
                        if len(speech_frames) >= min_chunks:
                            segment = b"".join(speech_frames)
                            log.debug("Speech segment ready: %.2f s",
                                      len(segment) / (config.SAMPLE_RATE * 2))
                            self._on_speech(segment)
                        else:
                            log.debug("Segment too short, discarding.")

                        speech_frames   = []
                        in_speech       = False
                        silence_counter = 0