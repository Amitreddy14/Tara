"""
voice/wakeword.py — Wake word detection using openwakeword.

Listens passively in the background. When the wake word is detected,
calls on_wake_callback() which switches Tara from SLEEPING to LISTENING.

After each activation, enters a cooldown period before listening again
to prevent double-triggers from the same utterance.

Supported models (set WAKE_WORD in config.py):
  "hey_jarvis"   — sounds like "Hey Tara" / "Hey Jarvis"  (recommended)
  "alexa"        — "Alexa"
  "hey_mycroft"  — "Hey Mycroft"

Usage:
    detector = WakeWordDetector(on_wake_callback=my_callback)
    detector.start()
    ...
    detector.stop()
"""

import logging
import threading
import time
from typing import Callable

import numpy as np
import pyaudio

import config

log = logging.getLogger(__name__)

CHUNK           = 1280      # openwakeword needs 80ms chunks at 16kHz
COOLDOWN_SEC    = 2.0       # seconds to wait after detection before re-arming
DETECTION_THRESHOLD = 0.5   # confidence threshold (0-1)


class WakeWordDetector:
    def __init__(self, on_wake_callback: Callable[[], None]):
        self._on_wake    = on_wake_callback
        self._stop_event = threading.Event()
        self._thread     = None
        self._model      = None
        self._available  = False
        self._load_model()

    def _load_model(self) -> None:
        wake_word = getattr(config, "WAKE_WORD", None)
        if not wake_word:
            log.info("Wake word disabled (WAKE_WORD is None in config).")
            return
        try:
            from openwakeword.model import Model
            self._model = Model(
                wakeword_models=[wake_word],
                inference_framework="onnx",
            )
            self._wake_word = wake_word
            self._available = True
            log.info("Wake word ready: '%s' (threshold=%.2f)", wake_word, DETECTION_THRESHOLD)
        except ImportError:
            log.warning("openwakeword not installed — pip install openwakeword")
        except Exception as e:
            log.warning("Wake word model failed to load: %s", e)

    @property
    def available(self) -> bool:
        return self._available

    def start(self) -> None:
        if not self._available:
            return
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="wakeword"
        )
        self._thread.start()
        log.info("Wake word detector started. Say '%s' to activate Tara.",
                 getattr(self, "_wake_word", "wake word"))

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)
        log.info("Wake word detector stopped.")

    def _run(self) -> None:
        pa = pyaudio.PyAudio()
        stream = pa.open(
            rate=16000,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=CHUNK,
        )

        log.info("Wake word listening...")
        last_detection = 0.0

        try:
            while not self._stop_event.is_set():
                raw = stream.read(CHUNK, exception_on_overflow=False)
                audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

                # Cooldown — don't re-trigger immediately after detection
                if time.time() - last_detection < COOLDOWN_SEC:
                    continue

                try:
                    pred = self._model.predict(audio)
                    # pred is a dict: {model_name: score} 
                    score = max(pred.values()) if pred else 0.0
                    if score >= DETECTION_THRESHOLD:
                        last_detection = time.time()
                        log.info("Wake word detected! (score=%.2f)", score)     
                        self._on_wake()
                except Exception as e:
                    log.debug("Wake word predict error: %s", e)
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()