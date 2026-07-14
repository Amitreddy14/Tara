"""
voice/microphone.py — continuous microphone capture.

Runs a background thread that reads raw PCM chunks from the default
input device and pushes them onto a thread-safe queue.  The VAD module
reads from this queue independently.

Usage:
    mic = Microphone()
    mic.start()
    chunk = mic.read()   # blocks until a chunk is available
    mic.stop()
"""

import queue
import threading
import logging

import pyaudio

import config

log = logging.getLogger(__name__)


class Microphone:
    def __init__(self):
        self._q: queue.Queue[bytes] = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._pa = pyaudio.PyAudio()
        self._stream = None

    # ── public API ───────────────────────────────────────────────

    def start(self) -> None:
        """Open the audio stream and begin capturing in the background."""
        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=config.CHANNELS,
            rate=config.SAMPLE_RATE,
            input=True,
            frames_per_buffer=config.CHUNK_SIZE,
            stream_callback=self._callback,
        )
        self._stream.start_stream()
        log.info("Microphone started (rate=%d, chunk=%d frames)",
                 config.SAMPLE_RATE, config.CHUNK_SIZE)

    def read(self, timeout: float = 1.0) -> bytes | None:
        """
        Block until a chunk is available, then return it.
        Returns None if no chunk arrives within `timeout` seconds.
        """
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        """Stop capture and release the audio device."""
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        self._pa.terminate()
        log.info("Microphone stopped.")

    def flush(self) -> None:
        """Discard buffered chunks (call before/after a TTS playback)."""
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except queue.Empty:
                break

    # ── internal ─────────────────────────────────────────────────

    def _callback(self, in_data, frame_count, time_info, status):
        """PyAudio calls this from its own thread for each chunk."""
        if status:
            log.debug("PyAudio status: %s", status)
        self._q.put(in_data)
        return (None, pyaudio.paContinue)