"""
main.py — Tara entry point (with UI server, ElevenLabs + wake word).
"""

import logging
import os
import signal
import sys
import threading
import time

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from core.config_validator import validate
validate()

import config
from core.state        import TaraState
from core.orchestrator import Orchestrator
from core.proactive    import ProactiveLoop
from core.health       import HealthMonitor
from voice.microphone  import Microphone
from voice.vad         import VAD
from voice.stt         import STT
from voice.tts         import TTS
from voice.wakeword    import WakeWordDetector

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("tara.main")

# ── UI server (optional — graceful if fastapi not installed) ──────────────────
try:
    from server import start_server_thread, emit_state, emit_message, emit_memory, emit_status
    UI_AVAILABLE = True
except ImportError:
    UI_AVAILABLE = False
    def emit_state(s): pass
    def emit_message(r, t): pass
    def emit_memory(f, t): pass
    def emit_status(d): pass

SEARCH_ACK_PHRASES = ["Let me check that.", "One moment.", "Looking that up.", "Let me find out."]
_ack_index = 0
def _next_ack():
    global _ack_index
    p = SEARCH_ACK_PHRASES[_ack_index % len(SEARCH_ACK_PHRASES)]
    _ack_index += 1
    return p

_SEARCH_INTENTS = [
    "what is", "who is", "who are", "when did", "when was",
    "search for", "look up", "find out", "tell me about",
    "latest", "news", "current", "price of", "weather",
    "how does", "explain", "define",
]
def _needs_search(text):
    t = text.lower()
    return any(k in t for k in _SEARCH_INTENTS)

ACTIVE_TIMEOUT = 30


class TaraPipeline:
    def __init__(self):
        self.state = TaraState.IDLE
        self._shutdown = False
        self._last_activity = time.time()

        log.info("Initialising Tara...")
        self.mic          = Microphone()
        self.stt          = STT()
        self.tts          = TTS()
        self.orchestrator = Orchestrator()
        self.vad          = VAD(self.mic, on_speech_callback=self._on_speech, tts=self.tts)
        self.proactive    = ProactiveLoop(
            tts=self.tts, memory=self.orchestrator._memory,
            get_state=lambda: self.state,
            on_proactive=self._on_proactive_message,
        )
        self.health   = HealthMonitor(self.orchestrator, self.tts)
        self.wakeword = WakeWordDetector(on_wake_callback=self._on_wake)
        self._lock    = threading.Lock()
        self._tts_thread = threading.Thread(target=self._tts_loop, daemon=True, name="tts-player")

    # ── lifecycle ─────────────────────────────────────────────────

    def start(self):
        self._tts_thread.start()
        if UI_AVAILABLE:
            start_server_thread()
        self.mic.start()
        self.vad.start()
        self.proactive.start()
        self.health.start()

        if self.wakeword.available:
            self.wakeword.start()
            self.state = TaraState.SLEEPING
            self._print_banner()
            emit_state("sleeping")
            self.tts.speak(f"Hey, I'm {config.TARA_NAME}. Say 'Hey Jarvis' to wake me up.")
        else:
            self.state = TaraState.IDLE
            self._print_banner()
            emit_state("idle")
            self.tts.speak(f"Hello. I'm {config.TARA_NAME}. How can I help?")

        log.info("Tara is ready. Ctrl-C to quit.")
        if UI_AVAILABLE:
            log.info("UI available at http://localhost:8000")

    def stop(self):
        log.info("Shutting down...")
        self._shutdown = True
        self.health.stop()
        self.proactive.stop()
        self.wakeword.stop()
        self.orchestrator.end_session()
        self.vad.stop()
        self.mic.stop()
        self.tts.flush_queue()
        self.tts.speak("Goodbye.")
        time.sleep(3)

    def run_loop(self):
        while not self._shutdown:
            if (self.wakeword.available
                    and self.state == TaraState.IDLE
                    and time.time() - self._last_activity > ACTIVE_TIMEOUT):
                log.info("No activity — returning to sleep.")
                self.state = TaraState.SLEEPING
                emit_state("sleeping")
                self.tts.speak("Going to sleep. Say 'Hey Jarvis' to wake me.")
            time.sleep(0.5)

    # ── TTS player ────────────────────────────────────────────────

    def _tts_loop(self):
        while not self._shutdown:
            self.tts.run_pending()
            time.sleep(0.05)

    # ── wake word ─────────────────────────────────────────────────

    def _on_wake(self):
        if self.state == TaraState.SLEEPING:
            log.info("Wake word triggered.")
            self.state = TaraState.IDLE
            self._last_activity = time.time()
            emit_state("idle")
            self.tts.speak("I'm listening.")

    # ── proactive ─────────────────────────────────────────────────

    def _on_proactive_message(self, message):
        if self.state in (TaraState.SPEAKING, TaraState.PROCESSING,
                          TaraState.LISTENING, TaraState.SLEEPING):
            return
        if not self._lock.acquire(blocking=False):
            return
        try:
            self.state = TaraState.SPEAKING
            emit_state("speaking")
            emit_message("tara", message)
            self.tts.speak(message)
            self._wait_for_speech()
            self.state = TaraState.IDLE
            emit_state("idle")
        finally:
            self._lock.release()

    # ── reactive ──────────────────────────────────────────────────

    def _on_speech(self, audio_bytes):
        if self.state == TaraState.SLEEPING:
            return
        if self.state == TaraState.SPEAKING:
            log.info("Interruption — stopping TTS.")
            self.tts.stop()
            self.state = TaraState.INTERRUPTED
            emit_state("interrupted")

        if not self._lock.acquire(blocking=False):
            return

        try:
            self.state = TaraState.PROCESSING
            emit_state("listening")
            self.mic.flush()
            self._last_activity = time.time()

            user_text = self.stt.transcribe(audio_bytes)
            if not user_text:
                self.state = TaraState.IDLE
                emit_state("idle")
                return

            if _is_reset_command(user_text):
                self.orchestrator.clear_history()
                self.tts.speak("Starting fresh.")
                self._wait_for_speech()
                self.state = TaraState.IDLE
                emit_state("idle")
                return

            log.info("User: %s", user_text)
            emit_message("user", user_text)
            emit_state("processing")

            if _needs_search(user_text):
                ack = _next_ack()
                self.tts.speak(ack)
                emit_message("tara", ack)
                self._wait_for_speech()

            reply = self.orchestrator.chat(user_text)

            self.state = TaraState.SPEAKING
            emit_state("speaking")
            emit_message("tara", reply)
            self.tts.speak(reply)
            self._wait_for_speech()

            # Refresh memory in UI after each exchange
            if self.orchestrator._memory:
                facts = self.orchestrator._memory.semantic.get_all()
                tasks = self.orchestrator._memory.tasks.get_pending()
                emit_memory(facts, tasks)

            self._last_activity = time.time()
            self.state = TaraState.IDLE
            emit_state("idle")

        finally:
            self._lock.release()

    def _wait_for_speech(self, timeout=30.0):
        deadline = time.time() + timeout
        time.sleep(0.2)
        while time.time() < deadline:
            if not self.tts.is_speaking and self.tts._queue.empty():
                break
            time.sleep(0.1)

    def _print_banner(self):
        memory_ok  = self.orchestrator._memory is not None
        subs_ok    = self.orchestrator._router is not None
        actual_tts = getattr(self.tts, '_engine_name', config.TTS_ENGINE)
        wake_status = getattr(config, 'WAKE_WORD', None) or 'always-on'
        print("\n" + "="*55)
        print(f"  TARA v0.8  —  {config.TARA_NAME} is ready")
        print("="*55)
        print(f"  STT model  : {config.WHISPER_MODEL}")
        print(f"  LLM model  : {config.LLM_MODEL} @ {config.OLLAMA_HOST}")
        print(f"  TTS engine : {actual_tts} ({config.TTS_VOICE or 'default'})")
        print(f"  Wake word  : {wake_status}")
        print(f"  Memory     : {'✓ ready' if memory_ok else '✗ unavailable'}")
        print(f"  Subsystems : {'✓ knowledge, news, coding, execution' if subs_ok else '✗ unavailable'}")
        print(f"  UI         : {'✓ http://localhost:8000' if UI_AVAILABLE else '✗ install fastapi uvicorn'}")
        print("="*55 + "\n")


def _is_reset_command(text):
    t = text.lower().strip()
    return any(p in t for p in ["start over", "clear history", "reset", "forget everything"])


def main():
    pipeline = TaraPipeline()
    pipeline.start()
    _shutting_down = False

    def _shutdown(sig, frame):
        nonlocal _shutting_down
        if _shutting_down:
            return
        _shutting_down = True
        print()
        pipeline.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        pipeline.run_loop()
    except KeyboardInterrupt:
        _shutdown(None, None)


if __name__ == "__main__":
    main()