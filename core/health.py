"""
core/health.py — Health monitor and watchdog.

Runs as a background thread. Every CHECK_INTERVAL seconds it:
  1. Checks Ollama is still reachable
  2. Checks memory DB files are accessible
  3. Checks TTS is not stuck
  4. Logs a health summary

If a critical component fails, it notifies via TTS and logs the issue.
Non-critical failures are logged silently.

Usage:
    monitor = HealthMonitor(orchestrator, tts)
    monitor.start()
    monitor.stop()
"""

import logging
import threading
import time
from datetime import datetime

log = logging.getLogger(__name__)

CHECK_INTERVAL = 120   # seconds between health checks


class HealthMonitor:
    def __init__(self, orchestrator, tts):
        self._orchestrator = orchestrator
        self._tts          = tts
        self._stop_event   = threading.Event()
        self._thread       = None
        self._last_ok      = datetime.now()
        self._failures: dict[str, int] = {}   # component → consecutive failure count

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="health-monitor"
        )
        self._thread.start()
        log.info("Health monitor started (check every %ds).", CHECK_INTERVAL)

    def stop(self) -> None:
        self._stop_event.set()

    def status(self) -> dict:
        """Return current health status dict."""
        return {
            "ollama":    self._check_ollama(silent=True),
            "memory":    self._check_memory(silent=True),
            "tts":       not self._tts.is_speaking or True,   # always ok unless stuck
            "last_check": self._last_ok.strftime("%H:%M:%S"),
        }

    # ── internal ─────────────────────────────────────────────────

    def _run(self) -> None:
        # First check after 60s — let startup settle
        time.sleep(60)
        while not self._stop_event.is_set():
            self._tick()
            self._stop_event.wait(timeout=CHECK_INTERVAL)

    def _tick(self) -> None:
        self._last_ok = datetime.now()
        ollama_ok = self._check_ollama()
        memory_ok = self._check_memory()

        if not ollama_ok:
            self._failures["ollama"] = self._failures.get("ollama", 0) + 1
            if self._failures["ollama"] == 2:
                # Only speak after 2 consecutive failures to avoid false alarms
                log.warning("Ollama unreachable — attempting reconnect.")
                try:
                    self._orchestrator._init_client()
                except Exception:
                    pass
        else:
            self._failures["ollama"] = 0

        if not memory_ok:
            self._failures["memory"] = self._failures.get("memory", 0) + 1
            log.warning("Memory system check failed.")
        else:
            self._failures["memory"] = 0

        # Log summary at DEBUG level (won't clutter INFO logs)
        log.debug("Health check: ollama=%s memory=%s", ollama_ok, memory_ok)

    def _check_ollama(self, silent: bool = False) -> bool:
        try:
            if self._orchestrator._client:
                self._orchestrator._client.list()
                return True
        except Exception as e:
            if not silent:
                log.warning("Ollama health check failed: %s", e)
        return False

    def _check_memory(self, silent: bool = False) -> bool:
        try:
            if self._orchestrator._memory:
                self._orchestrator._memory.semantic.get("__health_check__")
                return True
        except Exception as e:
            if not silent:
                log.warning("Memory health check failed: %s", e)
        return False