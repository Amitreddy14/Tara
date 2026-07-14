"""
core/proactive.py — Proactive Loop.

Runs as a daemon thread. Every CHECK_INTERVAL seconds it:
  1. Checks hard triggers (due tasks, scheduled reminders)
  2. Checks soft triggers (time-of-day greetings, idle nudges)
  3. Scores receptivity — suppresses if Tara is speaking or user is active
  4. Enforces interrupt budget (max N proactive messages per hour)
  5. If all checks pass: fires a proactive message via TTS

Proactive messages are short, single-sentence, and always relevant.
They never fire during an active conversation.

Usage (from main.py):
    loop = ProactiveLoop(tts, memory, orchestrator_state)
    loop.start()
    ...
    loop.stop()
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Callable

import config

log = logging.getLogger(__name__)

# ── Tuning constants ──────────────────────────────────────────────────────────
CHECK_INTERVAL       = 60       # seconds between proactive checks
MAX_PER_HOUR         = 4        # hard cap on proactive interrupts per hour
MIN_GAP_SECONDS      = 120      # minimum seconds between any two proactive messages
MORNING_HOUR         = 8        # hour to deliver morning greeting (24h)
EVENING_HOUR         = 20       # hour to deliver evening wind-down


class ProactiveLoop:
    def __init__(
        self,
        tts,
        memory,
        get_state: Callable,          # callable → current TaraState
        on_proactive: Callable[[str], None],  # callable to queue a TTS message
    ):
        self._tts          = tts
        self._memory       = memory
        self._get_state    = get_state
        self._on_proactive = on_proactive

        self._stop_event   = threading.Event()
        self._thread       = None

        # Interrupt budget tracking
        self._fired_times: list[datetime] = []       # timestamps of proactive fires
        self._last_fire: datetime | None  = None

        # Greeting flags (reset daily)
        self._greeted_morning = False
        self._greeted_evening = False
        self._last_greeting_date: datetime | None = None

        log.info("Proactive loop initialised (check every %ds, max %d/hr).",
                 CHECK_INTERVAL, MAX_PER_HOUR)

    # ── lifecycle ─────────────────────────────────────────────────

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="proactive"
        )
        self._thread.start()
        log.info("Proactive loop started.")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3)
        log.info("Proactive loop stopped.")

    # ── main loop ─────────────────────────────────────────────────

    def _run(self) -> None:
        # Stagger first check by 30s so startup isn't cluttered
        time.sleep(30)

        while not self._stop_event.is_set():
            try:
                self._tick()
            except Exception as e:
                log.error("Proactive loop error: %s", e)
            self._stop_event.wait(timeout=CHECK_INTERVAL)

    def _tick(self) -> None:
        """One check cycle."""
        now = datetime.now()

        # Reset daily greeting flags at midnight
        if self._last_greeting_date and self._last_greeting_date.date() != now.date():
            self._greeted_morning = False
            self._greeted_evening = False
        self._last_greeting_date = now

        # ── Receptivity gate ──────────────────────────────────────
        if not self._is_receptive():
            log.debug("Proactive: not receptive right now, skipping.")
            return

        # ── Interrupt budget gate ─────────────────────────────────
        if not self._within_budget(now):
            log.debug("Proactive: hourly budget exhausted, skipping.")
            return

        # ── Try triggers in priority order ────────────────────────
        message = (
            self._check_due_tasks(now)
            or self._check_morning_greeting(now)
            or self._check_evening_greeting(now)
            or self._check_idle_nudge(now)
        )

        if message:
            self._fire(message, now)

    # ── receptivity check ─────────────────────────────────────────

    def _is_receptive(self) -> bool:
        """
        Returns True if it's safe to interrupt the user.
        False if Tara is speaking or processing.
        """
        from core.state import TaraState
        state = self._get_state()
        if state in (TaraState.SPEAKING, TaraState.PROCESSING, TaraState.LISTENING):
            return False
        if self._tts.is_speaking:
            return False
        return True

    # ── budget check ──────────────────────────────────────────────

    def _within_budget(self, now: datetime) -> bool:
        # Remove fires older than 1 hour
        cutoff = now - timedelta(hours=1)
        self._fired_times = [t for t in self._fired_times if t > cutoff]

        if len(self._fired_times) >= MAX_PER_HOUR:
            return False

        if self._last_fire and (now - self._last_fire).total_seconds() < MIN_GAP_SECONDS:
            return False

        return True

    # ── trigger checks ────────────────────────────────────────────

    def _check_due_tasks(self, now: datetime) -> str | None:
        """Fire if a task is past due."""
        if not self._memory:
            return None
        try:
            due = self._memory.tasks.get_due_now()
            if due:
                task = due[0]
                self._memory.tasks.mark_done(task["id"])   # mark so it doesn't repeat
                return f"Just a heads up — your reminder is due: {task['title']}."
        except Exception as e:
            log.debug("Due task check failed: %s", e)
        return None

    def _check_morning_greeting(self, now: datetime) -> str | None:
        """Fire a morning greeting between MORNING_HOUR and MORNING_HOUR+1."""
        if self._greeted_morning:
            return None
        if now.hour == MORNING_HOUR:
            self._greeted_morning = True
            name = ""
            if self._memory:
                try:
                    stored = self._memory.semantic.get("name")
                    if stored:
                        name = f", {stored}"
                except Exception:
                    pass
            return f"Good morning{name}. I'm here whenever you need me."
        return None

    def _check_evening_greeting(self, now: datetime) -> str | None:
        """Fire an evening check-in at EVENING_HOUR."""
        if self._greeted_evening:
            return None
        if now.hour == EVENING_HOUR:
            self._greeted_evening = True
            return "Good evening. Let me know if there's anything you'd like to wrap up before the day ends."
        return None

    def _check_idle_nudge(self, now: datetime) -> str | None:
        """
        Fire a gentle nudge if there are pending tasks and Tara hasn't
        spoken proactively for a long time.
        Suppressed during night hours (23:00 - 07:00).
        """
        if now.hour >= 23 or now.hour < 7:
            return None
        if not self._memory:
            return None

        # Only nudge if it's been > 2 hours since last proactive fire
        if self._last_fire and (now - self._last_fire).total_seconds() < 7200:
            return None

        try:
            pending = self._memory.tasks.get_pending()
            if pending:
                count = len(pending)
                top   = pending[0]["title"]
                if count == 1:
                    return f"You still have one pending task: {top}."
                else:
                    return f"You have {count} pending tasks. The next one is: {top}."
        except Exception as e:
            log.debug("Idle nudge check failed: %s", e)
        return None

    # ── fire ──────────────────────────────────────────────────────

    def _fire(self, message: str, now: datetime) -> None:
        log.info("Proactive firing: %r", message)
        self._fired_times.append(now)
        self._last_fire = now
        self._on_proactive(message)