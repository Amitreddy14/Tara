"""
core/orchestrator.py — Central Brain (Phase 6: with execution system).
"""

import datetime
import logging
import re
import threading
from collections import deque
from typing import Optional

import config

log = logging.getLogger(__name__)


def _build_system_prompt(memory_context: str = "") -> str:
    now      = datetime.datetime.now()
    time_str = now.strftime("%I:%M %p")
    date_str = now.strftime("%A, %B %d, %Y")
    memory_block = f"\n\n{memory_context}" if memory_context else ""

    return f"""You are Tara, a voice assistant. You are mid-conversation — never re-introduce yourself.

CURRENT DATE: {date_str}
CURRENT TIME: {time_str}{memory_block}

STYLE (strict):
- 1 to 2 sentences. Expand only if asked for detail.
- No filler openers: no "Hello!", "Sure!", "Certainly!", "Great!", "Of course!"
- No "As Tara" or "As your assistant"
- Plain spoken English only — no markdown, bullets, or formatting
- Direct and natural, like a calm intelligent friend
- NEVER say "according to my search" or "based on my latest search"
- NEVER invent brand names, product names, or facts you are not certain of
- If you don't know something and no search result was provided, say so honestly

CAPABILITIES:
- Answer questions and have conversations
- Search the web for current information (done automatically when needed)
- Deliver news briefings
- Help with coding questions
- Remember facts and tasks across sessions
- Tell the current time and date
- Perform file and system operations (with your confirmation)

LIMITATIONS:
- Cannot send real notifications or emails
- Always asks before performing any computer action
"""


# ── Confirmation gate state ───────────────────────────────────────────────────

class ConfirmationGate:
    """
    Tracks pending execution actions waiting for user YES/NO.
    The gate is the ONLY path to execution — no bypasses.
    """
    def __init__(self):
        self._pending: Optional[dict] = None
        self._lock = threading.Lock()

    def set(self, action: dict) -> None:
        with self._lock:
            self._pending = action

    def get(self) -> Optional[dict]:
        with self._lock:
            return self._pending

    def clear(self) -> None:
        with self._lock:
            self._pending = None

    def has_pending(self) -> bool:
        with self._lock:
            return self._pending is not None


class Orchestrator:
    def __init__(self):
        self._history: deque[dict] = deque(maxlen=config.HISTORY_TURNS * 2)
        self._lock    = threading.Lock()
        self._client  = None
        self._memory  = None
        self._router  = None
        self._knowledge = None
        self._news    = None
        self._coding  = None
        self._execution = None
        self._gate    = ConfirmationGate()

        self._init_client()
        self._init_memory()
        self._init_subsystems()

    # ── initialisation ────────────────────────────────────────────

    def _init_client(self):
        try:
            from ollama import Client
            self._client = Client(host=config.OLLAMA_HOST)
            self._client.list()
            log.info("Ollama connected at %s — model: %s", config.OLLAMA_HOST, config.LLM_MODEL)
        except Exception as e:
            log.warning("Ollama not available: %s", e)
            self._client = None

    def _init_memory(self):
        try:
            from memory.manager import MemoryManager
            self._memory = MemoryManager(ollama_client=self._client)
            log.info("Memory system ready.")
        except Exception as e:
            log.warning("Memory unavailable: %s", e)
            self._memory = None

    def _init_subsystems(self):
        try:
            from subsystems.router    import SubsystemRouter
            from subsystems.knowledge import KnowledgeSubsystem
            from subsystems.news      import NewsSubsystem
            from subsystems.coding    import CodingSubsystem
            from subsystems.execution import ExecutionSubsystem

            self._router    = SubsystemRouter()
            self._knowledge = KnowledgeSubsystem(self._client, config.LLM_MODEL)
            self._news      = NewsSubsystem(self._client, config.LLM_MODEL)
            self._coding    = CodingSubsystem(self._client, config.LLM_MODEL)
            self._execution = ExecutionSubsystem(self._client, config.LLM_MODEL)
            log.info("Subsystems ready: knowledge, news, coding, execution.")
        except Exception as e:
            log.warning("Subsystems unavailable: %s", e)

    # ── main chat method ──────────────────────────────────────────

    def chat(self, user_text: str) -> str:
        with self._lock:
            if self._client is None:
                self._init_client()
            if self._client is None:
                return "I can't reach my language model. Is Ollama running?"

            # ── 1. Confirmation gate check (highest priority) ──
            if self._gate.has_pending():
                return self._handle_confirmation(user_text)

            # ── 2. Memory commands ──
            if self._memory:
                memory_response = self._memory.handle_memory_command(user_text)
                if memory_response:
                    return memory_response

            # ── 3. Undo command ──
            if self._execution and _is_undo_command(user_text):
                return self._execution.undo_last()

            # ── 4. Task detection ──
            if self._memory and _is_task_request(user_text):
                task = self._memory.parse_and_store_task(user_text)
                if task:
                    due = f" for {task['due_at']}" if task.get("due_at") else ""
                    return f"Done, I've saved that — {task['title']}{due}."

            # ── 5. Execution intent detection ──
            if self._execution and _is_execution_request(user_text):
                return self._handle_execution_request(user_text)

            # ── 6. Subsystem routing ──
            subsystem_result = self._route_to_subsystem(user_text)

            # ── 7. Build memory context ──
            memory_context = ""
            if self._memory:
                memory_context = self._memory.build_memory_context(user_text)

            # ── 8. LLM call ──
            history_content = user_text
            if subsystem_result:
                history_content = (
                    f"{user_text}\n\n"
                    f"[Subsystem result: {subsystem_result}]\n"
                    f"Incorporate the above naturally. Don't say 'according to my search'."
                )

            self._history.append({"role": "user", "content": history_content})
            messages = [{"role": "system", "content": _build_system_prompt(memory_context)}]
            messages.extend(list(self._history))

            try:
                response = self._client.chat(
                    model=config.LLM_MODEL,
                    messages=messages,
                    options={
                        "temperature": config.LLM_TEMPERATURE,
                        "num_predict": config.LLM_MAX_TOKENS,
                        "stop": ["User:", "Human:", "Tara:", "\n\n"],
                    },
                )
                reply = response["message"]["content"].strip()
                reply = _trim_to_complete_sentence(reply)
                reply = _clean_for_tts(reply)
                reply = _strip_filler_opener(reply)

                if not reply:
                    reply = "Could you say that again?"

                # Store clean version in history
                self._history.pop()
                self._history.append({"role": "user", "content": user_text})
                self._history.append({"role": "assistant", "content": reply})

                log.info("LLM (%d words): %r", len(reply.split()), reply[:140])

                if self._memory:
                    self._memory.extract_and_store_facts(list(self._history))

                return reply

            except Exception as e:
                log.error("LLM call failed: %s", e)
                self._history.pop()
                return "Something went wrong on my end. Try again?"

    # ── execution gate handling ───────────────────────────────────

    def _handle_execution_request(self, user_text: str) -> str:
        """Parse the action and present it for confirmation."""
        action = self._execution.parse_action(user_text)
        if not action:
            return self._llm_direct(user_text)

        # Safety check before even asking the user
        safe, reason = self._execution.is_safe(action)
        if not safe:
            log.warning("Execution blocked (safety): %s", reason)
            return f"I can't do that — {reason}"

        # Store pending action and ask for confirmation
        self._gate.set(action)
        confirmation_prompt = self._execution.describe(action)
        log.info("Execution gate: awaiting confirmation for %s", action.get("type"))
        return confirmation_prompt

    def _handle_confirmation(self, user_text: str) -> str:
        """Process user YES/NO response to a pending action."""
        action = self._gate.get()
        self._gate.clear()

        t = user_text.lower().strip()
        confirmed = any(w in t for w in [
            "yes", "yeah", "yep", "sure", "go ahead", "do it",
            "confirm", "ok", "okay", "proceed", "correct", "affirmative"
        ])

        if confirmed:
            log.info("Execution confirmed: %s", action.get("type"))
            result = self._execution.execute(action)
            return result
        else:
            log.info("Execution cancelled by user.")
            return "Okay, I won't do that."

    # ── subsystem routing ─────────────────────────────────────────

    def _route_to_subsystem(self, user_text: str) -> Optional[str]:
        if not self._router:
            return None

        intent = self._router.classify(user_text)
        log.info("Router: %r → %s", user_text[:50], intent)

        try:
            if intent == "knowledge" and self._knowledge:
                result = self._knowledge.search(user_text)
                return result or None
            elif intent == "news" and self._news:
                topic = self._router.extract_news_topic(user_text)
                result = self._news.get_briefing(topic=topic)
                return result or None
            elif intent == "coding" and self._coding:
                result = self._coding.answer(user_text)
                return result or None
        except Exception as e:
            log.error("Subsystem error for %r: %s", intent, e)

        return None

    def _llm_direct(self, user_text: str) -> str:
        """Direct LLM call bypassing subsystems."""
        self._history.append({"role": "user", "content": user_text})
        messages = [{"role": "system", "content": _build_system_prompt()}]
        messages.extend(list(self._history))
        try:
            response = self._client.chat(
                model=config.LLM_MODEL,
                messages=messages,
                options={"temperature": config.LLM_TEMPERATURE, "num_predict": config.LLM_MAX_TOKENS},
            )
            reply = _clean_for_tts(response["message"]["content"].strip())
            self._history.append({"role": "assistant", "content": reply})
            return reply
        except Exception as e:
            self._history.pop()
            return "Something went wrong. Try again?"

    # ── session management ────────────────────────────────────────

    def end_session(self):
        if self._memory and self._history:
            self._memory.end_session(list(self._history))
        if self._client:
            try:
                self._client._client.close()
            except Exception:
                pass

    def clear_history(self):
        with self._lock:
            self._history.clear()
            self._gate.clear()
        log.info("Conversation history cleared.")

    @property
    def history(self):
        return list(self._history)


# ── helpers ───────────────────────────────────────────────────────────────────

def _is_execution_request(text: str) -> bool:
    t = text.lower().strip()
    # Must start with or clearly contain an action verb targeting something
    start_triggers = [
        "open ", "launch ", "start ", "run ", "execute ",
        "delete ", "remove ", "move ", "rename ", "copy ",
        "create a file", "create file", "write to ",
        "read the file", "read file", "show me the file",
        "list files", "show files", "what files",
        "what's in the folder", "what's on my desktop",
    ]
    anywhere_triggers = [
        "open notepad", "open calculator", "open chrome", "open firefox",
        "open explorer", "open terminal", "open cmd",
        "run the command", "run command", "terminal command",
        "delete the file", "move the file", "rename the file",
    ]
    return (
        any(t.startswith(tr) for tr in start_triggers)
        or any(tr in t for tr in anywhere_triggers)
    )


def _is_undo_command(text: str) -> bool:
    t = text.lower().strip()
    return any(p in t for p in ["undo", "undo that", "reverse that", "take that back"])


def _is_task_request(text: str) -> bool:
    t = text.lower().strip()
    rejection_patterns = [
        "aren't you", "you didn't", "you forgot", "did you",
        "have you", "why didn't", "you were supposed", "i thought you",
        "you said you would", "what about", "supposed to",
    ]
    if any(p in t for p in rejection_patterns):
        return False
    affirmative_triggers = [
        "remind me to", "remind me about", "set a reminder",
        "remember to", "don't let me forget", "add a task",
        "add to my list", "note that i need to",
    ]
    return any(trigger in t for trigger in affirmative_triggers)


def _trim_to_complete_sentence(text: str) -> str:
    if not text or text[-1] in ".!?":
        return text
    last = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
    if last > len(text) // 2:
        return text[:last + 1]
    return text


def _clean_for_tts(text: str) -> str:
    text = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*•]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"`+", "", text)
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def _strip_filler_opener(text: str) -> str:
    fillers = [
        "good morning!", "good afternoon!", "good evening!", "good night!",
        "hello!", "hi there!", "hey there!", "greetings!",
        "sure!", "certainly!", "of course!", "absolutely!", "great!",
        "got it!", "understood!", "noted!", "alright!",
        "as tara,", "as your assistant,", "as an ai,",
    ]
    lower = text.lower()
    for filler in fillers:
        if lower.startswith(filler):
            text = text[len(filler):].strip()
            if text:
                text = text[0].upper() + text[1:]
            lower = text.lower()
    return text