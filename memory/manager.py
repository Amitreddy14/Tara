"""
memory/manager.py — Unified memory interface.

The orchestrator talks only to this class. It coordinates:
  - SemanticMemory  (user facts, SQLite)
  - EpisodicMemory  (past conversations, ChromaDB)
  - TaskMemory      (reminders and tasks, SQLite)

Also handles:
  - Salience scoring of conversations (LLM call at session end)
  - Fact extraction from conversations (LLM call, runs async-ish)
  - Task parsing from natural language (e.g. "remind me to call mum tomorrow")
"""

import logging
import re
import threading
from datetime import datetime, timedelta
from typing import Optional

import config

log = logging.getLogger(__name__)


class MemoryManager:
    def __init__(self, ollama_client=None):
        from memory.semantic  import SemanticMemory
        from memory.episodic  import EpisodicMemory
        from memory.tasks     import TaskMemory

        self.semantic  = SemanticMemory()
        self.episodic  = EpisodicMemory()
        self.tasks     = TaskMemory()
        self._client   = ollama_client   # shared Ollama client for salience scoring
        self._lock     = threading.Lock()

    # ── prompt context injection ──────────────────────────────────

    def build_memory_context(self, user_query: str) -> str:
        """
        Build a memory context block to inject into the system prompt.
        Called at the start of each LLM call.
        """
        parts = []

        semantic_block = self.semantic.format_for_prompt()
        if semantic_block:
            parts.append(semantic_block)

        episodic_block = self.episodic.format_for_prompt(user_query)
        if episodic_block:
            parts.append(episodic_block)

        task_block = self.tasks.format_for_prompt()
        if task_block:
            parts.append(task_block)

        return "\n\n".join(parts)

    # ── fact extraction ───────────────────────────────────────────

    def extract_and_store_facts(self, conversation: list[dict]) -> None:
        """
        Run in a background thread after each exchange.
        Asks the LLM to extract learnable facts from the conversation.
        """
        if not self._client or not conversation:
            return
        threading.Thread(
            target=self._extract_facts_worker,
            args=(conversation,),
            daemon=True,
            name="fact-extractor"
        ).start()

    def _extract_facts_worker(self, conversation: list[dict]) -> None:
        excerpt = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}"
            for m in conversation[-6:]   # last 3 exchanges
        )
        prompt = f"""Read this conversation and extract ONLY clear, stable, explicitly stated facts about the user.

Rules:
- Only extract facts the user directly stated about themselves (e.g. "my name is X", "I work at Y", "I live in Z")
- Do NOT infer or guess facts
- Do NOT extract names of places, countries, or things being discussed
- Do NOT extract facts Tara stated — only what the USER said about themselves
- If no clear facts were stated, return {{}}
- Return ONLY a JSON object like {{"key": "value"}}

Valid examples:
  User says "my name is Amit" → {{"name": "Amit"}}
  User says "I'm a software engineer" → {{"profession": "software engineer"}}
  User says "I live in Chennai" → {{"location": "Chennai"}}

Invalid examples (do NOT extract these):
  Tara says "you're Boss Tara" → ignore (Tara said it, not user)
  User asks "what's the capital of America" → ignore (discussing a place, not a personal fact)
  User says "and" or "it" → ignore (not a fact)

Conversation:
{excerpt}

JSON:"""

        try:
            response = self._client.chat(
                model=config.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0, "num_predict": 200},
            )
            raw = response["message"]["content"].strip()
            facts = _parse_json_safe(raw)
            for key, value in facts.items():
                if key and value and isinstance(value, str):
                    self.semantic.set(key, value)
        except Exception as e:
            log.debug("Fact extraction failed: %s", e)

    # ── task parsing ──────────────────────────────────────────────

    def parse_and_store_task(self, user_text: str) -> Optional[dict]:
        """
        Check if the user's message contains a task/reminder request.
        If so, parse it and store it. Returns the task dict or None.
        """
        if not self._client:
            return None

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        prompt = f"""Today is {now_str}.
The user said: "{user_text}"

If this contains a request to remember, remind, or schedule something, extract:
- title: short task description
- due_at: ISO datetime string (YYYY-MM-DD HH:MM) or null if no time given

Return ONLY JSON like {{"title": "...", "due_at": "..."}} or {{}} if no task.

JSON:"""

        try:
            response = self._client.chat(
                model=config.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0, "num_predict": 100},
            )
            raw = response["message"]["content"].strip()
            data = _parse_json_safe(raw)
            if data.get("title"):
                task_id = self.tasks.add(
                    title=data["title"],
                    due_at=data.get("due_at"),
                )
                task = {"id": task_id, **data}
                log.info("Task stored: %s", task)
                return task
        except Exception as e:
            log.debug("Task parsing failed: %s", e)
        return None

    # ── session end: salience scoring + episodic storage ─────────

    def end_session(self, conversation: list[dict]) -> None:
        """
        Call this when shutting down or after a long conversation gap.
        Scores the conversation for salience and stores high-value summaries.
        """
        if not self._client or len(conversation) < 2:
            return
        threading.Thread(
            target=self._end_session_worker,
            args=(conversation,),
            daemon=True,
            name="session-archiver"
        ).start()

    def _end_session_worker(self, conversation: list[dict]) -> None:
        excerpt = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}"
            for m in conversation[-20:]
        )
        prompt = f"""Read this conversation between a user and Tara (AI assistant).

1. Write a 1-2 sentence summary of what was discussed.
2. Score the importance of storing this for future reference (0-10).
   10 = very important personal info or complex topic worth remembering
   0  = small talk, simple questions with no personal relevance

Return ONLY JSON: {{"summary": "...", "score": N}}

Conversation:
{excerpt}

JSON:"""

        try:
            response = self._client.chat(
                model=config.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0, "num_predict": 200},
            )
            raw = response["message"]["content"].strip()
            data = _parse_json_safe(raw)
            summary = data.get("summary", "")
            score   = int(data.get("score", 0))
            if summary:
                self.episodic.store(summary, score=score)
        except Exception as e:
            log.debug("Session archiving failed: %s", e)

    # ── memory commands (voice commands from user) ────────────────

    def handle_memory_command(self, text: str) -> Optional[str]:
        """
        Detect and handle explicit memory commands from the user.
        Returns a response string if handled, None otherwise.

        Supported commands:
          "what do you remember about me"
          "forget my name" / "forget that"
          "what tasks do I have" / "what's on my list"
        """
        t = text.lower().strip()

        if any(p in t for p in ["what do you know about me", "what do you remember"]):
            facts = self.semantic.get_all()
            tasks = self.tasks.get_pending()
            if not facts and not tasks:
                return "I don't have anything stored about you yet."
            parts = []
            if facts:
                fact_lines = ", ".join(f"{k}: {v}" for k, v in facts.items())
                parts.append(f"I know {fact_lines}.")
            if tasks:
                task_lines = ", ".join(t["title"] for t in tasks[:3])
                parts.append(f"You have these pending tasks: {task_lines}.")
            return " ".join(parts)

        if any(p in t for p in ["what tasks", "my tasks", "my reminders", "what's on my list"]):
            pending = self.tasks.get_pending()
            if not pending:
                return "You have no pending tasks."
            lines = [f"{i+1}. {t['title']}" + (f" (due {t['due_at']})" if t["due_at"] else "")
                     for i, t in enumerate(pending[:5])]
            return "Your tasks: " + ". ".join(lines) + "."

        if t.startswith("forget "):
            key = t.replace("forget ", "").strip().rstrip(".")
            deleted = self.semantic.delete(key)
            return f"Done, I've forgotten your {key}." if deleted else f"I didn't have anything stored for '{key}'."

        return None   # not a memory command


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_json_safe(text: str) -> dict:
    """Extract and parse the first JSON object found in text."""
    import json
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group())
    except Exception:
        return {}