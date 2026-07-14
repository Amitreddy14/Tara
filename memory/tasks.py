"""
memory/tasks.py — Task memory (reminders, todos, schedules).

Stores tasks the user asks Tara to remember. The proactive loop
(Phase 4) will read from this to trigger reminders at the right time.
For now: Tara stores tasks and can recall them when asked.

Schema:
  tasks(id, title, due_at, created_at, done, notes)

Usage:
  tm = TaskMemory()
  tm.add("Call dentist", due_at="2026-04-19 09:00")
  tm.get_pending()       # → list of upcoming undone tasks
  tm.mark_done(task_id)
  tm.format_for_prompt() # → summary string for system prompt injection
"""

import logging
import os
import sqlite3
from datetime import datetime, timedelta
from typing import Optional

import config

log = logging.getLogger(__name__)


class TaskMemory:
    def __init__(self):
        os.makedirs(config.MEMORY_DIR, exist_ok=True)
        self._conn = sqlite3.connect(config.TASKS_DB, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._setup()

    def _setup(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                title      TEXT NOT NULL,
                due_at     TEXT,
                created_at TEXT NOT NULL,
                done       INTEGER DEFAULT 0,
                notes      TEXT
            )
        """)
        self._conn.commit()
        log.debug("Task memory DB ready: %s", config.TASKS_DB)

    def add(self, title: str, due_at: Optional[str] = None, notes: Optional[str] = None) -> int:
        cur = self._conn.execute("""
            INSERT INTO tasks(title, due_at, created_at, notes)
            VALUES (?, ?, ?, ?)
        """, (title.strip(), due_at, datetime.now().isoformat(), notes))
        self._conn.commit()
        task_id = cur.lastrowid
        log.info("Task added [%d]: %r due=%s", task_id, title, due_at)
        return task_id

    def get_pending(self) -> list[dict]:
        rows = self._conn.execute("""
            SELECT * FROM tasks WHERE done = 0
            ORDER BY due_at ASC NULLS LAST, created_at ASC
        """).fetchall()
        return [dict(r) for r in rows]

    def get_due_now(self) -> list[dict]:
        """Return tasks whose due_at <= now, are not done, and were created > 60s ago."""
        now = datetime.now().isoformat()
        # created_at check prevents tasks from firing the moment they're stored
        cutoff = (datetime.now() - timedelta(seconds=90)).isoformat()
        rows = self._conn.execute("""
            SELECT * FROM tasks
            WHERE done = 0
              AND due_at IS NOT NULL
              AND due_at <= ?
              AND created_at <= ?
            ORDER BY due_at ASC
        """, (now, cutoff)).fetchall()
        return [dict(r) for r in rows]

    def mark_done(self, task_id: int) -> bool:
        cur = self._conn.execute(
            "UPDATE tasks SET done = 1 WHERE id = ?", (task_id,)
        )
        self._conn.commit()
        return cur.rowcount > 0

    def delete(self, task_id: int) -> bool:
        cur = self._conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        self._conn.commit()
        return cur.rowcount > 0

    def format_for_prompt(self) -> str:
        pending = self.get_pending()
        if not pending:
            return ""
        lines = []
        for t in pending[:5]:   # cap at 5 to avoid bloating the prompt
            due = f" (due: {t['due_at']})" if t["due_at"] else ""
            lines.append(f"- [{t['id']}] {t['title']}{due}")
        return "Pending tasks:\n" + "\n".join(lines)