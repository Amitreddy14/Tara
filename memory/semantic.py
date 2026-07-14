"""
memory/semantic.py — Semantic memory (stable user facts).

Stores key facts about the user that persist across sessions:
  - name, location, preferences, profession, etc.
  - Any fact Tara learns and the user hasn't asked to forget.

Schema:
  facts(key TEXT PRIMARY KEY, value TEXT, updated_at TIMESTAMP)

Usage:
  sm = SemanticMemory()
  sm.set("name", "Amit")
  sm.get("name")          # → "Amit"
  sm.get_all()            # → {"name": "Amit", ...}
  sm.delete("name")
  sm.format_for_prompt()  # → "What I know about you:\n- name: Amit\n..."
"""

import logging
import os
import sqlite3
from datetime import datetime

import config

log = logging.getLogger(__name__)


class SemanticMemory:
    def __init__(self):
        os.makedirs(config.MEMORY_DIR, exist_ok=True)
        self._conn = sqlite3.connect(config.SEMANTIC_DB, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._setup()

    def _setup(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                key        TEXT PRIMARY KEY,
                value      TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        self._conn.commit()
        log.debug("Semantic memory DB ready: %s", config.SEMANTIC_DB)

    def set(self, key: str, value: str) -> None:
        key = key.lower().strip()
        self._conn.execute("""
            INSERT INTO facts(key, value, updated_at) VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
        """, (key, value.strip(), datetime.now().isoformat()))
        self._conn.commit()
        log.info("Semantic memory set: %s = %r", key, value)

    def get(self, key: str) -> str | None:
        row = self._conn.execute(
            "SELECT value FROM facts WHERE key = ?", (key.lower().strip(),)
        ).fetchone()
        return row["value"] if row else None

    def get_all(self) -> dict[str, str]:
        rows = self._conn.execute("SELECT key, value FROM facts ORDER BY key").fetchall()
        return {r["key"]: r["value"] for r in rows}

    def delete(self, key: str) -> bool:
        cur = self._conn.execute("DELETE FROM facts WHERE key = ?", (key.lower().strip(),))
        self._conn.commit()
        deleted = cur.rowcount > 0
        if deleted:
            log.info("Semantic memory deleted: %s", key)
        return deleted

    def clear_all(self) -> None:
        self._conn.execute("DELETE FROM facts")
        self._conn.commit()
        log.info("Semantic memory cleared.")

    def format_for_prompt(self) -> str:
        facts = self.get_all()
        if not facts:
            return ""
        lines = [f"- {k}: {v}" for k, v in facts.items()]
        return "What I know about you:\n" + "\n".join(lines)