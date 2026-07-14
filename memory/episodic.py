"""
memory/episodic.py — Episodic memory (past conversation summaries).

At the end of each session, important conversations are summarised
and stored as vector embeddings in ChromaDB. At the start of each
new session, the most semantically relevant memories are retrieved
and injected into the system prompt.

Salience filtering: only conversations scoring >= SALIENCE_THRESHOLD
(on a 0-10 scale judged by the LLM) are stored.

Usage:
  em = EpisodicMemory()
  em.store("User asked about Python decorators. Tara explained with examples.", score=8)
  results = em.recall("decorators in Python")   # → list of relevant past summaries
  em.format_for_prompt("what are decorators")   # → string for prompt injection
"""

import logging
import os
import uuid
from datetime import datetime
from typing import Optional

import config

log = logging.getLogger(__name__)


class EpisodicMemory:
    def __init__(self):
        os.makedirs(config.CHROMA_DIR, exist_ok=True)
        self._collection = None
        self._embed_fn   = None
        self._init()

    def _init(self):
        try:
            import chromadb
            from chromadb.utils import embedding_functions

            client = chromadb.PersistentClient(path=config.CHROMA_DIR)
            self._embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=config.EMBED_MODEL
            )
            self._collection = client.get_or_create_collection(
                name="episodic",
                embedding_function=self._embed_fn,
                metadata={"hnsw:space": "cosine"},
            )
            log.info("Episodic memory ready: %d entries stored.", self._collection.count())
        except Exception as e:
            log.warning("ChromaDB unavailable — episodic memory disabled: %s", e)
            self._collection = None

    @property
    def available(self) -> bool:
        return self._collection is not None

    def store(self, summary: str, score: int, metadata: Optional[dict] = None) -> None:
        """
        Store a conversation summary if its salience score meets the threshold.
        """
        if not self.available:
            return
        if score < config.SALIENCE_THRESHOLD:
            log.debug("Episodic: score %d < threshold %d, not storing.", score, config.SALIENCE_THRESHOLD)
            return

        doc_id = str(uuid.uuid4())
        meta = {
            "stored_at": datetime.now().isoformat(),
            "score":     score,
        }
        if metadata:
            meta.update(metadata)

        self._collection.add(
            documents=[summary],
            ids=[doc_id],
            metadatas=[meta],
        )
        log.info("Episodic stored (score=%d): %r", score, summary[:80])

    def recall(self, query: str, k: Optional[int] = None) -> list[str]:
        """
        Return the top-k most semantically similar past summaries to `query`.
        """
        if not self.available or self._collection.count() == 0:
            return []

        k = k or config.EPISODIC_RECALL_K
        k = min(k, self._collection.count())

        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=k,
            )
            docs = results.get("documents", [[]])[0]
            log.debug("Episodic recalled %d memories for query %r", len(docs), query[:40])
            return docs
        except Exception as e:
            log.warning("Episodic recall failed: %s", e)
            return []

    def format_for_prompt(self, query: str) -> str:
        memories = self.recall(query)
        if not memories:
            return ""
        lines = [f"- {m}" for m in memories]
        return "Relevant things I remember from past conversations:\n" + "\n".join(lines)

    def count(self) -> int:
        return self._collection.count() if self.available else 0