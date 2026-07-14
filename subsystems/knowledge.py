"""
subsystems/knowledge.py — Web search subsystem.

Uses DuckDuckGo (free, no API key) to search the web and summarise
results into a spoken answer.

The orchestrator calls search() when it detects a knowledge query
that requires current or specific factual information.
"""

import logging
import re

log = logging.getLogger(__name__)


class KnowledgeSubsystem:
    def __init__(self, ollama_client=None, model: str = "llama3"):
        self._client = ollama_client
        self._model  = model

    def search(self, query: str, max_results: int = 4) -> str:
        """
        Search DuckDuckGo for query, summarise results into a
        1-2 sentence spoken answer. Returns summary string.
        """
        log.info("Knowledge search: %r", query)
        try:
            try:
                from ddgs import DDGS          # new package name
            except ImportError:
                from duckduckgo_search import DDGS   # old name fallback
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
        except Exception as e:
            log.error("DuckDuckGo search failed: %s", e)
            return ""

        if not results:
            return ""

        # Build a context block from search snippets
        context = "\n".join(
            f"- {r.get('title','')}: {r.get('body','')}"
            for r in results
        )

        if not self._client:
            # No LLM — return the top snippet directly
            top = results[0].get("body", "")
            return _clean(top[:300])

        # Ask LLM to synthesise a spoken answer from snippets
        prompt = f"""Based only on the search results below, give a concise 1-2 sentence spoken answer to: "{query}"
Do not use bullet points or markdown. Speak naturally. If results don't answer the question, say so.

Search results:
{context}

Answer:"""

        try:
            response = self._client.chat(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.1, "num_predict": 150},
            )
            return _clean(response["message"]["content"].strip())
        except Exception as e:
            log.error("Knowledge summarisation failed: %s", e)
            top = results[0].get("body", "")
            return _clean(top[:300])


def _clean(text: str) -> str:
    text = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", text)
    text = re.sub(r"\n+", " ", text)
    return text.strip()