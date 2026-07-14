"""
subsystems/coding.py — Coding assistant subsystem.

For simple code questions: uses local llama3 (fast).
For complex tasks (debug, architect, explain large snippets): routes
to Claude claude-sonnet-4-20250514 via Anthropic API for higher quality.

Requires ANTHROPIC_API_KEY env var for complex mode.
Falls back to local model if API key not set.
"""

import logging
import os
import re

log = logging.getLogger(__name__)

# Complexity signals — if any appear, route to API model
COMPLEX_SIGNALS = [
    "debug", "fix this", "why isn't", "why is this", "not working",
    "refactor", "architect", "design", "optimize", "review my code",
    "explain this code", "what does this do", "write a", "build a",
    "implement", "algorithm",
]


class CodingSubsystem:
    def __init__(self, ollama_client=None, local_model: str = "llama3"):
        self._ollama  = ollama_client
        self._local_model = local_model
        self._api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        self._api_client = None

        if self._api_key:
            try:
                import anthropic
                self._api_client = anthropic.Anthropic(api_key=self._api_key)
                log.info("Coding subsystem: Claude API available for complex tasks.")
            except ImportError:
                log.warning("anthropic package not installed — pip install anthropic")
        else:
            log.info("Coding subsystem: local model only (set ANTHROPIC_API_KEY for Claude).")

    def answer(self, query: str) -> str:
        """
        Answer a coding question. Routes to API or local based on complexity.
        Always returns a plain-text spoken answer (no markdown for voice).
        """
        is_complex = any(sig in query.lower() for sig in COMPLEX_SIGNALS)

        if is_complex and self._api_client:
            log.info("Coding: routing to Claude API (complex query)")
            return self._ask_claude(query)
        else:
            log.info("Coding: routing to local model")
            return self._ask_local(query)

    def _ask_local(self, query: str) -> str:
        if not self._ollama:
            return "I don't have a language model available right now."

        prompt = f"""Answer this coding question concisely in plain spoken English.
No markdown, no code blocks for voice — describe the solution verbally.
If showing code is essential, keep it to one short snippet and explain it.

Question: {query}"""

        try:
            response = self._ollama.chat(
                model=self._local_model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.2, "num_predict": 300},
            )
            return _clean(response["message"]["content"].strip())
        except Exception as e:
            log.error("Local coding LLM failed: %s", e)
            return "I ran into an issue answering that. Could you try rephrasing?"

    def _ask_claude(self, query: str) -> str:
        try:
            message = self._api_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=400,
                messages=[{
                    "role": "user",
                    "content": (
                        f"Answer this coding question for a voice assistant. "
                        f"Use plain English — no markdown, no bullet points. "
                        f"Be concise (2-4 sentences for voice). "
                        f"If you must show code, keep it to one short line.\n\n{query}"
                    )
                }]
            )
            return _clean(message.content[0].text.strip())
        except Exception as e:
            log.error("Claude API coding failed: %s", e)
            return self._ask_local(query)   # graceful fallback


def _clean(text: str) -> str:
    """Strip markdown for voice output."""
    text = re.sub(r"```[\s\S]*?```", "[code block]", text)
    text = re.sub(r"`(.+?)`", r"\1", text)
    text = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*•]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n+", " ", text)
    return text.strip()