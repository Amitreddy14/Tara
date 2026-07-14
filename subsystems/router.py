"""
subsystems/router.py — Intent classifier and subsystem dispatcher.

Classifies each user message into one of:
  - KNOWLEDGE  → web search needed (current events, facts, "what is", "who is")
  - NEWS       → news briefing requested
  - CODING     → coding / technical question
  - DIRECT     → answer directly from LLM (no subsystem needed)

Uses keyword matching first (fast), then LLM classification for ambiguous cases.
"""

import logging
import re

log = logging.getLogger(__name__)

# ── Fast keyword classifiers ──────────────────────────────────────────────────

NEWS_KEYWORDS = [
    "news", "headlines", "top stories", "briefing",
    "what's happening", "current events", "what happened today", "breaking news",
    "give me the news", "news today", "latest news",
]

KNOWLEDGE_KEYWORDS = [
    "search for", "look up", "find out", "what is", "who is", "who are",
    "when did", "when was", "where is", "how does", "tell me about",
    "what's the", "current price", "how much is", "weather in",
    "population of", "definition of", "explain what",
    "latest iphone", "latest samsung", "newest", "most recent",
    "just released", "just launched",
]

CODING_KEYWORDS = [
    "code", "function", "script", "program", "python", "javascript",
    "error", "bug", "exception", "syntax", "algorithm", "api",
    "database", "sql", "html", "css", "git", "debug", "implement",
    "class", "method", "variable", "loop", "library", "framework",
]

# Phrases that should NEVER go to web search
DIRECT_OVERRIDES = [
    "what time", "what's the time", "what day", "remind me", "remember",
    "my name", "who am i", "tasks", "how are you", "tell me a joke",
    "what can you", "start over", "reset",
    # Identity — Tara should answer these herself, not search
    "who are you", "what are you", "are you an ai", "are you human",
    "what's your name", "your name", "introduce yourself",
]


class SubsystemRouter:
    """Routes user queries to the appropriate subsystem."""

    def classify(self, text: str) -> str:
        """
        Returns one of: "knowledge", "news", "coding", "direct"
        """
        t = text.lower().strip()

        # Hard overrides — always answer directly
        if any(p in t for p in DIRECT_OVERRIDES):
            return "direct"

        # Coding detection (check before news/knowledge)
        coding_hits = sum(1 for k in CODING_KEYWORDS if k in t)
        if coding_hits >= 2 or (coding_hits >= 1 and "?" in text):
            return "coding"

        # News detection — but only if NOT about a specific product/person
        is_news_query = any(k in t for k in NEWS_KEYWORDS)
        if is_news_query:
            # If query mentions a specific product/brand, route to knowledge instead
            product_signals = [
                "iphone", "samsung", "google pixel", "android", "apple",
                "tesla", "nvidia", "openai", "chatgpt", "gemini", "meta",
                "microsoft", "amazon", "netflix", "spotify",
            ]
            if any(p in t for p in product_signals):
                return "knowledge"
            # Generic news request
            return "news"

        # Knowledge/search detection
        if any(k in t for k in KNOWLEDGE_KEYWORDS):
            if _is_simple_factual(t):
                return "direct"
            return "knowledge"

        return "direct"

    def extract_news_topic(self, text: str) -> str | None:
        """Extract a specific topic from a news request, if present."""
        t = text.lower()
        for trigger in ["news about", "headlines about", "latest on", "what's happening with"]:
            if trigger in t:
                topic = t.split(trigger)[-1].strip().rstrip("?.")
                if topic and len(topic) > 2:
                    return topic
        return None


def _is_simple_factual(text: str) -> bool:
    """
    Returns True for questions Tara can answer from training knowledge
    without needing web search.
    """
    simple_patterns = [
        r"capital of",
        r"how many (days|months|hours)",
        r"what (language|currency)",
        r"(largest|smallest|tallest|longest)",
        r"who (wrote|invented|discovered|founded)",
        r"when (was|were|did) .+ (born|founded|invented|discovered|die)",
    ]
    return any(re.search(p, text) for p in simple_patterns)