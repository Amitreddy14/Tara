"""
subsystems/news.py — News briefing subsystem.

Pulls headlines from RSS feeds and delivers a short spoken briefing.
No API key needed — uses public RSS feeds.

Feeds are configurable in config.py (NEWS_FEEDS list).
"""

import logging
import re
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger(__name__)

DEFAULT_FEEDS = [
    ("BBC World",    "http://feeds.bbci.co.uk/news/world/rss.xml"),
    ("Reuters",      "https://feeds.reuters.com/reuters/topNews"),
    ("Tech Crunch",  "https://techcrunch.com/feed/"),
]


class NewsSubsystem:
    def __init__(self, ollama_client=None, model: str = "llama3"):
        self._client = ollama_client
        self._model  = model

    def get_briefing(self, topic: Optional[str] = None, max_items: int = 5) -> str:
        """
        Fetch headlines and return a spoken news briefing.
        If topic is given, filter headlines to that topic.
        """
        log.info("News briefing request (topic=%r)", topic)
        headlines = self._fetch_headlines(max_items * 3)   # fetch more, filter down

        if not headlines:
            return "I couldn't fetch the news right now. Please check your internet connection."

        if topic:
            filtered = [h for h in headlines if topic.lower() in h.lower()]
            if filtered:
                headlines = filtered[:max_items]
            else:
                headlines = headlines[:max_items]
        else:
            headlines = headlines[:max_items]

        if not self._client:
            items = ". ".join(f"{i+1}. {h}" for i, h in enumerate(headlines))
            return f"Here are the latest headlines: {items}."

        items_text = "\n".join(f"- {h}" for h in headlines)
        prompt = f"""Turn these news headlines into a natural, spoken 2-3 sentence briefing.
No bullet points, no markdown. Write as if speaking aloud.
{"Focus on: " + topic if topic else "Cover the most important stories."}

Headlines:
{items_text}

Briefing:"""

        try:
            response = self._client.chat(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.3, "num_predict": 200},
            )
            return _clean(response["message"]["content"].strip())
        except Exception as e:
            log.error("News summarisation failed: %s", e)
            return "Here are the top headlines: " + ". ".join(headlines[:3]) + "."

    def _fetch_headlines(self, limit: int = 15) -> list[str]:
        try:
            import feedparser
        except ImportError:
            log.error("feedparser not installed — pip install feedparser")
            return []

        import config
        feeds = getattr(config, "NEWS_FEEDS", DEFAULT_FEEDS)
        headlines = []

        for name, url in feeds:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:limit // len(feeds) + 2]:
                    title = entry.get("title", "").strip()
                    if title and len(title) > 10:
                        headlines.append(title)
            except Exception as e:
                log.warning("Feed %r failed: %s", name, e)

        return headlines[:limit]


def _clean(text: str) -> str:
    text = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", text)
    text = re.sub(r"\n+", " ", text)
    return text.strip()