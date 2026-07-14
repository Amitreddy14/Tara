"""
Tara — central configuration.
Edit this file to tune behaviour without touching any other module.
"""

# ── Audio capture ──────────────────────────────────────────────
SAMPLE_RATE = 16000
CHANNELS    = 1
CHUNK_MS    = 32
CHUNK_SIZE  = 512

# ── Voice Activity Detection ───────────────────────────────────
VAD_THRESHOLD  = 0.5
SPEECH_PAD_MS  = 300
MIN_SPEECH_MS  = 500    # ignore utterances shorter than 500ms (filters noise/single words)

# ── Speech-to-Text (Whisper) ───────────────────────────────────
WHISPER_MODEL  = "base.en"
WHISPER_DEVICE = "cpu"
WHISPER_LANG   = "en"

# ── Text-to-Speech ─────────────────────────────────────────────
TTS_ENGINE  = "elevenlabs"   # "elevenlabs" (best) | "powershell" (offline)
TTS_RATE    = 175
TTS_VOLUME  = 1.0
TTS_VOICE   = "Rachel"       # ElevenLabs voice name | "Zira" for PowerShell

# ── Persona ────────────────────────────────────────────────────
TARA_NAME   = "Tara"
WAKE_WORD   = "hey_jarvis"   # "hey_jarvis" | "alexa" | None (always-on)

# ── LLM (Ollama) ───────────────────────────────────────────────
OLLAMA_HOST     = "http://localhost:11434"
LLM_MODEL       = "llama3"   # ollama pull llama3  (better instruction following than mistral)
LLM_TEMPERATURE = 0.5        # lower = more consistent, less likely to drift from instructions
LLM_MAX_TOKENS  = 250   # memory-aware responses need more room

# ── Conversation history ────────────────────────────────────────
HISTORY_TURNS   = 10

# ── Logging ────────────────────────────────────────────────────
LOG_LEVEL   = "INFO"

# ── Memory (Phase 3) ────────────────────────────────────────────
MEMORY_DIR          = "memory_store"   # folder created next to main.py
SEMANTIC_DB         = "memory_store/semantic.db"
TASKS_DB            = "memory_store/tasks.db"
CHROMA_DIR          = "memory_store/chroma"
EMBED_MODEL         = "all-MiniLM-L6-v2"   # fast local embeddings, ~80 MB download

SALIENCE_THRESHOLD  = 6      # 0-10; only store episodic memories scoring >= this
EPISODIC_RECALL_K   = 3      # how many past memories to inject per conversation

# ── Proactive loop (Phase 4) ────────────────────────────────────
PROACTIVE_CHECK_INTERVAL = 60    # seconds between checks
PROACTIVE_MAX_PER_HOUR   = 4     # hard cap on unprompted messages per hour
PROACTIVE_MIN_GAP        = 120   # minimum seconds between any two proactive fires
PROACTIVE_MORNING_HOUR   = 8     # hour for morning greeting (24h clock)
PROACTIVE_EVENING_HOUR   = 20    # hour for evening check-in

# ── Subsystems (Phase 5) ────────────────────────────────────────
NEWS_FEEDS = [
    ("BBC World",   "http://feeds.bbci.co.uk/news/world/rss.xml"),
    ("Reuters",     "https://feeds.reuters.com/reuters/topNews"),
    ("Tech Crunch", "https://techcrunch.com/feed/"),
]
SEARCH_MAX_RESULTS = 4      # DuckDuckGo results to fetch per query