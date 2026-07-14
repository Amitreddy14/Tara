# TARA — Tactical AI Response Agent

> A persistent, voice-first AI assistant built from scratch. Hears you, thinks with a local LLM, remembers you across sessions, searches the web, manages tasks, opens apps, and speaks back in a natural voice — all running locally on Windows.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Ollama](https://img.shields.io/badge/LLM-Ollama%20llama3-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)

---

## What Tara can do

| Capability | Details |
|---|---|
|  Voice I/O | Always-on mic with Silero VAD, Whisper STT, ElevenLabs / PowerShell TTS |
|  Local LLM | llama3 via Ollama — no cloud required |
|  Persistent memory | Remembers your name, preferences, and past conversations across restarts |
|  Proactive reminders | Speaks without being asked — morning greetings, task reminders |
|  Web search | DuckDuckGo search + RSS news briefings |
|  System execution | Opens apps, manages files — with mandatory confirmation gate |
|  Wake word | "Hey Jarvis" activates Tara from sleep |
|  Jarvis UI | Iron Man-style dashboard at `http://localhost:8000` |

---

## Architecture

```
Voice Input (Whisper STT)
        ↓
Central Orchestrator (llama3)
        ↓
   ┌────┴────┐
Router    Memory
   ↓         ↓
Knowledge  Semantic (SQLite)
News       Episodic (ChromaDB)
Coding     Tasks (SQLite)
Execution
        ↓
Voice Output (ElevenLabs / PowerShell TTS)
        ↓
Jarvis UI (FastAPI + WebSocket + React)
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- Windows (PowerShell TTS fallback) or ElevenLabs API key

### 1. Clone and set up

```bash
git clone https://github.com/Amitreddy14/Tara.git
cd Tara
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Pull the LLM

```bash
ollama pull llama3
```

### 3. Set environment variables

```powershell
$env:ELEVENLABS_API_KEY="your_key_here"   # optional — PowerShell TTS works without it
```

### 4. Run

```powershell
python main.py
```

Open **http://localhost:8000** for the Tara UI.

---

## Configuration

All settings live in `config.py`:

```python
WHISPER_MODEL  = "small.en"      # tiny.en | base.en | small.en | medium.en
LLM_MODEL      = "llama3"        # any model pulled via ollama
TTS_ENGINE     = "elevenlabs"    # "elevenlabs" | "powershell"
TTS_VOICE      = "Rachel"        # ElevenLabs voice name
WAKE_WORD      = "hey_jarvis"    # None = always-on
VAD_THRESHOLD  = 0.6             # raise to reduce false triggers
```

---

## Project Structure

```
tara/
├── main.py                  # Entry point
├── server.py                # FastAPI + WebSocket UI server
├── config.py                # All settings
├── requirements.txt
├── core/
│   ├── orchestrator.py      # Central brain — routes all decisions
│   ├── proactive.py         # Background scheduler for unprompted messages
│   ├── health.py            # Watchdog monitor
│   ├── state.py             # State machine
│   └── config_validator.py  # Startup validation
├── voice/
│   ├── microphone.py        # Audio capture
│   ├── vad.py               # Silero voice activity detection
│   ├── stt.py               # Whisper speech-to-text
│   ├── tts.py               # ElevenLabs / PowerShell TTS
│   └── wakeword.py          # Wake word detection
├── memory/
│   ├── manager.py           # Unified memory interface
│   ├── semantic.py          # User facts (SQLite)
│   ├── episodic.py          # Conversation summaries (ChromaDB)
│   └── tasks.py             # Reminders and tasks (SQLite)
├── subsystems/
│   ├── router.py            # Intent classifier
│   ├── knowledge.py         # DuckDuckGo web search
│   ├── news.py              # RSS news briefings
│   ├── coding.py            # Coding assistant
│   └── execution.py         # OS actions with confirmation gate
└── ui/
    └── index.html           # Jarvis-style dashboard
```

---

## Voice Commands

| You say | Tara does |
|---|---|
| "My name is Amit" | Stores your name permanently |
| "Remind me to call dentist at 9am tomorrow" | Creates a task with due time |
| "What do you remember about me?" | Reads back stored facts |
| "What tasks do I have?" | Lists pending reminders |
| "What's the latest news?" | Fetches RSS headlines |
| "Search for Python 3.13 features" | DuckDuckGo web search |
| "Open notepad" | Asks for confirmation, then opens |
| "Start over" | Clears conversation history |
| "Hey Jarvis" | Wakes Tara from sleep |

---

## Memory System

Tara uses three memory layers:

- **Semantic** — stable facts (name, location, preferences) stored in SQLite, never expire
- **Episodic** — conversation summaries stored in ChromaDB with vector search, retrieved by relevance
- **Task** — reminders with due times, checked every 60 seconds by the proactive loop

The salience filter scores each conversation 0–10 before storing — only important sessions are archived.

---

## Safety

The execution system has a hardcoded confirmation gate — Tara always asks before performing any file or system action. System directories (`C:\Windows`, `C:\Program Files`) and dangerous file types (`.exe`, `.dll`, `.sys`) are permanently blocked regardless of user instruction. Every action is logged to `memory_store/execution_audit.log`.

---

## Built With

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — local STT
- [Ollama](https://ollama.com) — local LLM runtime
- [Silero VAD](https://github.com/snakers4/silero-vad) — voice activity detection
- [ElevenLabs](https://elevenlabs.io) — natural TTS
- [ChromaDB](https://www.trychroma.com) — vector memory
- [DuckDuckGo Search](https://pypi.org/project/ddgs/) — web search
- [openwakeword](https://github.com/dscripka/openWakeWord) — wake word detection
- [FastAPI](https://fastapi.tiangolo.com) — UI server

---

## Roadmap

- [ ] Google Calendar integration
- [ ] Streaming TTS for lower latency
- [ ] Custom "Hey Tara" wake word training
- [ ] Mobile interface
- [ ] Claude API for complex reasoning tasks

---

## License

MIT — do whatever you want with it.

---

*Built by [Amit Reddy](https://github.com/Amitreddy14)*
