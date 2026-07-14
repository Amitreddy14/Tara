"""
core/config_validator.py — Validates config.py at startup.

Catches common misconfigurations before they cause cryptic errors deep
in the pipeline. Prints a clear, actionable error message and exits if
any critical setting is invalid.

Usage:
    from core.config_validator import validate
    validate()   # call before anything else in main.py
"""

import logging
import os
import sys

log = logging.getLogger(__name__)


def validate() -> None:
    """
    Validate all config settings. Exits with a clear message if anything
    critical is wrong. Logs warnings for non-critical issues.
    """
    import config

    errors   = []
    warnings = []

    # ── Audio ──
    if config.SAMPLE_RATE not in (16000, 8000):
        errors.append(f"SAMPLE_RATE must be 16000 (got {config.SAMPLE_RATE})")
    if config.CHUNK_SIZE < 512:
        errors.append(f"CHUNK_SIZE must be >= 512 for Silero VAD (got {config.CHUNK_SIZE})")
    if not 0.0 <= config.VAD_THRESHOLD <= 1.0:
        errors.append(f"VAD_THRESHOLD must be 0.0-1.0 (got {config.VAD_THRESHOLD})")

    # ── STT ──
    valid_models = {"tiny.en", "base.en", "small.en", "medium.en", "large-v2", "large-v3"}
    if config.WHISPER_MODEL not in valid_models:
        warnings.append(f"WHISPER_MODEL '{config.WHISPER_MODEL}' is not a standard model name.")
    if config.WHISPER_DEVICE not in ("cpu", "cuda", "auto"):
        errors.append(f"WHISPER_DEVICE must be 'cpu', 'cuda', or 'auto' (got {config.WHISPER_DEVICE})")

    # ── TTS ──
    if config.TTS_ENGINE not in ("pyttsx3", "powershell", "elevenlabs"):
        errors.append(f"TTS_ENGINE must be 'pyttsx3', 'powershell', or 'elevenlabs'")
    if config.TTS_ENGINE == "elevenlabs" and not os.environ.get("ELEVENLABS_API_KEY"):
        errors.append("TTS_ENGINE is 'elevenlabs' but ELEVENLABS_API_KEY is not set.")
    if not 0.0 <= config.TTS_VOLUME <= 1.0:
        errors.append(f"TTS_VOLUME must be 0.0-1.0 (got {config.TTS_VOLUME})")

    # ── LLM ──
    if not config.OLLAMA_HOST.startswith("http"):
        errors.append(f"OLLAMA_HOST must be a URL (got {config.OLLAMA_HOST})")
    if config.LLM_MAX_TOKENS < 50:
        warnings.append(f"LLM_MAX_TOKENS={config.LLM_MAX_TOKENS} is very low — responses may be truncated.")
    if config.LLM_MAX_TOKENS > 1000:
        warnings.append(f"LLM_MAX_TOKENS={config.LLM_MAX_TOKENS} is high — voice responses will be very long.")

    # ── Memory ──
    memory_dir = config.MEMORY_DIR
    if memory_dir and not os.path.exists(memory_dir):
        try:
            os.makedirs(memory_dir, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create MEMORY_DIR '{memory_dir}': {e}")

    # ── Logging ──
    valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    if config.LOG_LEVEL not in valid_levels:
        errors.append(f"LOG_LEVEL must be one of {valid_levels} (got {config.LOG_LEVEL})")

    # ── Report ──
    for w in warnings:
        log.warning("Config warning: %s", w)

    if errors:
        print("\n" + "="*60)
        print("TARA CONFIG ERROR — Cannot start:")
        for e in errors:
            print(f"  ✗ {e}")
        print("\nEdit config.py to fix these issues.")
        print("="*60 + "\n")
        sys.exit(1)

    log.debug("Config validation passed (%d warnings).", len(warnings))