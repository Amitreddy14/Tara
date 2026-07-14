"""
voice/tts.py — Text-to-Speech output.

Backends (set TTS_ENGINE in config.py):
  "elevenlabs"  — natural human voice via API (recommended)
  "powershell"  — Windows built-in, offline, robotic but reliable
  "pyttsx3"     — auto-upgraded to powershell on Windows

ElevenLabs setup:   
  1. Sign up at elevenlabs.io (free tier: 10k chars/month)
  2. Copy your API key
  3. Set environment variable: $env:ELEVENLABS_API_KEY="your_key_here"
  4. Set TTS_ENGINE = "elevenlabs" in config.py
"""

import logging
import os
import queue
import subprocess
import threading

import config

log = logging.getLogger(__name__)


class TTS:
    def __init__(self):
        self._engine_name = config.TTS_ENGINE
        self._queue: queue.Queue[str] = queue.Queue()
        self._speaking    = False
        self._stop_flag   = threading.Event()
        self._proc        = None

        # Auto-upgrade pyttsx3 → powershell on Windows
        if self._engine_name == "pyttsx3":
            log.info("Switching pyttsx3 → PowerShell (more reliable on Windows)")
            self._engine_name = "powershell"

        # Validate ElevenLabs key at startup
        if self._engine_name == "elevenlabs":
            if not os.environ.get("ELEVENLABS_API_KEY"):
                log.warning("ELEVENLABS_API_KEY not set — falling back to PowerShell.")
                self._engine_name = "powershell"
            else:
                try:
                    from elevenlabs.client import ElevenLabs
                    self._el_client = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])
                    log.info("TTS engine: ElevenLabs (voice=%s)", config.TTS_VOICE or "Rachel")
                except ImportError:
                    log.warning("elevenlabs package not installed — pip install elevenlabs")
                    self._engine_name = "powershell"
        
        if self._engine_name != "elevenlabs":
            log.info("TTS engine: %s (voice=%s)", self._engine_name, config.TTS_VOICE or "default")

    # ── public API ────────────────────────────────────────────────

    @property
    def is_speaking(self) -> bool:
        return self._speaking

    def speak(self, text: str) -> None:
        """Queue text for playback. Returns immediately."""
        if not text or not text.strip():
            return
        log.info("TTS speaking: %r", text[:80])
        self._queue.put(text.strip())

    def run_pending(self) -> None:
        """Play one queued item. Call from the TTS player thread."""
        try:
            text = self._queue.get_nowait()
        except queue.Empty:
            return

        self._stop_flag.clear()
        self._speaking = True
        try:
            if self._engine_name == "elevenlabs":
                self._speak_elevenlabs(text)
            else:
                self._speak_powershell(text)
        except Exception as e:
            log.error("TTS error: %s", e)
            # Fallback to PowerShell if ElevenLabs fails
            if self._engine_name == "elevenlabs":
                try:
                    self._speak_powershell(text)
                except Exception:
                    pass
        finally:
            self._speaking = False

    def stop(self) -> None:
        """Kill current speech immediately."""
        self._stop_flag.set()
        self._speaking = False
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.kill()
            except Exception:
                pass
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        log.debug("TTS stopped.")

    def flush_queue(self) -> None:
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    # ── ElevenLabs backend ────────────────────────────────────────

    def _speak_elevenlabs(self, text: str) -> None:
        import pyaudio

        try:
            # Try new SDK style first (v1.0+)
            try:
                from elevenlabs import generate, stream as el_stream
                audio = generate(
                    text=text,
                    voice=config.TTS_VOICE or "Rachel",
                    model="eleven_turbo_v2",
                    api_key=os.environ.get("ELEVENLABS_API_KEY", ""),
                )
                pa = pyaudio.PyAudio()
                stream = pa.open(format=pyaudio.paInt16, channels=1, rate=22050, output=True)
                try:
                    if isinstance(audio, bytes):
                        if not self._stop_flag.is_set():
                            stream.write(audio)
                    else:
                        for chunk in audio:
                            if self._stop_flag.is_set():
                                break
                            if chunk:
                                stream.write(chunk)
                finally:
                    stream.stop_stream()
                    stream.close()
                    pa.terminate()

            except ImportError:
                # Newer SDK: use client.text_to_speech.convert()
                audio_bytes = self._el_client.text_to_speech.convert(
                    voice_id=self._get_voice_id(config.TTS_VOICE or "Rachel"),
                    text=text,
                    model_id="eleven_turbo_v2",
                    output_format="pcm_22050",
                )
                pa = pyaudio.PyAudio()
                stream = pa.open(format=pyaudio.paInt16, channels=1, rate=22050, output=True)
                try:
                    for chunk in audio_bytes:
                        if self._stop_flag.is_set():
                            break
                        if chunk:
                            stream.write(chunk)
                finally:
                    stream.stop_stream()
                    stream.close()
                    pa.terminate()

        except Exception as e:
            log.error("ElevenLabs TTS failed: %s", e)
            raise

    def _get_voice_id(self, voice_name: str) -> str:
        """Get ElevenLabs voice ID from name. Falls back to Rachel's ID."""
        known = {
            "rachel": "21m00Tcm4TlvDq8ikWAM",
            "adam":   "pNInz6obpgDQGcFmaJgB",
            "bella":  "EXAVITQu4vr4xnSDxMaL",
            "josh":   "TxGEqnHWrfWFTfGW9XjX",
        }
        return known.get(voice_name.lower(), "21m00Tcm4TlvDq8ikWAM")

    # ── PowerShell backend ────────────────────────────────────────

    def _speak_powershell(self, text: str) -> None:
        safe = text.replace("'", "''")
        voice_line = "$s.SelectVoiceByHints([System.Speech.Synthesis.VoiceGender]::Female); "
        rate = max(-10, min(10, int((config.TTS_RATE - 175) / 20)))
        ps_script = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"{voice_line}"
            f"$s.Rate = {rate}; "
            f"$s.Speak('{safe}'); "
            "$s.Dispose()"
        )
        try:
            self._proc = subprocess.Popen(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_script],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            self._proc.wait()
        except Exception as e:
            log.error("PowerShell TTS error: %s", e)
        finally:
            self._proc = None