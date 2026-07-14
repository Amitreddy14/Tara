"""
Tara state machine.

Every module reads/writes TaraState.  The orchestrator is the only
component that transitions state; subsystems treat it as read-only.
"""

from enum import Enum, auto


class TaraState(Enum):
    SLEEPING    = auto()   # wake word mode — not processing audio
    IDLE        = auto()   # waiting for speech / wake-word
    LISTENING   = auto()   # VAD has detected speech onset
    PROCESSING  = auto()   # STT running, LLM thinking
    SPEAKING    = auto()   # TTS playing audio
    INTERRUPTED = auto()   # user spoke while Tara was speaking