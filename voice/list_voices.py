"""
Run this once to see all voices available on your system:
    python list_voices.py
"""
import pyttsx3

engine = pyttsx3.init()
voices = engine.getProperty("voices")

print(f"\nFound {len(voices)} voices:\n")
for i, v in enumerate(voices):
    print(f"  [{i}] name : {v.name}")
    print(f"       id   : {v.id}")
    print(f"       lang : {getattr(v, 'languages', 'N/A')}")
    print()

print("To use a voice, set TTS_VOICE in config.py to part of its name.")
print('Example: TTS_VOICE = "Zira"  or  TTS_VOICE = "Hazel"\n')