"""Paths and tunables. No secrets — this file is committed."""

from __future__ import annotations

import os
import platform
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
LOGS = ROOT / "logs"

IS_MAC = platform.system() == "Darwin"
IS_WINDOWS = platform.system() == "Windows"

# Audio. 16 kHz mono is what both Whisper and Silero VAD expect.
SAMPLE_RATE = 16_000
BLOCK_MS = 32
BLOCK_SAMPLES = SAMPLE_RATE * BLOCK_MS // 1000

# Local servers. Persistent so model load never lands inside a turn.
LLAMA_HOST, LLAMA_PORT = "127.0.0.1", 8081
WHISPER_HOST, WHISPER_PORT = "127.0.0.1", 8082

LLM_GGUF = MODELS / os.environ.get("BRUNO_LLM", "Llama-3.2-3B-Instruct-Q4_K_M.gguf")
STT_GGML = MODELS / os.environ.get("BRUNO_STT", "ggml-base.en.bin")
# Truncated Whisper encoder context. 512 frames ~= 10s of audio and is the single
# biggest latency lever measured: base.en 633ms -> 38ms. 0 disables truncation.
STT_AUDIO_CTX = int(os.environ.get("BRUNO_STT_AUDIO_CTX", "512"))

# TTS. Piper is the default on measured latency; Kokoro sounds better but is
# 25-40x slower here. Bruno's actual voice is not chosen yet — this is a placeholder.
TTS_BACKEND = os.environ.get("BRUNO_TTS", "piper")  # piper | kokoro
PIPER_MODEL = MODELS / "piper" / os.environ.get(
    "BRUNO_PIPER_VOICE", "en_US-lessac-medium.onnx"
)
TTS_VOICE = os.environ.get("BRUNO_VOICE", "af_heart")  # kokoro only
TTS_MODEL = MODELS / "kokoro-v1.0.onnx"
TTS_VOICES = MODELS / "voices-v1.0.bin"


def make_tts():
    if TTS_BACKEND == "kokoro":
        from bruno.backends.tts_kokoro import KokoroTTS

        return KokoroTTS()
    from bruno.backends.tts_piper import PiperTTS

    return PiperTTS()

# Generation. Short and blunt: long answers blow the latency budget and are off-register.
MAX_TOKENS = 200
TEMPERATURE = 0.7

SYSTEM_PROMPT = (
    "You are Bruno, Devansh's local voice assistant. You are being spoken to out loud "
    "and your reply is read aloud, so answer in one or two short sentences. "
    "Be blunt and technical. No filler, no preamble, no bullet lists, no markdown. "
    "If you do not know something, say so plainly rather than guessing."
)
