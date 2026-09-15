"""Paths and tunables. No secrets: this file is committed."""

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
# 25-40x slower here. Bruno's actual voice is not chosen yet. This is a placeholder.
# Orpheus by default: 377ms to first audio through the real seam, which clears the
# same 500ms gate the reflex tier is held to, and it is the only backend that can
# express emotion. Piper stays available and is still what the tests synthesize
# with, since it needs no server.
TTS_BACKEND = os.environ.get("BRUNO_TTS", "orpheus")  # orpheus | piper | kokoro
# Orpheus: 382ms to first audio measured, and the only backend that can express
# emotion. Voice is one of tara/leah/jess/leo/dan/mia/zac/zoe, all American.
ORPHEUS_GGUF = MODELS / os.environ.get("BRUNO_ORPHEUS", "orpheus-3b-ft-q4_k_m.gguf")
ORPHEUS_VOICE = os.environ.get("BRUNO_ORPHEUS_VOICE", "leo")

# The full set Orpheus was trained on. The prompt offers a subset, but anything
# outside this list must be stripped: an unknown tag is not ignored, it is spoken
# as its letters.
ORPHEUS_TAGS = ("<laugh>", "<chuckle>", "<sigh>", "<gasp>", "<groan>", "<yawn>",
                "<cough>", "<sniffle>")
# Offered to the model. Deliberately narrower: a cough or a sniffle from an
# assistant is unsettling rather than expressive.
EMOTION_TAGS = ("<laugh>", "<chuckle>", "<sigh>", "<gasp>", "<groan>", "<yawn>")

EMOTION_PROMPT = (
    "You can express feeling by writing one of these tags inline, and it is spoken "
    "as the sound itself: " + " ".join(EMOTION_TAGS) + ". Use at most one per reply "
    "and only when you actually mean it. A tag in every answer reads as a tic, and "
    "a sigh you did not mean is worse than none."
)


def supports_emotion() -> bool:
    return TTS_BACKEND == "orpheus"


def system_prompt() -> str:
    """The base prompt, plus emotion only when the voice can render it."""
    return SYSTEM_PROMPT + ("\n\n" + EMOTION_PROMPT if supports_emotion() else "")
PIPER_MODEL = MODELS / "piper" / os.environ.get(
    "BRUNO_PIPER_VOICE", "en_US-lessac-medium.onnx"
)
TTS_VOICE = os.environ.get("BRUNO_VOICE", "af_heart")  # kokoro only
TTS_MODEL = MODELS / "kokoro-v1.0.onnx"
TTS_VOICES = MODELS / "voices-v1.0.bin"


def make_tts():
    if TTS_BACKEND == "orpheus":
        from bruno.backends.tts_orpheus import OrpheusTTS

        return OrpheusTTS()
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
    "If you do not know something, say so plainly rather than guessing. "
    "Never end on a one-word question or a bare monosyllable like 'Go?', 'Yes?', "
    "'Ready?' or 'Ok?'. A single short syllable is unintelligible through a speaker "
    "with no surrounding words to disambiguate it. Ask the full question instead: "
    "'Do you want me to start it?'"
)

# Enforced downstream of the model too, because a 3B will drift off any instruction.
# A synthesized monosyllable carries no context for the ear to recover, so short
# confirmations get expanded rather than spoken as-is.
BANNED_UTTERANCES = {
    "go?": "Do you want me to start it?",
    "go": "Do you want me to start it?",
    "ok?": "Does that work for you?",
    "okay?": "Does that work for you?",
    "yes?": "Is that a yes?",
    "ready?": "Do you want me to go ahead?",
    "now?": "Do you want me to do it now?",
    "good?": "Does that look right to you?",
    "right?": "Is that right?",
    "sure?": "Are you sure about that?",
}


# Cloud tier. Unset ANTHROPIC_API_KEY means the agent tier degrades to local and
# says so out loud rather than pretending to be confident.
CLOUD_MODEL = os.environ.get("BRUNO_CLOUD_MODEL", "claude-sonnet-5")
CLOUD_MAX_TOKENS = int(os.environ.get("BRUNO_CLOUD_MAX_TOKENS", "1024"))

# Target directory for `claude -p`. Still an open question: spoken each time, or
# a "current project" set verbally. Until that is decided, one env var.
CLI_TARGET_DIR = os.environ.get("BRUNO_PROJECT_DIR", "")


# Superseded by RETRIEVE_FOR_FACTS below. This flag gated the *Wikipedia*
# grounding path, which measured worse than memory because Wikipedia keyword
# search cannot disambiguate a bare question -- "What does ATP stand for?"
# returned the tennis tour. The conclusion was "grounding hurts", and it was
# wrong: the retrieval was bad, not the idea. Kept only so the old experiment
# can be reproduced.
GROUND_TUTORING = os.environ.get("BRUNO_GROUNDING", "0") == "1"

# Measured confident-wrong rate of the configured local model, worst case over
# three benchmark passes. Drives how strongly Bruno hedges a local answer.
LOCAL_CONFIDENT_WRONG_PCT = float(os.environ.get("BRUNO_CW_PCT", "6.1"))


# Factual questions retrieve before answering. Measured: SimpleQA from memory is
# 0/120 on the 3B; the answer is present in web results 7 times in 8. This is not
# an optimisation, it is what makes factual questions answerable at all.
RETRIEVE_FOR_FACTS = os.environ.get("BRUNO_RETRIEVE", "1") == "1"
