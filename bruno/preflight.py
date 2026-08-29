"""Check the machine can actually run Bruno, before a turn depends on it.

Every one of these has failed silently at least once: a missing GGUF surfaced as a
llama-server timeout, a revoked microphone permission surfaced as a stream that
blocks forever, and an unset voice surfaced as Bruno saying the voice name aloud.
Failing at startup with the reason is cheaper than any of those.
"""

from __future__ import annotations

from dataclasses import dataclass

from bruno import config


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""
    fatal: bool = True


def run(*, need_mic: bool = True) -> list[Check]:
    out: list[Check] = []

    out.append(Check("llm weights", config.LLM_GGUF.exists(), str(config.LLM_GGUF)))
    out.append(Check("stt weights", config.STT_GGML.exists(), str(config.STT_GGML)))

    if config.TTS_BACKEND == "orpheus":
        out.append(Check("orpheus weights", config.ORPHEUS_GGUF.exists(),
                         str(config.ORPHEUS_GGUF)))
        from bruno.backends.tts_orpheus import VOICES
        out.append(Check("orpheus voice", config.ORPHEUS_VOICE in VOICES,
                         f"{config.ORPHEUS_VOICE!r} not in {', '.join(VOICES)}"))
    else:
        out.append(Check("piper voice", config.PIPER_MODEL.exists(),
                         str(config.PIPER_MODEL)))

    import shutil
    out.append(Check("llama-server on PATH", shutil.which("llama-server") is not None,
                     "brew install llama.cpp"))

    vad = config.MODELS / "silero_vad.onnx"
    out.append(Check("vad model", vad.exists(), str(vad)))

    if need_mic:
        from bruno import audio
        problem = audio.check_mic()
        out.append(Check("microphone", problem is None, problem or ""))

    from bruno import speaker
    # Not fatal: Bruno runs unenrolled and says so, rather than refusing to start.
    out.append(Check("voiceprint", speaker.load() is not None,
                     "run python enroll.py — anyone can wake him until then",
                     fatal=False))
    return out


def report(checks: list[Check]) -> bool:
    """Print the table. Returns False if anything fatal failed."""
    bad = [c for c in checks if not c.ok and c.fatal]
    for c in checks:
        mark = "ok  " if c.ok else ("FAIL" if c.fatal else "warn")
        print(f"  {mark}  {c.name:22} {'' if c.ok else c.detail}")
    return not bad
