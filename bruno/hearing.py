"""Reject what whisper says when nobody spoke.

Whisper does not return nothing for non-speech, it returns text. Measured on this
build: silence gives "you", room noise gives "(static)", a hiss gives
"(water running)", a click gives "[drum beats]". Always-on listening turns every
one of those into a routed request.

Three filters, cheapest first. The last one is the principled one: the VAD already
knows how much of the turn was voiced, so a transcript can be checked against
whether there was anything to transcribe.
"""

from __future__ import annotations

import re

# Whisper's own non-speech annotations. It emits these in brackets or parentheses
# rather than as speech, so a transcript made only of them contains no words.
_ANNOTATION = re.compile(r"^[\s]*([\[(][^\])]*[\])][\s]*)+$")

# Documented silence hallucinations. Whisper was trained on YouTube subtitles and
# falls back to their most common phrases when the audio carries no speech.
_GHOSTS = {
    "you", "thank you", "thank you.", "thanks for watching", "thanks for watching!",
    "please subscribe", "like and subscribe", "bye", "bye.", "okay", "okay.", "ok",
    ".", "..", "...", "!", "?", "the", "a", "so", "and", "uh", "um",
    "thank you for watching", "see you next time", "subtitles by the amara.org community",
}


def is_noise(text: str, speech_ms: float | None = None,
             min_speech_ms: float = 300.0) -> bool:
    """True when this transcript should be discarded rather than routed."""
    t = (text or "").strip()
    if not t:
        return True
    if _ANNOTATION.match(t):
        return True
    bare = t.lower().strip(" .!?,-'\"")
    # Punctuation only. Stripping it leaves nothing, which is not in the ghost set,
    # so an earlier version let "." through as a request.
    if not bare or bare in _GHOSTS:
        return True
    # A real request takes longer than a third of a second to say. If the VAD found
    # less voiced audio than that, whatever whisper produced was invented.
    if speech_ms is not None and speech_ms < min_speech_ms:
        return True
    return False


def clean(text: str) -> str:
    """Strip inline annotations from a transcript that does contain speech."""
    return re.sub(r"[\[(][^\])]{0,40}[\])]", "", text or "").strip()
