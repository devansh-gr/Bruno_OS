"""Speaker verification: is this Devansh?

The wake phrase is two words, which on its own would fire for anyone in the room
saying them -- including a friend actually named Bruno. This is what makes the
short phrase safe: the phrase establishes that Bruno is being addressed, the
voiceprint establishes by whom.

An embedding model turns a few seconds of speech into a fixed-length vector, and
two recordings of the same person land near each other by cosine distance while
different people do not. Enrollment averages several utterances into one reference
vector; verification compares a new one against it.

Strictly LOCAL_ONLY. His voiceprint never leaves the machine and is gitignored.

Not yet enrolled: this needs about 30 seconds of his speech, which is the one part
of the voice layer that cannot be built without him. Everything else here runs and
is tested against synthetic voices standing in for different speakers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bruno import config

PRINT_PATH = config.ROOT / "voiceprint.json"

# Cosine similarity above which a sample is accepted as him. ECAPA-TDNN's
# conventional operating point on clean speech is around 0.25-0.30 EER-balanced;
# this sits deliberately higher because the cost is asymmetric. A false reject
# means saying it twice. A false accept means someone else drives his laptop.
THRESHOLD = float(config.__dict__.get("BRUNO_SPK_THRESHOLD", 0) or 0.45)

# Below this there is not enough voiced audio to embed reliably; verification
# abstains rather than guessing.
MIN_SECONDS = 0.8


@dataclass
class Verdict:
    ok: bool
    score: float
    reason: str = ""


class Embedder:
    """ECAPA-TDNN speaker embeddings.

    Loaded lazily and cached: it is only touched on a wake attempt, and paying the
    model load at import would sit inside Bruno's startup for a feature that may
    never be enrolled.
    """

    _model = None

    def _load(self):
        if Embedder._model is None:
            from speechbrain.inference.speaker import EncoderClassifier

            Embedder._model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=str(config.MODELS / "ecapa"),
                run_opts={"device": "cpu"},
            )
        return Embedder._model

    def embed(self, audio: np.ndarray, sample_rate: int = config.SAMPLE_RATE) -> np.ndarray:
        import torch

        if sample_rate != 16_000:
            n = int(len(audio) * 16_000 / sample_rate)
            audio = np.interp(np.linspace(0, len(audio) - 1, n),
                              np.arange(len(audio)), audio).astype(np.float32)
        wav = torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0)
        with torch.no_grad():
            e = self._load().encode_batch(wav).squeeze().cpu().numpy()
        return e / (np.linalg.norm(e) + 1e-9)


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9))


def enroll(clips: list[np.ndarray], sample_rate: int = config.SAMPLE_RATE) -> dict:
    """Average several utterances into one reference vector.

    Several, not one: a single clip bakes in whatever that sentence's phonemes and
    that moment's mic position were, and verification then fails on a different
    sentence spoken from a different angle.
    """
    emb = Embedder()
    vecs = [emb.embed(c, sample_rate) for c in clips
            if len(c) / sample_rate >= MIN_SECONDS]
    if not vecs:
        raise ValueError("no clip was long enough to enrol")
    mean = np.mean(vecs, axis=0)
    mean = mean / (np.linalg.norm(mean) + 1e-9)
    # Self-similarity across the enrolment clips is the honest read on how
    # consistent the voiceprint is. If these disagree with each other, no
    # threshold will separate him from anyone else.
    spread = [similarity(v, mean) for v in vecs]
    rec = {
        "vector": mean.tolist(),
        "clips": len(vecs),
        "self_similarity_min": float(min(spread)),
        "self_similarity_mean": float(float(np.mean(spread))),
    }
    PRINT_PATH.write_text(json.dumps(rec))
    return rec


def load() -> np.ndarray | None:
    if not PRINT_PATH.exists():
        return None
    try:
        return np.array(json.loads(PRINT_PATH.read_text())["vector"], dtype=np.float32)
    except (json.JSONDecodeError, KeyError, ValueError):
        return None


def verify(audio: np.ndarray, sample_rate: int = config.SAMPLE_RATE,
           threshold: float = THRESHOLD) -> Verdict:
    """Is this him?

    Unenrolled returns ok=True. That is deliberate: until there is a voiceprint,
    refusing every request would make the assistant unusable, and the wake phrase
    is still required. The caller is told which mode it is in via `reason`.
    """
    ref = load()
    if ref is None:
        return Verdict(True, 0.0, "not enrolled — anyone can wake him")
    if len(audio) / sample_rate < MIN_SECONDS:
        return Verdict(True, 0.0, "too short to verify")
    score = similarity(Embedder().embed(audio, sample_rate), ref)
    return Verdict(score >= threshold, score,
                   "" if score >= threshold else "voice did not match")
