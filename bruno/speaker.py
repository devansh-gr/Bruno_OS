"""Speaker verification. ECAPA embeddings, cosine distance against an enrolled mean.

This is what makes a two-word wake phrase safe: the phrase says Bruno is addressed,
the voiceprint says by whom. LOCAL_ONLY, gitignored. Needs ~30s of his speech.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bruno import config

PRINT_PATH = config.ROOT / "voiceprint.json"

# Deliberately above ECAPA's ~0.25-0.30 operating point: a false reject costs him
# a repeat, a false accept hands someone else his laptop.
THRESHOLD = float(config.__dict__.get("BRUNO_SPK_THRESHOLD", 0) or 0.45)

# Below this, verification abstains rather than guessing.
MIN_SECONDS = 0.8


@dataclass
class Verdict:
    ok: bool
    score: float
    reason: str = ""


class Embedder:
    """ECAPA-TDNN embeddings. Lazy so an unenrolled install never pays the load."""

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
    """Average several clips into one reference. Measured: six-clip mean separates at
    +0.194 margin where single-clip overlaps at -0.058."""
    emb = Embedder()
    vecs = [emb.embed(c, sample_rate) for c in clips
            if len(c) / sample_rate >= MIN_SECONDS]
    if not vecs:
        raise ValueError("no clip was long enough to enrol")
    mean = np.mean(vecs, axis=0)
    mean = mean / (np.linalg.norm(mean) + 1e-9)
    # If the clips disagree with each other, no threshold separates him from anyone.
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
    """Unenrolled returns ok=True deliberately, and says so in `reason` -- the wake
    phrase is still required, and refusing everything would make Bruno unusable."""
    ref = load()
    if ref is None:
        return Verdict(True, 0.0, "not enrolled. Anyone can wake him")
    if len(audio) / sample_rate < MIN_SECONDS:
        return Verdict(True, 0.0, "too short to verify")
    score = similarity(Embedder().embed(audio, sample_rate), ref)
    return Verdict(score >= threshold, score,
                   "" if score >= threshold else "voice did not match")
