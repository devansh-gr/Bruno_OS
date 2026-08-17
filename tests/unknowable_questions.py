"""Questions with no answer in any model's weights.

The earlier grounding test was unfair and produced a wrong conclusion. It measured
retrieval on questions the 3B already knew — mitochondria, ATP, 27 bones — plus
invented terms and trick premises. Retrieval had nothing to add there and plenty
to break, so of course it scored worse.

This is the fair test: every question here is either **more recent than any
training run**, or **specific enough that memorising it is implausible**. A model
answering from weights should fail. A model with retrieval should succeed. If
grounding is worth having, this is where it shows.

Ground truth for the volatile ones is fetched at run time rather than hard-coded,
because a benchmark whose answers rot is worse than no benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class Unknowable:
    question: str
    # Returns the accepted answer fragments, fetched live where the truth moves.
    truth: Callable[[], tuple[str, ...]]
    note: str = ""


def _price(symbol: str) -> Callable[[], tuple[str, ...]]:
    """Returns a numeric target. String matching failed here: the tool says
    "305 dollars 59" and a rounded truth of "306" never matches, so a correct
    live price scored as a bluff. Compare numbers as numbers."""
    def fetch():
        from bruno.tools.public import market_quote
        r = market_quote(symbol=symbol)
        if not r.ok or not r.data:
            return ()
        return (f"~NUM~{r.data['price']}",)
    return fetch


def _headline_words(topic: str, n: int = 6) -> Callable[[], tuple[str, ...]]:
    def fetch():
        from bruno.tools.public import news
        r = news(topic=topic, limit=n)
        if not r.ok or not r.data:
            return ()
        words = set()
        for item in r.data:
            for w in item["title"].split():
                w = w.strip(".,:;'\"!?()").lower()
                if len(w) > 5:
                    words.add(w)
        return tuple(words)
    return fetch


def _static(*vals: str) -> Callable[[], tuple[str, ...]]:
    return lambda: vals


QUESTIONS: list[Unknowable] = [
    # --- moves faster than any training run ---------------------------------
    Unknowable("What is NVIDIA stock trading at right now, in dollars?",
               _price("NVDA"), "live price"),
    Unknowable("What is Apple's current share price?", _price("AAPL"), "live price"),
    Unknowable("What is Bitcoin worth right now in US dollars?",
               _price("BTC-USD"), "live price"),
    Unknowable("What is one thing in the technology news today?",
               _headline_words("tech"), "today's headlines"),
    Unknowable("What is happening in world news right now?",
               _headline_words("world"), "today's headlines"),
    Unknowable("What is one science story in the news at the moment?",
               _headline_words("science"), "today's headlines"),

    # --- obscure enough that memorising it is implausible --------------------
    # Verified against manufacturer documentation before being added here.
    Unknowable("How many GPIO pins does the ESP32-WROOM-32 module expose on its "
               "castellated edge?", _static("38", "thirty-eight"), "datasheet"),
    Unknowable("What is the maximum sampling rate of the ADS1299 biopotential "
               "front-end, in samples per second?",
               _static("16k", "16,000", "16000"), "datasheet"),
    Unknowable("What is the input impedance of the OpenBCI Cyton board's analog "
               "front end?", _static("1g", "1 g", "giga", "10^9", "1e9"), "spec"),
    Unknowable("How many degrees of freedom does the UFACTORY xArm 6 have?",
               _static("6", "six"), "product spec"),
    Unknowable("What is the maximum payload of the UFACTORY xArm 6, in kilograms?",
               _static("5", "five"), "product spec"),
    Unknowable("What resolution does the Intel RealSense D435 depth stream support "
               "at 90 frames per second?",
               _static("848", "480", "640"), "product spec"),
]


def truth_for(q: Unknowable) -> tuple[str, ...]:
    try:
        return q.truth()
    except Exception:  # noqa: BLE001 — a dead source shouldn't kill the run
        return ()
