"""The whole loop, spoken. Listen, transcribe, route, answer, speak, listen again.

This is what M1's harness was a scaffold for. The differences that matter:

  - no key presses; the VAD decides when a turn began and ended
  - the clock starts at the endpoint, which is the instant a person expects a reply
  - the listener is paused while Bruno speaks

That last one is not a simplification, it is a measured necessity. On the built-in
speakers Bruno's own voice returns to the mic at peak 0.435 against a room floor of
0.019, and Silero scores it 1.000 -- 137 of 227 frames, two complete false turns.
Without pausing, he answers himself in a loop. Barge-in needs echo cancellation or
a headset, and neither is decided yet; half-duplex is what every option needs
underneath, so it is what is built.
"""

from __future__ import annotations

import time

from bruno import audio, config
from bruno.latency import Trace
from bruno.listen import Listener
from bruno.memory import Conversation
from bruno.router import Router
from bruno.vad import Endpoint


class Voice:
    def __init__(self, speak: bool = True):
        from bruno.agent import Agent
        from bruno.backends.stt_whispercpp import WhisperCppSTT
        from bruno.tools import brief, calc, email, media, public, workspace  # noqa: F401

        self.stt = WhisperCppSTT()
        self.tts = config.make_tts()
        self.router, self.agent, self.convo = Router(), Agent(), Conversation()
        self.listener = Listener(ep=Endpoint())
        self.speaker = audio.Speaker(self.tts.sample_rate) if speak else None
        self.speak = speak
        self.pending = None   # a route awaiting a spoken yes

    def warmup(self) -> None:
        t = time.perf_counter()
        self.stt.warmup()
        self.tts.warmup()
        self.router.route("warm up")
        self.listener.open()
        print(f"  ready in {time.perf_counter() - t:.1f}s")

    def _say(self, text: str) -> None:
        """Speak, with the mic decoupled for the duration. See the module docstring."""
        if not (self.speak and text):
            return
        from bruno.backends.tts_piper import chunk_text

        self.listener.paused = True
        try:
            self.speaker.start()
            for chunk in self.tts.stream(chunk_text(iter([text]))):
                self.speaker.write(chunk)
            self.speaker.finish()
            self.speaker.wait(timeout=90)
            self.speaker.stop()
            # The tail of Bruno's own audio is still in flight through the room.
            # Resuming immediately puts it in the next turn's first frames.
            time.sleep(0.25)
        finally:
            self.listener.vad.reset()
            self.listener.paused = False

    def handle(self, turn) -> None:
        tr = Trace()
        with tr.stage("stt"):
            text = self.stt.transcribe(turn.audio, config.SAMPLE_RATE)
        text = (text or "").strip()
        if not text:
            return
        print(f"\n  you    {text}   ({turn.seconds:.1f}s heard)")

        yes = text.lower().strip(" .!") in {"yes", "yeah", "yep", "go ahead", "do it", "sure"}
        if self.pending is not None and yes:
            route, confirmed = self.pending, True
        else:
            route, confirmed = self.router.route(text, self.convo), False
        self.pending = None

        with tr.stage("answer"):
            answer = self.agent.run(route, confirmed=confirmed)
        if "awaiting confirmation" in answer.notes:
            self.pending = route

        # Measured from the endpoint, not from the top of handle(): the wait a person
        # actually experiences starts the moment they stop talking.
        waited = (time.perf_counter() - turn.endpointed_at) * 1000
        print(f"  route  {route.describe()}")
        print(f"  bruno  {answer.text}")
        print(f"  took   {waited:.0f}ms to first word ({tr.total_ms:.0f}ms compute)")
        self.convo.add(text, answer.text, answer.privacy, route.tool)
        self._say(answer.text)

    def run(self) -> int:
        self.warmup()
        print("  listening. just talk. Ctrl-C to quit.\n")
        try:
            for turn in self.listener.turns():
                self.handle(turn)
        except KeyboardInterrupt:
            print()
        finally:
            self.listener.stop()
            self.listener.close()
        return 0
