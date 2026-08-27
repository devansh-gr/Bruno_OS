"""Listen, transcribe, route, answer, speak, listen again.

Duplex mode is chosen from the output device, not assumed. Through headphones
nothing reaches the mic (measured 0.06) so the mic stays live and he can be cut off
mid-sentence. On speakers his own voice returns at 0.435 and Silero scores it 1.000,
so the mic is decoupled while he talks. See aec.py for cancelling that instead.
"""

from __future__ import annotations

import time

from bruno import audio, config, speaker, wake
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
        # Asleep until addressed. Awake, the bare name is enough and usually not
        # needed at all -- this is the difference between talking to someone and
        # re-summoning a phone assistant before every sentence.
        self.awake = False
        self.awake_until = 0.0
        self.full_duplex = audio.output_is_headphones()

    def warmup(self) -> None:
        t = time.perf_counter()
        self.stt.warmup()
        self.tts.warmup()
        self.router.route("warm up")
        self.listener.open()
        print(f"  ready in {time.perf_counter() - t:.1f}s")

    def _say(self, text: str) -> None:
        """Speak with the mic decoupled."""
        if not (self.speak and text):
            return
        from bruno.backends.tts_piper import chunk_text

        # Only decouple when the room would carry his voice back to the mic.
        self.listener.paused = not self.full_duplex
        try:
            self.speaker.start()
            for chunk in self.tts.stream(chunk_text(iter([text]))):
                self.speaker.write(chunk)
            self.speaker.finish()
            self.speaker.wait(timeout=90)
            self.speaker.stop()
            # His own tail is still crossing the room.
            if not self.full_duplex:
                time.sleep(0.25)   # his own tail is still crossing the room
        finally:
            if not self.full_duplex:
                self.listener.vad.reset()
            self.listener.paused = False

    # Long enough for a follow-up, short enough that the next room does not reach him.
    STAY_AWAKE_S = 45.0

    def _admit(self, text: str, turn) -> str | None:
        """The request if this turn is for Bruno, else None -- which is most turns."""
        now = time.perf_counter()
        if self.awake and now > self.awake_until:
            self.awake = False

        addressed = wake.wakes(text, strict=not self.awake)
        if not (self.awake or addressed):
            return None

        # Only on the waking turn; the conversation window is short.
        if addressed and not self.awake:
            v = speaker.verify(turn.audio, config.SAMPLE_RATE)
            if not v.ok:
                print(f"  ignored  wake phrase heard but {v.reason} ({v.score:.2f})")
                return None
            if v.reason:
                print(f"  note   {v.reason}")

        if wake.sleeps(text):
            self.awake = False
            print("  bruno  (going quiet)")
            self._say("Alright.")
            return None

        self.awake = True
        self.awake_until = now + self.STAY_AWAKE_S
        req = wake.strip_wake(text)
        # "Hey Bruno" with nothing after it is a summons, not a request.
        return req or ""

    def handle(self, turn) -> None:
        tr = Trace()
        with tr.stage("stt"):
            text = self.stt.transcribe(turn.audio, config.SAMPLE_RATE)
        text = (text or "").strip()
        if not text:
            return

        req = self._admit(text, turn)
        if req is None:
            return
        print(f"\n  you    {text}   ({turn.seconds:.1f}s heard)")
        if req == "":
            self._say("Yeah?")
            return
        text = req

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

        # From the endpoint: that is when the wait actually starts for a person.
        waited = (time.perf_counter() - turn.endpointed_at) * 1000
        print(f"  route  {route.describe()}")
        print(f"  bruno  {answer.text}")
        print(f"  took   {waited:.0f}ms to first word ({tr.total_ms:.0f}ms compute)")
        self.convo.add(text, answer.text, answer.privacy, route.tool)
        self._say(answer.text)

    def run(self) -> int:
        self.warmup()
        enrolled = speaker.load() is not None
        print(f"  asleep. say \"hey bruno\". voiceprint: "
              f"{'enrolled' if enrolled else 'NOT enrolled — anyone can wake him'}")
        print(f"  output {audio.output_name()!r}: "
              f"{'full duplex, you can cut him off' if self.full_duplex else 'half duplex, mic mutes while he talks'}")
        print("  Ctrl-C to quit.\n")
        try:
            for turn in self.listener.turns():
                self.handle(turn)
        except KeyboardInterrupt:
            print()
        finally:
            self.listener.stop()
            self.listener.close()
        return 0
