"""Listen, transcribe, route, answer, speak, listen again.

Duplex mode is chosen from the output device, not assumed. Through headphones
nothing reaches the mic (measured 0.06) so the mic stays live and he can be cut off
mid-sentence. On speakers his own voice returns at 0.435 and Silero scores it 1.000,
so the mic is decoupled while he talks. See aec.py for cancelling that instead.
"""

from __future__ import annotations

import time

import statistics
import threading

import numpy as np

import time as _time

from bruno import audio, config, hearing, presence, speaker, wake
from bruno.latency import Trace
from bruno.listen import Listener
from bruno.memory import Conversation
from bruno.router import Router
from bruno.tools.base import Tier
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
        self._speaking = threading.Event()
        self._interrupted = False
        self.latencies: list[float] = []
        self.presence = presence.load()

    def warmup(self) -> None:
        from bruno import preflight
        if not preflight.report(preflight.run(need_mic=True)):
            raise SystemExit("preflight failed, see above")
        t = time.perf_counter()
        self.stt.warmup()
        self.tts.warmup()
        self.router.route("warm up")
        self.listener.open()
        print(f"  ready in {time.perf_counter() - t:.1f}s")

    def _say(self, text: str) -> float | None:
        """Speak. In full duplex, stop the moment he starts talking over it.

        Returns the clock at first audio, which is the only latency that matters.
        """
        if not (self.speak and text):
            return None
        from bruno.backends.tts_piper import chunk_text

        # Only decouple when the room would carry his voice back to the mic.
        self.listener.paused = not self.full_duplex
        self._interrupted = False
        watch = None
        if self.full_duplex:
            watch = threading.Thread(target=self._watch_for_barge_in, daemon=True)
            watch.start()
        try:
            self._speaking.set()
            self.speaker.start()
            first = None
            try:
                for chunk in self.tts.stream(chunk_text(iter([text]))):
                    if self._interrupted:
                        break
                    if first is None:
                        first = time.perf_counter()
                    self.speaker.write(chunk)
            except Exception as e:
                # The answer is already on screen. A voice failure costs this reply,
                # not the conversation.
                print(f"  (voice failed: {type(e).__name__}: {e})")
            self.speaker.finish()
            self.speaker.wait(timeout=90)
            self.speaker.stop()
            # His own tail is still crossing the room.
            if not self.full_duplex:
                time.sleep(0.25)   # his own tail is still crossing the room
        finally:
            self._speaking.clear()
            if watch is not None:
                watch.join(timeout=1)
            if not self.full_duplex:
                self.listener.vad.reset()
            self.listener.paused = False
        return first

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

        if presence.going_to_sleep(text):
            self.presence.asleep = True
            self.presence.since = _time.time()
            self.presence.held = 0
            presence.save(self.presence)
            self.awake = False
            self._say("Night. I'll keep quiet unless something's urgent.")
            return None

        if presence.waking_up(text) and self.presence.asleep:
            held = self.presence.held
            self.presence = presence.State()
            presence.save(self.presence)
            self._say(f"Morning. {held} thing{'s' if held != 1 else ''} waited."
                      if held else "Morning.")
            return None

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

    # Sustained speech, not a single frame: a cough or a keyboard clack should not
    # cut him off mid-sentence, but a real interruption is continuous.
    BARGE_IN_MS = 320.0

    # Tiers that reach the network or spawn a terminal. Measured: generation alone
    # is 804ms median, and retrieval adds seconds on top. Dead silence for that long
    # reads as "it did not hear me", and the answer gets repeated.
    SLOW_TIERS = (Tier.HEAVY,)
    # Never one word. His rule, from the same reason the banned-utterance list
    # exists: a short sound through a speaker with nothing around it to disambiguate
    # it cannot be made out. "Checking." fails that as surely as "Go?" does.
    ACKS = {
        "news": "Checking the news.", "briefing": "Give me a second.",
        "market_quote": "Looking it up.", "video_search": "Searching for it.",
        "check_email": "Checking your mail.", "draft_email": "Drafting it now.",
        "project_status": "Let me look.", "claude_sessions": "Checking on it.",
        "play_song": "Putting it on.", "tell_me_more": "Give me a second.",
        "pull_up": "Pulling it up.",
    }

    def _ack_for(self, route) -> str | None:
        """A short word so the wait is not dead air. None when the answer is fast."""
        if route.tool in self.ACKS:
            return self.ACKS[route.tool]
        if route.tier in self.SLOW_TIERS:
            return "Handing that to Claude."
        if route.tier is Tier.AGENT and route.tool is None and config.RETRIEVE_FOR_FACTS:
            return "Let me check."      # tutoring retrieves, which costs seconds
        return None

    def _watch_for_barge_in(self) -> None:
        """Cut playback when he talks over it. Full duplex only, where the mic is
        hearing the room rather than Bruno."""
        from bruno.vad import FRAME_MS
        need = int(self.BARGE_IN_MS / FRAME_MS)
        run = 0
        vad = self.listener.vad
        cursor = None
        _, cursor = self.listener.rec.since(None)
        buf = np.zeros(0, dtype=np.float32)
        while self._speaking.is_set():
            chunk, cursor = self.listener.rec.since(cursor)
            if len(chunk):
                buf = np.concatenate([buf, chunk])
            from bruno.vad import FRAME
            while len(buf) >= FRAME:
                run = run + 1 if vad.prob(buf[:FRAME]) >= self.listener.ep.start_prob else 0
                buf = buf[FRAME:]
                if run >= need:
                    self._interrupted = True
                    self.tts.cancel()
                    self.speaker.stop()
                    print("  (interrupted)")
                    return
            time.sleep(0.01)

    def handle(self, turn) -> None:
        tr = Trace()
        with tr.stage("stt"):
            text = self.stt.transcribe(turn.audio, config.SAMPLE_RATE)
        text = (text or "").strip()
        # Whisper invents text for silence, hiss and clicks. Without this the loop
        # routes "you" and "(static)" as requests.
        if hearing.is_noise(text, turn.speech_ms):
            return
        text = hearing.clean(text)
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

        ack = self._ack_for(route)
        if ack:
            self._say(ack)
        with tr.stage("answer"):
            answer = self.agent.run(route, confirmed=confirmed)
        if "awaiting confirmation" in answer.notes:
            self.pending = route

        # From the endpoint: that is when the wait actually starts for a person.
        print(f"  route  {route.describe()}")
        print(f"  bruno  {answer.text}")
        self.convo.add(text, answer.text, answer.privacy, route.tool)

        spoke_at = self._say(answer.text)
        # "Time to first word" is the wait he actually experiences: from the moment
        # he stopped talking to the moment sound comes out. Reporting it before the
        # voice ran measured everything except the part he waits on.
        end = spoke_at if spoke_at is not None else time.perf_counter()
        waited = (end - turn.endpointed_at) * 1000
        self.latencies.append(waited)
        print(f"  took   {waited:.0f}ms to first word "
              f"(stt+route+answer {tr.total_ms:.0f}ms, median {statistics.median(self.latencies):.0f}ms)")

    def speak_unprompted(self, text: str,
                         urgency: presence.Urgency = presence.Urgency.NORMAL) -> bool:
        """Say something he did not ask for. Returns whether it was actually said.

        The only route for Bruno-initiated speech, so the sleep gate cannot be
        bypassed by a caller that forgets it exists.
        """
        if not presence.may_speak(urgency, self.presence):
            self.presence.held += 1
            presence.save(self.presence)
            return False
        self._say(text)
        return True

    def run(self) -> int:
        self.warmup()
        enrolled = speaker.load() is not None
        print(f"  asleep. say \"hey bruno\". voiceprint: "
              f"{'enrolled' if enrolled else 'NOT enrolled — anyone can wake him'}")
        print(f"  output {audio.output_name()!r}: "
              f"{'full duplex, you can cut him off' if self.full_duplex else 'half duplex, mic mutes while he talks'}")
        if self.presence.asleep:
            print(f"  do not disturb, {presence.hours_asleep(self.presence):.1f}h "
                  f"({self.presence.held} held). say \"good morning\".")
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
