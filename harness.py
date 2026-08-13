#!/usr/bin/env python3
"""M1 — end-to-end latency harness.

Push-to-talk, no wake word, no AEC, no barge-in. Its only job is to produce real
per-stage numbers for the budget the spec demands:

    VAD endpoint -> STT -> LLM first token -> TTS first audio out

The clock starts the instant recording stops, because that is the moment a real
VAD would have endpointed. Everything before that is the user still talking.

    python harness.py                 interactive push-to-talk
    python harness.py --bench N       N synthetic turns, no mic, prints medians
    python harness.py --devices       list audio devices
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

import numpy as np

from bruno import audio, config
from bruno.backends.llm_llamacpp import LlamaCppLLM
from bruno.backends.stt_whispercpp import WhisperCppSTT
from bruno.backends.tts_piper import chunk_text
from bruno.latency import Trace

BENCH_PROMPTS = [
    "What is the difference between a Kalman filter and a complementary filter?",
    "Explain what quantization does to a neural network in one sentence.",
    "How many bones are in the human hand?",
    "What sample rate does Whisper expect?",
    "Give me a one sentence definition of a state space model.",
]


def run_turn(stt, llm, tts, speaker, clip, sample_rate, text=None, speak=True):
    """One voice turn. Returns (trace, transcript, reply)."""
    tr = Trace()

    if text is None:
        with tr.stage("stt"):
            text = stt.transcribe(clip, sample_rate)
        tr.mark("stt_done")
        if not text:
            return tr, "", ""
    else:
        tr.mark("stt_skipped")

    reply_parts = []
    first_token_seen = False

    def deltas():
        nonlocal first_token_seen
        for d in llm.stream(text):
            if not first_token_seen:
                first_token_seen = True
                tr.mark("llm_first_token")
            reply_parts.append(d)
            yield d

    first_audio = None
    played = 0
    for chunk in tts.stream(chunk_text(deltas())):
        if first_audio is None:
            tr.mark("tts_first_chunk")
            if speak:
                speaker.start()
            first_audio = tr.mark("first_audio_out")
        if speak:
            speaker.write(chunk)
        played += len(chunk)

    tr.mark("llm_done")
    if speak and first_audio is not None:
        speaker.finish()
        speaker.wait(timeout=played / tts.sample_rate + 5)
        speaker.stop()

    return tr, text, "".join(reply_parts)


def boot():
    stt, llm, tts = WhisperCppSTT(), LlamaCppLLM(), config.make_tts()
    print(f"backends: stt={stt.name}({config.STT_GGML.name})  "
          f"llm={llm.name}({config.LLM_GGUF.name})  tts={tts.name}")
    print("warming up (model load is excluded from every measured turn)...", flush=True)
    t = time.perf_counter()
    stt.warmup()
    llm.warmup()
    tts.warmup()
    print(f"ready in {time.perf_counter() - t:.1f}s\n")
    return stt, llm, tts


def _as_mic_audio(tts, text):
    """Synthesize a prompt and resample to the mic rate, so the bench exercises the
    real STT path instead of skipping it. Not a substitute for a human voice — it
    has no room noise, no disfluency — but it is honest about the compute cost."""
    clip = tts.synth(text)
    n = int(len(clip) * config.SAMPLE_RATE / tts.sample_rate)
    return np.interp(
        np.linspace(0, len(clip) - 1, n), np.arange(len(clip)), clip
    ).astype(np.float32)


def bench(n, with_stt):
    stt, llm, tts = boot()
    speaker = audio.Speaker(tts.sample_rate)
    keys = ["stt_done", "llm_first_token", "tts_first_chunk", "first_audio_out", "llm_done"]
    rows = []
    for i in range(n):
        prompt = BENCH_PROMPTS[i % len(BENCH_PROMPTS)]
        clip = _as_mic_audio(tts, prompt) if with_stt else None
        tr, heard, reply = run_turn(
            stt, llm, tts, speaker, clip, config.SAMPLE_RATE,
            text=None if with_stt else prompt, speak=False,
        )
        rows.append(tr.marks)
        print(f"[{i + 1}/{n}] first_audio_out {tr.marks['first_audio_out']:6.1f}ms  "
              f"| {reply.strip()[:58]}")

    label = ("full pipeline: synthesized speech -> STT -> LLM -> TTS"
             if with_stt else "STT excluded — LLM + TTS only")
    print(f"\n  {label}")
    print(f"  {'mark':<18} {'median':>9} {'min':>9} {'max':>9}")
    print("  " + "-" * 48)
    for k in keys:
        vals = [r[k] for r in rows if k in r]
        if vals:
            print(f"  {k:<18} {statistics.median(vals):9.1f} {min(vals):9.1f} {max(vals):9.1f}")

    med = statistics.median([r["first_audio_out"] for r in rows])
    worst = max(r["first_audio_out"] for r in rows)
    if with_stt:
        print(f"\n  GATE first_audio_out < 500ms: median {med:.1f}ms "
              f"{'PASS' if med < 500 else 'FAIL'}, worst {worst:.1f}ms "
              f"{'PASS' if worst < 500 else 'FAIL'}")
    else:
        print(f"\n  budget left for STT if the 500ms gate is to hold: {500 - med:.1f}ms")
    return 0


def interactive():
    print("checking microphone...", flush=True)
    problem = audio.check_mic()
    if problem:
        print(f"\n  {problem}\n")
        print("  --bench still works without a mic:")
        print("      python harness.py --bench 10 --with-stt\n")
        return 1

    stt, llm, tts = boot()
    rec = audio.Recorder()
    rec.open()  # once, at boot — a per-turn open loses the first ~0.5s to Core Audio warmup
    speaker = audio.Speaker(tts.sample_rate)
    print(f"mic open, {rec.pre_roll * 1000:.0f}ms pre-roll (start talking early, it still lands)")
    print("ENTER to start talking, ENTER again to stop. Ctrl-C to quit.\n")
    while True:
        try:
            input("[press ENTER to talk] ")
            rec.start()
            input("[recording... ENTER to stop] ")
            clip = rec.stop()
        except (KeyboardInterrupt, EOFError):
            print()
            break

        secs = len(clip) / config.SAMPLE_RATE
        if secs < 0.2:
            print("  too short, ignored\n")
            continue

        tr, text, reply = run_turn(stt, llm, tts, speaker, clip, config.SAMPLE_RATE)
        print(f"\n  you   : {text}")
        print(f"  bruno : {reply.strip()}")
        print(f"  (spoke {secs:.1f}s){tr.report()}\n")

    rec.close()
    llm.stop()
    stt.stop()
    return 0


def show_route(text):
    """Classify without executing. The fastest way to see why Bruno chose a tier."""
    from bruno.router import Router
    from bruno.tools import media, public, workspace  # noqa: F401 — registers tools

    print(Router().route(text).describe())
    return 0


def text_mode(speak):
    """The whole loop, driven by typing. No mic, no wake word — this is how M2 is
    exercised while the voice layer is still unbuilt."""
    from bruno.agent import Agent
    from bruno.memory import Conversation
    from bruno.router import Router
    from bruno.tools import media, public, workspace  # noqa: F401 — registers tools

    router, agent, convo = Router(), Agent(), Conversation()
    tts = speaker = None
    if speak:
        tts = config.make_tts()
        tts.warmup()
        speaker = audio.Speaker(tts.sample_rate)

    print("warming the router...", flush=True)
    router.route("warm up")
    print("type a request, Ctrl-C to quit. 'y' confirms a pending action.\n")

    pending = None
    while True:
        try:
            text = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break
        if not text:
            continue

        if pending is not None and text.lower() in {"y", "yes", "yeah", "go", "do it"}:
            route, confirmed = pending, True
        else:
            route, confirmed = router.route(text, convo), False
        pending = None

        print(f"  route  {route.describe()}")
        tr = Trace()
        with tr.stage("execute"):
            answer = agent.run(route, confirmed=confirmed)
        if "awaiting confirmation" in answer.notes:
            pending = route

        print(f"  via    {answer.backend}{'  DEGRADED' if answer.degraded else ''}")
        for note in answer.notes:
            print(f"  note   {note}")
        print(f"  bruno  {answer.text}")
        print(f"  took   {tr.total_ms:.0f}ms\n")
        convo.add(text, answer.text, answer.privacy, route.tool)

        if speak and answer.text:
            speaker.start()
            for chunk in tts.stream(chunk_text(iter([answer.text]))):
                speaker.write(chunk)
            speaker.finish()
            speaker.wait(timeout=60)
            speaker.stop()
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", type=int, metavar="N", help="synthetic turns, no mic")
    ap.add_argument("--with-stt", action="store_true",
                    help="feed synthesized speech through STT for a full-pipeline number")
    ap.add_argument("--route", metavar="TEXT", help="classify TEXT and exit, without executing")
    ap.add_argument("--text", action="store_true", help="type requests instead of speaking")
    ap.add_argument("--speak", action="store_true", help="with --text, also speak the answer")
    ap.add_argument("--devices", action="store_true")
    a = ap.parse_args()
    if a.devices:
        print(audio.list_devices())
        return 0
    if a.route:
        return show_route(a.route)
    if a.text:
        return text_mode(a.speak)
    return bench(a.bench, a.with_stt) if a.bench else interactive()


if __name__ == "__main__":
    sys.exit(main())
