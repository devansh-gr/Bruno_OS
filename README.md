# Bruno_OS

Local full-duplex voice assistant. Routes light work to on-device models and hands heavy
codegen to the Claude Code CLI. Docs and planning live in the vault at
`Devansh AI Brain/03 Projects/Bruno_OS/`; this repo is the code.

## Status — M2 complete

End-to-end latency harness, measured on M4 Pro / 24 GB.

| stage | median | worst |
|---|---|---|
| STT (whisper.cpp base.en, audio_ctx 512) | 35 ms | 45 ms |
| LLM first token (llama.cpp, Llama-3.2-3B Q4_K_M) | 88 ms | 146 ms |
| **first audio out** | **200 ms** | **264 ms** |
| full response generated | 539 ms | 816 ms |

Gate is sub-500 ms to first audio. **Passes with 2.5x headroom.** Numbers come from
`python harness.py --bench 10 --with-stt`, which pushes synthesized speech through the
real STT path. Model load is excluded — servers are persistent, so it never lands in a turn.

## Routing and tools (M2)

Three tiers, chosen per request by a schema-constrained local classifier:

| Tier | Runs on | Handles |
|---|---|---|
| `reflex` | local 3B | acks, cancels, chat — the only tier the 500ms gate applies to |
| `agent` | tool call, or Claude | news, video, quotes, tutoring |
| `heavy` | Claude Code CLI | codegen and hard maths, confirmed aloud first |

Measured: **50/50 tier accuracy, 21/21 tool accuracy** on `tests/routing_cases.py`, median 290ms.
**13/13 privacy checks** in `tests/test_privacy.py`.

**Privacy is structural.** Tools declare a `Privacy` class; `Composer.pick_backend` derives the
backend *from the data's class*, so there is no code path that sends `local_only` text to a cloud
model. A tool whose result contradicts its declaration raises `PrivacyViolation` rather than
degrading silently. The classifier is never shown privacy classes and never emits one — it picks
which tool runs, and the registry decides where that tool's output may go.

Without `ANTHROPIC_API_KEY`, the agent tier degrades to the local 3B for tutoring and says so out
loud. That is deliberate: a 3B answering a factual question confidently is the failure this project
most needs to avoid.

## Setup

```bash
brew install llama.cpp whisper-cpp portaudio
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt

hf download bartowski/Llama-3.2-3B-Instruct-GGUF Llama-3.2-3B-Instruct-Q4_K_M.gguf --local-dir models
hf download ggerganov/whisper.cpp ggml-base.en.bin --local-dir models
python -m piper.download_voices en_US-lessac-medium --data-dir models/piper
```

## Run

```bash
python harness.py                      # push-to-talk, speaks the reply
python harness.py --text               # drive the whole loop by typing (no mic)
python harness.py --text --speak       # same, but Bruno speaks the answer
python harness.py --route "how is nvidia doing"   # classify without executing
python harness.py --bench 10 --with-stt   # full pipeline timing
python harness.py --devices            # list audio devices

python -m tests.eval_routing           # routing accuracy against the labelled set
python -m tests.test_privacy           # privacy enforcement
```

## Layout

```
bruno/
  interfaces.py   the four platform seams
  config.py       paths, tunables, backend selection
  latency.py      stage timing — every reported number comes from here
  audio.py        PortAudio capture and streaming playback
  router.py       tier + tool selection, schema-constrained
  agent.py        executes a route; the only place a backend is chosen
  tools/
    base.py       the fifth seam: Tool protocol, registry, Privacy
    public.py     news, video search, market quotes (all cloud_ok)
  backends/
    stt_whispercpp.py   whisper.cpp server (Metal / CUDA)
    llm_llamacpp.py     llama.cpp server (Metal / CUDA)
    tts_piper.py        Piper via ONNX Runtime
    tts_kokoro.py       Kokoro — better voice, too slow here, kept behind the seam
    launcher.py         Claude Code CLI spawn: AppleScript / PowerShell
harness.py        M1 entry point
```

## Cross-platform

macOS today, Windows later. Only four things are platform-specific — LLM runtime, STT,
TTS, terminal launcher — and each sits behind a protocol in `bruno/interfaces.py`.
Portable defaults are GGUF and ONNX, which run on Metal and CUDA alike. The Windows
launcher is written but untested; there is no Windows machine yet.

## Two things that dominated the measurements

**Whisper `--audio-ctx 512`.** The encoder always processes a 30 s mel window regardless of
how long you actually spoke. Truncating the context to 512 frames took base.en from 633 ms
to 38 ms with an identical transcript. Cost: turns are capped at ~10 s of audio, and
`WhisperCppSTT.transcribe` warns loudly rather than silently dropping the tail.

**First-chunk TTS policy.** Waiting for a clause boundary before synthesizing stalled first
audio up to 3.3 s on answers opening with a long clause. The first chunk now cuts at a word
boundary after 8–36 characters; later chunks stay long, which reads better and is free once
audio is already playing. This alone took the median from 419 ms to 183 ms.

## Not built yet

Wake word ("Hey Bruno, wake up") with speaker verification, acoustic echo cancellation, barge-in,
VAD endpointing, email, portfolio, vault retrieval, and the tone fine-tune. `cancel()` exists on the
LLM and TTS backends because barge-in needs it, but nothing calls it yet — the same hooks will carry
speculative routing, since routing currently sits in front of the reply on 42 of 50 cases.

Blocked: the agent tier has no Claude credentials, so tutoring degrades to local. The codegen handoff
confirms and then refuses, because the target directory for `claude -p` is still undecided.

## macOS microphone permission

`harness.py` preflights the mic and exits with instructions if it is blocked. macOS does
not raise on an unauthorized input stream — it blocks forever — so the check probes on a
thread with a deadline. Grant access in System Settings > Privacy & Security > Microphone
for whichever terminal you launch from, then restart that terminal. The grant is per-app,
not per-script. `--bench` needs no mic.
