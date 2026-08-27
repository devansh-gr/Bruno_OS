# Bruno_OS

Local full-duplex voice assistant. Routes light work to on-device models and hands heavy
codegen to the Claude Code CLI. Docs and planning live in the vault at
`Devansh AI Brain/03 Projects/Bruno_OS/`; this repo is the code.

## Status: M3, retrieval

End-to-end latency harness, measured on M4 Pro / 24 GB.

| stage | median | worst |
|---|---|---|
| STT (whisper.cpp base.en, audio_ctx 512) | 35 ms | 45 ms |
| LLM first token (llama.cpp, Llama-3.2-3B Q4_K_M) | 88 ms | 146 ms |
| **first audio out** | **200 ms** | **264 ms** |
| full response generated | 539 ms | 816 ms |

Gate is sub-500 ms to first audio. **Passes with 2.5x headroom.** Numbers come from
`python harness.py --bench 10 --with-stt`, which pushes synthesized speech through the
real STT path. Model load is excluded, since the servers are persistent and it never lands in a turn.

## Why factual questions retrieve (M3)

Measured on **SimpleQA** (OpenAI, 4326 adversarially collected factoids), answering from the
model's own memory:

| | correct | bluffs | declined |
|---|---|---|---|
| Llama-3.2-3B | 0 / 120 | 27 | 93 |
| Qwen3-8B | 2 / 120 | 67 | 51 |

Frontier models score 38-45% here. Both local models are at zero, and the wrong answers are
confident inventions rather than near-misses. `Park Geun-hye` came back as `Moon Jae-in`,
`11 December 1836` as `July 15, 1817`.

On **TruthfulQA MC1** the same models score 56.7% and 72.5% against a ~20-25% random baseline.
So the split is sharp: **they can reason about concepts and cannot recall specifics.**

Retrieval closes the second half. Where the answer actually appears:

| source | answer present |
|---|---|
| model memory | 0 / 120 |
| Wikipedia keyword search | 1 / 10 |
| web search | **7 / 8** |

Wikipedia fails because SimpleQA questions are long descriptive sentences and its search wants
entities, and half the queries returned no article at all. That earlier result led to the wrong
conclusion that "grounding hurts"; the retrieval was bad, not the idea.

### How retrieval works

Four stages, in `bruno/websearch.py` and `bruno/retrieve.py`. Each exists because something
measurably failed without it.

**1. Search.** The question goes verbatim to DuckDuckGo's HTML endpoint, with Bing as a fallback.
Verbatim matters. The earlier Wikipedia path tried to extract an entity first and that is where it
lost, because "What is the name of India's first nano-satellite?" has no entity to extract. A search
engine already handles the long descriptive phrasing that SimpleQA is written in.

Requests are paced at one every 3 seconds. Below that the endpoint starts answering `202` with an
empty body. Not an error, not a rejection, just nothing, which is indistinguishable from "no
results exist" unless you know to look for it. That case raises `Throttled`, and the benchmark
excludes throttled questions from its denominator instead of scoring them as model failures.

**2. Strip.** Search HTML is mostly not results. Tags come out, entities are unescaped, and a small
set of chrome patterns ("All Regions", "Safe search: moderate", the time filters) are removed so
they cannot win a relevance score against real text.

**3. Focus.** A 3B has ~8k of context and the stripped page is 15-30k characters. A sliding 420-char
window is scored by how many of the question's content words it contains; the top 6 non-overlapping
windows become the evidence. Scoring uses content words only. Stopwords are dropped, and
possessives are stripped, because `india's` never matches "India" on any page. That single bug cost
the whole first retrieval run.

**4. Ground.** The evidence is passed with a system prompt that permits *only* what the results say:

```
You answer using ONLY the search results provided.
First check the question's premise against the results.
If the results do not contain the answer, reply exactly: I don't know.
Never fill the gap from memory.
```

The premise check is there because SimpleQA contains questions with false premises, and a model that
answers them at all has already failed. `I don't know` is scored as `not_attempted`, not wrong. The
distinction is the entire point: a tutor that declines is usable, a tutor that bluffs is not.

Retrieval runs on every factual question (`RETRIEVE_FOR_FACTS`, on by default). When it comes back
empty, Bruno falls through to the model and says the answer came from memory rather than a source.

### Result

Same 25 SimpleQA questions, same model (Llama-3.2-3B-Instruct-Q4_K_M), same seed. The only
difference is whether the question was searched before the model saw it.

| | from memory | with retrieval |
|---|---|---|
| correct | **0 / 25** (0.0%) | **17 / 25** (68.0%) |
| bluffs (confident and wrong) | 7 | 5 |
| declined | 18 | 3 |
| accuracy when it answered | 0.0% | **77.3%** |
| F-score | 0.0 | **72.3** |

For scale: frontier models score **38-45%** on SimpleQA. The 25-question subset is noisy and this is
not a claim of beating them, but the direction is not ambiguous. A 3B that could not answer a
single factual question now answers two in three.

The second row matters as much as the first. Bluffs fell 7 → 5 while attempts rose sharply: from
memory the model declined 18 of 25 and was wrong on every one of the 7 it tried. With retrieval it
attempted 22 and was right on 17. It is not just answering more, it is answering more *and* being
wrong less often.

The three remaining declines are the pipeline working correctly. Search returned pages that did not
contain the answer, and the grounded prompt required `I don't know` rather than a guess.

## Routing and tools (M2)

Three tiers, chosen per request by a schema-constrained local classifier:

| Tier | Runs on | Handles |
|---|---|---|
| `reflex` | local 3B | acks, cancels, chat; the only tier the 500ms gate applies to |
| `agent` | tool call, or Claude | news, video, quotes, tutoring |
| `heavy` | Claude Code CLI | codegen and hard maths, confirmed aloud first |

Measured: **100/100 tier accuracy, 67/67 tool accuracy** on `tests/routing_cases.py`, median 216ms.
**17/17 privacy checks** in `tests/test_privacy.py`.

**Privacy is structural.** Tools declare a `Privacy` class; `Composer.pick_backend` derives the
backend *from the data's class*, so there is no code path that sends `local_only` text to a cloud
model. A tool whose result contradicts its declaration raises `PrivacyViolation` rather than
degrading silently. The classifier is never shown privacy classes and never emits one. It picks
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
  latency.py      stage timing; every reported number comes from here
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
    tts_kokoro.py       Kokoro: better voice, too slow here, kept behind the seam
    launcher.py         Claude Code CLI spawn: AppleScript / PowerShell
harness.py        M1 entry point
```

## Cross-platform

macOS today, Windows later. Only four things are platform-specific: LLM runtime, STT,
TTS and terminal launcher. Each sits behind a protocol in `bruno/interfaces.py`.
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
LLM and TTS backends because barge-in needs it, but nothing calls it yet. The same hooks will carry
speculative routing, since routing currently sits in front of the reply on 42 of 50 cases.

Blocked: the agent tier has no Claude credentials, so tutoring degrades to local. The codegen handoff
confirms and then refuses, because the target directory for `claude -p` is still undecided.

## macOS microphone permission

`harness.py` preflights the mic and exits with instructions if it is blocked. macOS does
not raise on an unauthorized input stream, it blocks forever, so the check probes on a
thread with a deadline. Grant access in System Settings > Privacy & Security > Microphone
for whichever terminal you launch from, then restart that terminal. The grant is per-app,
not per-script. `--bench` needs no mic.
