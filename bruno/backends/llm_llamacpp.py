"""Portable LLM backend: llama.cpp server over HTTP.

Metal on macOS, CUDA on Windows, same code. The server is persistent so model
load and KV-cache warmup never land inside a measured turn.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from typing import Iterator

import requests

from pathlib import Path

from bruno import config


class ReasoningOnly(RuntimeError):
    """The model produced only chain-of-thought and never an answer."""


class LlamaCppLLM:
    name = "llama.cpp"

    def __init__(self, gguf=None, host=None, port=None, n_gpu_layers=999, ctx=4096):
        self.gguf = Path(gguf) if gguf else config.LLM_GGUF
        self.host = host or config.LLAMA_HOST
        self.port = port or config.LLAMA_PORT
        self.n_gpu_layers = n_gpu_layers
        self.ctx = ctx
        self.base = f"http://{self.host}:{self.port}"
        self._proc: subprocess.Popen | None = None
        self._cancel = threading.Event()

    # -- server lifecycle ---------------------------------------------------

    def is_up(self) -> bool:
        try:
            return requests.get(f"{self.base}/health", timeout=0.5).status_code == 200
        except requests.RequestException:
            return False

    def start(self, timeout=180.0) -> None:
        if self.is_up():
            return
        if not self.gguf.exists():
            raise FileNotFoundError(f"missing GGUF: {self.gguf}")
        config.LOGS.mkdir(exist_ok=True)
        log = open(config.LOGS / "llama-server.log", "ab")
        self._proc = subprocess.Popen(
            [
                "llama-server",
                "-m", str(self.gguf),
                "--host", self.host,
                "--port", str(self.port),
                "-c", str(self.ctx),
                "-ngl", str(self.n_gpu_layers),
                "--no-webui",
                "-fa", "on",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.is_up():
                return
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"llama-server died on startup, see {config.LOGS / 'llama-server.log'}"
                )
            time.sleep(0.25)
        raise TimeoutError("llama-server did not come up")

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()

    # -- generation ---------------------------------------------------------

    def stream(self, prompt: str, system: str | None = None) -> Iterator[str]:
        self._cancel.clear()
        reasoning: list[str] = []
        saw_content = False
        body = {
            "messages": [
                {"role": "system", "content": system or config.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "stream": True,
            "max_tokens": config.MAX_TOKENS,
            "temperature": config.TEMPERATURE,
            # Hybrid reasoning models (Qwen3 and friends) think by default and put
            # it in reasoning_content, leaving content empty until the thinking
            # ends. On a sub-500ms voice budget that is a non-starter -- Qwen3-8B
            # spent 300 tokens deliberating on "how many bones are in the hand"
            # and never reached an answer. Harmless on models without the switch.
            "chat_template_kwargs": {"enable_thinking": False},
        }
        with requests.post(
            f"{self.base}/v1/chat/completions", json=body, stream=True, timeout=120
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if self._cancel.is_set():
                    r.close()
                    return
                if not line or not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    return
                choice = json.loads(payload)["choices"][0]
                delta = choice.get("delta", {})
                text = delta.get("content")
                if text:
                    saw_content = True
                    yield text
                elif delta.get("reasoning_content"):
                    # Never silently yield nothing. A thinking model with thinking
                    # left on returns empty content forever, which reads downstream
                    # as a working model that answers "" -- the benchmark scored
                    # that as 100% confident-wrong before this was caught.
                    reasoning.append(delta["reasoning_content"])
            if not saw_content and reasoning:
                joined = "".join(reasoning).strip()
                raise ReasoningOnly(
                    f"model returned only reasoning ({len(joined)} chars), no answer; "
                    "thinking is enabled and the token budget ran out inside it"
                )

    def cancel(self) -> None:
        self._cancel.set()

    def warmup(self) -> None:
        self.start()
        for _ in self.stream("hi", system="Reply with one word."):
            pass
