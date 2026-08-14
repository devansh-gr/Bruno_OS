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
        body = {
            "messages": [
                {"role": "system", "content": system or config.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "stream": True,
            "max_tokens": config.MAX_TOKENS,
            "temperature": config.TEMPERATURE,
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
                delta = json.loads(payload)["choices"][0].get("delta", {}).get("content")
                if delta:
                    yield delta

    def cancel(self) -> None:
        self._cancel.set()

    def warmup(self) -> None:
        self.start()
        for _ in self.stream("hi", system="Reply with one word."):
            pass
