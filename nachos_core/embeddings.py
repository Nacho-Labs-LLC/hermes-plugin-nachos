"""Embedding backends for semantic recall and duplicate detection.

Stdlib only at import time. Every backend is resolved lazily and may be
missing; callers treat ``None`` from :func:`make_embedder` (or an exception
from ``embed``) as "use lexical matching only", never as a failure.

Backends:

* ``nachos``  — the ``nachos-embeddings`` CLI (local Transformers.js model by
  default). A persistent ``serve`` worker (JSON lines over a pipe) keeps the
  model loaded; older CLIs without ``serve`` get one subprocess per batch.
* ``openai``  — any OpenAI-compatible ``/embeddings`` endpoint over plain
  HTTP: Ollama, LM Studio, llama.cpp server, vLLM, or OpenAI itself. The API
  key is read from an environment variable at call time and never stored.
* ``sentence-transformers`` — in-process model, if the package is installed.
* ``auto`` (default) — the first of nachos CLI on PATH, sentence-transformers
  importable, or an OpenAI-compatible base URL if one is configured.
* ``off`` — lexical only.
"""

from __future__ import annotations

import atexit
import contextlib
import importlib.util
import json
import os
import queue
import shutil
import subprocess
import threading
import urllib.parse
import urllib.request
import weakref
from abc import ABC, abstractmethod
from typing import IO, Any

EMBEDDING_BACKENDS = ("auto", "nachos", "openai", "sentence-transformers", "off")

_BATCH = 64
_MAX_TEXT_CHARS = 2000


class Embedder(ABC):
    """Turns texts into vectors. ``model_id`` is stored alongside vectors so a
    model change re-embeds instead of comparing incompatible spaces."""

    model_id: str = ""
    #: Vector length when the backend reports it (checked on every embed).
    dimension: int | None = None

    @abstractmethod
    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed one batch; must return exactly one vector per text."""

    def close(self) -> None:  # noqa: B027 - optional hook, deliberately not abstract
        """Release any resources (worker processes). No-op by default."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        cleaned = [(t or "").replace("\n", " ")[:_MAX_TEXT_CHARS] for t in texts]
        out: list[list[float]] = []
        for start in range(0, len(cleaned), _BATCH):
            batch = cleaned[start : start + _BATCH]
            vectors = self._embed_batch(batch)
            if len(vectors) != len(batch):
                raise RuntimeError(
                    f"{self.model_id}: expected {len(batch)} vectors, got {len(vectors)}"
                )
            for v in vectors:
                if self.dimension is not None and len(v) != self.dimension:
                    raise RuntimeError(
                        f"{self.model_id}: expected {self.dimension}-dimensional vectors, "
                        f"got {len(v)}"
                    )
            out.extend([float(x) for x in v] for v in vectors)
        return out


class _WorkerError(Exception):
    """The serve worker died, hung or spoke garbage (not a per-request error)."""


class _Worker:
    """One ``nachos-embeddings serve`` process plus a thread draining its stdout."""

    def __init__(self, cmd: list[str]) -> None:
        self.proc = subprocess.Popen(  # noqa: S603 - argv list from operator config, no shell
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        if self.proc.stdin is None or self.proc.stdout is None:  # pragma: no cover - PIPE set above
            raise _WorkerError("worker pipes unavailable")
        self._stdin: IO[str] = self.proc.stdin
        self._stdout: IO[str] = self.proc.stdout
        self.lines: queue.Queue[str | None] = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        try:
            for line in self._stdout:
                self.lines.put(line)
        except (OSError, ValueError):
            pass
        self.lines.put(None)  # EOF marker

    def read(self, timeout: float) -> dict[str, Any]:
        try:
            line = self.lines.get(timeout=timeout)
        except queue.Empty:
            raise _WorkerError(f"no reply within {timeout:g}s") from None
        if line is None:
            raise _WorkerError("worker exited")
        try:
            msg = json.loads(line)
        except ValueError:
            raise _WorkerError("non-JSON line from worker") from None
        if not isinstance(msg, dict):
            raise _WorkerError("unexpected message from worker")
        return msg

    def send(self, payload: dict[str, Any]) -> None:
        try:
            self._stdin.write(json.dumps(payload) + "\n")
            self._stdin.flush()
        except (OSError, ValueError) as exc:
            raise _WorkerError(f"cannot write to worker: {exc}") from None

    def stop(self) -> None:
        with contextlib.suppress(OSError, ValueError):
            self._stdin.close()
        try:
            self.proc.terminate()
            self.proc.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            self.proc.kill()
            with contextlib.suppress(subprocess.TimeoutExpired):
                self.proc.wait(timeout=5)


class CliEmbedder(Embedder):
    """``nachos-embeddings`` CLI (see the nachos-embeddings repo).

    Prefers one long-lived ``serve`` worker (model loaded once); falls back to
    one-shot ``embed --stdin --json`` if the CLI has no ``serve`` mode or the
    worker keeps dying. The fallback is remembered for the life of the process.
    """

    def __init__(
        self,
        command: str = "nachos-embeddings",
        model: str = "",
        timeout: float = 180.0,
        request_timeout: float = 60.0,
        dtype: str = "",
    ):
        self.command = command
        self.model = model
        self.dtype = dtype
        # First run downloads the model (~25MB), so allow a generous timeout.
        self.timeout = timeout
        self.request_timeout = request_timeout
        self.model_id = f"nachos:{model or 'default'}" + (f":{dtype}" if dtype else "")
        self._lock = threading.Lock()
        self._worker: _Worker | None = None
        self._serve_disabled = False
        self._restarts = 0
        self._next_id = 0
        atexit.register(_close_ref, weakref.ref(self))

    # -- worker lifecycle -------------------------------------------------
    def _serve_cmd(self) -> list[str]:
        cmd = [self.command, "serve"]
        if self.model:
            cmd += ["--model", self.model]
        if self.dtype:
            cmd += ["--dtype", self.dtype]
        return cmd

    def _start_worker(self) -> _Worker:
        worker = _Worker(self._serve_cmd())
        try:
            msg = worker.read(self.timeout)
            if msg.get("ready") is not True:
                raise _WorkerError("no ready line from worker")
        except _WorkerError:
            worker.stop()
            raise
        self._adopt_ready(msg)
        return worker

    def _adopt_ready(self, msg: dict[str, Any]) -> None:
        """Take the dimension from the ready line (absent on older CLIs).

        ``model_id`` stays the deterministic ``nachos:<model>[:<dtype>]`` because
        the memory service reads it before the worker has started; adopting the
        ready line's id later would make the same index look stale on every run.
        """
        dim = msg.get("dimension")
        if isinstance(dim, int) and not isinstance(dim, bool) and dim > 0:
            self.dimension = dim

    def _drop_worker(self) -> None:
        worker, self._worker = self._worker, None
        if worker is not None:
            worker.stop()

    def close(self) -> None:
        with self._lock:
            self._drop_worker()

    # -- embedding --------------------------------------------------------
    def _request(self, worker: _Worker, texts: list[str]) -> list[list[float]]:
        self._next_id += 1
        rid = self._next_id
        worker.send({"id": rid, "texts": texts})
        while True:
            msg = worker.read(self.request_timeout)
            if msg.get("id") != rid:
                continue  # stale reply from an abandoned request
            if "error" in msg:
                raise RuntimeError(f"nachos-embeddings: {str(msg['error'])[:200]}")
            vectors = msg.get("vectors")
            if not isinstance(vectors, list):
                raise _WorkerError("reply without vectors")
            return vectors

    def _serve_batch(self, texts: list[str]) -> list[list[float]] | None:
        """Vectors from the worker, or None if serve mode is unusable."""
        while not self._serve_disabled:
            try:
                if self._worker is None:
                    self._worker = self._start_worker()
                return self._request(self._worker, texts)
            except (_WorkerError, OSError):  # noqa: PERF203 - retry loop: the try IS the loop body
                was_running = self._worker is not None
                self._drop_worker()
                if not was_running or self._restarts >= 1:
                    self._serve_disabled = True  # unsupported, or died again
                else:
                    self._restarts += 1  # restart once, then retry
        return None

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        with self._lock:
            vectors = self._serve_batch(texts)
        if vectors is not None:
            return vectors
        return self._embed_oneshot(texts)

    def _embed_oneshot(self, texts: list[str]) -> list[list[float]]:
        cmd = [self.command, "embed", "--stdin", "--json"]
        if self.model:
            cmd += ["--model", self.model]
        if self.dtype:
            cmd += ["--dtype", self.dtype]
        proc = subprocess.run(  # noqa: S603 - argv list from operator config, no shell
            cmd,
            input="\n".join(texts) + "\n",
            capture_output=True,
            text=True,
            timeout=self.timeout,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"nachos-embeddings exited {proc.returncode}: {proc.stderr.strip()[:200]}"
            )
        return [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]


def _close_ref(ref: weakref.ReferenceType[CliEmbedder]) -> None:
    embedder = ref()
    if embedder is not None:
        embedder.close()


class OpenAICompatEmbedder(Embedder):
    """POST ``{base_url}/embeddings`` — works with Ollama, LM Studio, OpenAI, ..."""

    def __init__(self, base_url: str, model: str, api_key_env: str = "", timeout: float = 60.0):
        if not base_url or not model:
            raise ValueError("openai embeddings need embedding_base_url and embedding_model")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key_env = api_key_env
        self.timeout = timeout
        self.model_id = f"openai:{self.base_url}:{model}"

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        headers = {"Content-Type": "application/json"}
        key = os.environ.get(self.api_key_env, "") if self.api_key_env else ""
        if key:
            headers["Authorization"] = f"Bearer {key}"
        if urllib.parse.urlsplit(self.base_url).scheme not in ("http", "https"):
            raise ValueError(f"embeddings base_url must be http(s): {self.base_url!r}")
        req = urllib.request.Request(  # noqa: S310 - scheme validated just above
            f"{self.base_url}/embeddings",
            data=json.dumps({"model": self.model, "input": texts}).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310 - http(s) only
            payload = json.loads(resp.read().decode("utf-8"))
        rows = sorted(payload.get("data", []), key=lambda d: d.get("index", 0))
        return [row["embedding"] for row in rows]


class SentenceTransformersEmbedder(Embedder):
    def __init__(self, model: str = "") -> None:
        self.model = model or "all-MiniLM-L6-v2"
        self.model_id = f"st:{self.model}"
        self._model: Any = None

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        if self._model is None:
            # lazy: optional, heavy dependency
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415

            self._model = SentenceTransformer(self.model)
        return [list(map(float, v)) for v in self._model.encode(texts)]


def make_embedder(
    backend: str = "auto",
    *,
    model: str = "",
    base_url: str = "",
    api_key_env: str = "",
    command: str = "nachos-embeddings",
    dtype: str = "",
) -> Embedder | None:
    """Resolve a configured backend name to an Embedder, or None for lexical-only."""
    kind = (backend or "auto").strip().lower()
    if kind == "off":
        return None
    if kind == "auto":
        return _auto_embedder(model, base_url, api_key_env, command, dtype)
    if kind == "nachos":
        return CliEmbedder(command, model, dtype=dtype)
    if kind == "openai":
        return OpenAICompatEmbedder(base_url, model, api_key_env)
    if kind == "sentence-transformers":
        return SentenceTransformersEmbedder(model)
    raise ValueError(f"Unknown embeddings backend {backend!r}; use one of {EMBEDDING_BACKENDS}")


def _auto_embedder(
    model: str, base_url: str, api_key_env: str, command: str, dtype: str = ""
) -> Embedder | None:
    """First available backend: nachos CLI, sentence-transformers, then OpenAI-compatible."""
    if shutil.which(command):
        return CliEmbedder(command, model, dtype=dtype)
    if importlib.util.find_spec("sentence_transformers") is not None:
        return SentenceTransformersEmbedder(model)
    if base_url and model:
        return OpenAICompatEmbedder(base_url, model, api_key_env)
    return None
