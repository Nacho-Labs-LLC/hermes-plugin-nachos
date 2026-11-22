"""Embedding backends for semantic recall and duplicate detection.

Stdlib only at import time. Every backend is resolved lazily and may be
missing; callers treat ``None`` from :func:`make_embedder` (or an exception
from ``embed``) as "use lexical matching only", never as a failure.

Backends:

* ``nachos``  — the ``nachos-embeddings`` CLI (local Transformers.js model by
  default). A persistent ``serve`` worker (JSON lines over a pipe) keeps the
  model loaded; older CLIs without ``serve`` get one subprocess per batch.
  CLIs that support input types (the ready line carries ``pooling`` /
  prefix fields) are told whether a text is a query or a document, so
  retrieval models apply their own prefixes; older CLIs are never sent a type.
* ``openai``  — any OpenAI-compatible ``/embeddings`` endpoint over plain
  HTTP: Ollama, LM Studio, llama.cpp server, vLLM, or OpenAI itself. The API
  key is resolved by an injected host resolver at request time, never from
  the process environment. Without a key name, local endpoints need no auth.
* ``sentence-transformers`` — in-process model, if the package is installed.

``openai`` and ``sentence-transformers`` have no query/document concept, so
the configured ``embedding_query_prefix`` / ``embedding_document_prefix`` are
prepended client-side (e.g. ``query: `` / ``passage: `` for E5 models).
* ``auto`` (default) — the first of nachos CLI on PATH, sentence-transformers
  importable, or an OpenAI-compatible base URL if one is configured.
* ``off`` — lexical only.
"""

from __future__ import annotations

import atexit
import contextlib
import importlib.util
import json
import logging
import queue
import re
import shutil
import subprocess
import threading
import urllib.parse
import urllib.request
import weakref
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import IO, Any

from .models import prefix_argument

logger = logging.getLogger(__name__)

DEFAULT_RERANK_MODEL = "jinaai/jina-reranker-v1-tiny-en"
EMBEDDING_BACKENDS = ("auto", "nachos", "openai", "sentence-transformers", "off")
MIN_CLI_VERSION = (0, 7, 0)
MIN_CLI_VERSION_STR = ".".join(str(part) for part in MIN_CLI_VERSION)
_VERSION_RE = re.compile(r"\bv?(\d+)\.(\d+)\.(\d+)\b")

_BATCH = 64
_MAX_TEXT_CHARS = 2000
QUERY = "query"
DOCUMENT = "document"


def parse_cli_version(text: str) -> tuple[int, int, int] | None:
    """Parse a single-line semantic version string (e.g. ``0.6.0`` or ``nachos-embeddings 0.6.0``)."""
    stripped = (text or "").strip()
    if not stripped or "\n" in stripped or "\r" in stripped:
        return None
    match = _VERSION_RE.search(stripped)
    if match is None:
        return None
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))


def _probe_cli_version(command: str) -> tuple[bool, tuple[int, int, int] | None]:
    """Run ``command --version``: ``(ran, semver)``.

    ``ran`` is False when the binary is missing or hangs (no verdict; embedding then fails or
    falls back to lexical as usual). A CLI that runs but exits non-zero or prints nothing
    parseable has no ``--version`` and so predates 0.7.0: ``(True, None)``.
    """
    try:
        proc = subprocess.run(  # noqa: S603 - argv list from operator config, no shell
            [command, "--version"],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False, None
    if proc.returncode != 0:
        return True, None
    return True, parse_cli_version(proc.stdout)


class Embedder(ABC):
    """Turns texts into vectors. ``model_id`` is stored alongside vectors so a
    model change re-embeds instead of comparing incompatible spaces."""

    model_id: str = ""
    #: Vector length when the backend reports it (checked on every embed).
    dimension: int | None = None
    #: Client-side prefixes for backends without a native input type.
    query_prefix: str = ""
    document_prefix: str = ""
    #: True once the backend can score (query, passage) pairs with a cross-encoder.
    supports_rerank: bool = False

    def rerank(self, query: str, passages: list[str]) -> list[float] | None:  # noqa: ARG002
        """Relevance scores (higher is better, one per passage), or None if unsupported."""
        return None

    @abstractmethod
    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed one batch; must return exactly one vector per text."""

    def _embed_typed(self, texts: list[str], input_type: str) -> list[list[float]]:
        """Embed one batch for ``input_type``; backends with a native type override this."""
        del input_type  # no native type: prefixes were applied by embed()
        return self._embed_batch(texts)

    def warm_up(self) -> None:  # noqa: B027 - optional hook, deliberately not abstract
        """Learn anything that changes ``model_id`` (e.g. start a worker). Best effort, never raises."""

    def close(self) -> None:  # noqa: B027 - optional hook, deliberately not abstract
        """Release any resources (worker processes). No-op by default."""

    def embed(self, texts: list[str], input_type: str = DOCUMENT) -> list[list[float]]:
        """Vectors for ``texts``; ``input_type`` is ``"document"`` (default) or ``"query"``."""
        if input_type not in (QUERY, DOCUMENT):
            raise ValueError(f"input_type must be {QUERY!r} or {DOCUMENT!r}, got {input_type!r}")
        prefix = self.query_prefix if input_type == QUERY else self.document_prefix
        cleaned = [prefix + (t or "").replace("\n", " ")[:_MAX_TEXT_CHARS] for t in texts]
        out: list[list[float]] = []
        for start in range(0, len(cleaned), _BATCH):
            batch = cleaned[start : start + _BATCH]
            vectors = self._embed_typed(batch, input_type)
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
        rerank: bool = False,
        rerank_model: str = "",
        rerank_dtype: str = "",
        pooling: str = "",
        max_length: int = 0,
        normalize: bool | None = None,
        query_prefix: str = "",
        document_prefix: str = "",
        rerank_max_length: int = 0,
    ):
        self.command = command
        self.model = model
        self.dtype = dtype
        #: Overrides of the CLI's per-model profile; unset (blank/0/None) = profile default.
        self.pooling = pooling
        self.max_length = max_length
        self.normalize = normalize
        self.cli_query_prefix = query_prefix
        self.cli_document_prefix = document_prefix
        self.rerank_max_length = rerank_max_length
        #: Ask the serve worker to load a cross-encoder (needs a CLI with ``--rerank-model``).
        self.rerank_enabled = rerank
        self.rerank_model = rerank_model or DEFAULT_RERANK_MODEL
        self.rerank_dtype = rerank_dtype
        # First run downloads the model (~25MB), so allow a generous timeout.
        self.timeout = timeout
        self.request_timeout = request_timeout
        self._base_id = f"nachos:{model or 'default'}" + (f":{dtype}" if dtype else "")
        self.model_id = self._base_id
        self._lock = threading.Lock()
        self._worker: _Worker | None = None
        self._serve_disabled = False
        #: True once a ready line shows the CLI understands input types.
        self.supports_type = False
        #: Set False to keep legacy behaviour (never send a type) even if supported.
        self.send_type = True
        self._restarts = 0
        self._next_id = 0
        self._rerank_failed = False  # logged once, then reranking stays off for the process
        #: Parsed semver of the nachos-embeddings CLI once probed (from --version or ready line).
        self.cli_version: tuple[int, int, int] | None = None
        self._version_checked = False
        self._legacy_cli = False
        self._version_warned = False
        atexit.register(_close_ref, weakref.ref(self))

    # -- CLI version check ------------------------------------------------
    def _ensure_cli_version(self) -> None:
        if self._version_checked:
            return
        self._version_checked = True
        ran, version = _probe_cli_version(self.command)
        if version is not None:
            self._apply_cli_version(version)
        elif ran:
            self._degrade_to_legacy("(no --version)")

    def _apply_cli_version(self, version: tuple[int, int, int]) -> None:
        """Record the CLI's version; a ready-line version also overrides an earlier probe verdict."""
        self.cli_version = version
        if version >= MIN_CLI_VERSION:
            if self._legacy_cli:
                self._legacy_cli = False
                self._rerank_failed = False
            return
        self._degrade_to_legacy(".".join(str(part) for part in version))

    def _degrade_to_legacy(self, found: str) -> None:
        """Stop using input types, reranking and model overrides; warn once."""
        self._legacy_cli = True
        self.supports_type = False
        self.supports_rerank = False
        self._rerank_failed = True
        if not self._version_warned:
            self._version_warned = True
            logger.warning(
                "nachos-embeddings %s is older than %s; disabling input types, reranking, "
                "and model overrides (upgrade with: npm i -g @nacho-labs/nachos-embeddings).",
                found,
                MIN_CLI_VERSION_STR,
            )

    def _check_ready_version(self, msg: dict[str, Any]) -> None:
        ver = msg.get("version")
        if isinstance(ver, str):
            parsed = parse_cli_version(ver)
            if parsed is not None:
                self._apply_cli_version(parsed)

    # -- worker lifecycle -------------------------------------------------
    def _model_args(self) -> list[str]:
        """Model flags shared by ``serve`` and one-shot ``embed``; only what was configured."""
        args: list[str] = []
        if self.model:
            args += ["--model", self.model]
        if self.dtype:
            args += ["--dtype", self.dtype]
        if self._legacy_cli:
            return args
        if self.pooling:
            args += ["--pooling", self.pooling]
        if self.max_length > 0:
            args += ["--max-length", str(self.max_length)]
        if self.normalize is not None:
            args.append("--normalize" if self.normalize else "--no-normalize")
        for flag, value in (
            ("--query-prefix", prefix_argument(self.cli_query_prefix)),
            ("--document-prefix", prefix_argument(self.cli_document_prefix)),
        ):
            if value is not None:
                args += [flag, value]
        return args

    def _serve_cmd(self) -> list[str]:
        cmd = [self.command, "serve", *self._model_args()]
        if self.rerank_enabled and not self._rerank_failed:
            cmd += ["--rerank-model", self.rerank_model]
            if self.rerank_dtype:
                cmd += ["--rerank-dtype", self.rerank_dtype]
            if self.rerank_max_length > 0:
                cmd += ["--rerank-max-length", str(self.rerank_max_length)]
        return cmd

    def _start_worker(self) -> _Worker:
        self._ensure_cli_version()
        try:
            return self._launch(self._serve_cmd())
        except _WorkerError:
            if not (self.rerank_enabled and not self._rerank_failed):
                raise
            # An older CLI exits on the unknown --rerank-* flags: embed without reranking.
            self._note_rerank_unavailable("the CLI did not start with --rerank-model")
            return self._launch(self._serve_cmd())

    def _launch(self, cmd: list[str]) -> _Worker:
        worker = _Worker(cmd)
        try:
            msg = worker.read(self.timeout)
            if msg.get("ready") is not True:
                raise _WorkerError("no ready line from worker")
        except _WorkerError:
            worker.stop()
            raise
        self._adopt_ready(msg)
        if self.rerank_enabled and not self.supports_rerank:
            self._note_rerank_unavailable("the CLI's ready line has no rerank support")
        return worker

    def _note_rerank_unavailable(self, why: str) -> None:
        if not self._rerank_failed:
            self._rerank_failed = True
            logger.warning("Nachos reranking disabled (%s); results keep their fused order.", why)

    def _adopt_ready(self, msg: dict[str, Any]) -> None:
        """Take the dimension, type support and vector-affecting settings from the ready line.

        ``model_id`` is ``nachos:<model>[:<dtype>]`` plus a suffix only when the CLI embeds
        documents differently from a 0.6.0 CLI (non-mean pooling or a document prefix), so an
        upgrade re-embeds exactly the models whose stored vectors changed and leaves MiniLM
        alone. The id is complete only after the worker has started (see :meth:`warm_up`); the
        ready line's own ``model`` string is not adopted because it would vary run to run.
        """
        self._check_ready_version(msg)
        dim = msg.get("dimension")
        if isinstance(dim, int) and not isinstance(dim, bool) and dim > 0:
            self.dimension = dim
        # Type support shipped together with these fields; older CLIs lack all of them.
        self.supports_type = (not self._legacy_cli) and any(
            k in msg for k in ("pooling", "queryPrefix", "documentPrefix")
        )
        self.supports_rerank = (not self._legacy_cli) and isinstance(msg.get("rerank"), dict)
        pooling = msg.get("pooling") if isinstance(msg.get("pooling"), str) else "mean"
        doc_prefix = msg.get("documentPrefix") if isinstance(msg.get("documentPrefix"), str) else ""
        suffix = f":pooling={pooling}" if pooling != "mean" else ""
        suffix += f":dp={doc_prefix!r}" if doc_prefix else ""
        # Reported only when set / non-default, so unconfigured ids stay as they were.
        max_length = msg.get("maxLength")
        if isinstance(max_length, int) and not isinstance(max_length, bool) and max_length > 0:
            suffix += f":ml={max_length}"
        suffix += ":norm=off" if msg.get("normalize") is False else ""
        self.model_id = self._base_id + suffix

    def _drop_worker(self) -> None:
        worker, self._worker = self._worker, None
        if worker is not None:
            worker.stop()

    def warm_up(self) -> None:
        with self._lock:
            if self._worker is None and not self._serve_disabled:
                try:
                    self._worker = self._start_worker()
                except (_WorkerError, OSError):
                    self._serve_disabled = True  # embed() will use the one-shot path

    def close(self) -> None:
        with self._lock:
            self._drop_worker()

    # -- embedding --------------------------------------------------------
    def _request(self, worker: _Worker, texts: list[str], input_type: str) -> list[list[float]]:
        self._next_id += 1
        rid = self._next_id
        payload: dict[str, Any] = {"id": rid, "texts": texts}
        if self._type_enabled():
            payload["type"] = input_type
        worker.send(payload)
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

    def rerank(self, query: str, passages: list[str]) -> list[float] | None:
        """Cross-encoder scores via the serve worker, or None when reranking is unavailable.

        Raises ``RuntimeError`` for an error reply (the caller disables reranking).
        """
        if not self.rerank_enabled or not passages:
            return None
        with self._lock:
            self._ensure_cli_version()
            if self._serve_disabled or self._legacy_cli:
                return None
            try:
                if self._worker is None:
                    self._worker = self._start_worker()
                if not self.supports_rerank:
                    return None
                return self._rerank_request(self._worker, query, passages)
            except (_WorkerError, OSError) as exc:
                self._drop_worker()
                raise RuntimeError(f"nachos-embeddings rerank worker failed: {exc}") from None

    def _rerank_request(self, worker: _Worker, query: str, passages: list[str]) -> list[float]:
        self._next_id += 1
        rid = self._next_id
        worker.send({"id": rid, "op": "rerank", "query": query, "passages": passages})
        while True:
            msg = worker.read(self.request_timeout)
            if msg.get("id") != rid:
                continue
            if "error" in msg:
                raise RuntimeError(f"nachos-embeddings rerank: {str(msg['error'])[:200]}")
            scores = msg.get("scores")
            if not isinstance(scores, list) or len(scores) != len(passages):
                raise _WorkerError("rerank reply without one score per passage")
            return [float(x) for x in scores]

    def _type_enabled(self) -> bool:
        return self.supports_type and self.send_type and not self._legacy_cli

    def _serve_batch(self, texts: list[str], input_type: str) -> list[list[float]] | None:
        """Vectors from the worker, or None if serve mode is unusable."""
        while not self._serve_disabled:
            try:
                if self._worker is None:
                    self._worker = self._start_worker()
                return self._request(self._worker, texts, input_type)
            except (_WorkerError, OSError):  # noqa: PERF203 - retry loop: the try IS the loop body
                was_running = self._worker is not None
                self._drop_worker()
                if not was_running or self._restarts >= 1:
                    self._serve_disabled = True  # unsupported, or died again
                else:
                    self._restarts += 1  # restart once, then retry
        return None

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self._embed_typed(texts, DOCUMENT)

    def _embed_typed(self, texts: list[str], input_type: str) -> list[list[float]]:
        with self._lock:
            vectors = self._serve_batch(texts, input_type)
        if vectors is not None:
            return vectors
        return self._embed_oneshot(texts, input_type)

    def _embed_oneshot(self, texts: list[str], input_type: str = DOCUMENT) -> list[list[float]]:
        with self._lock:
            self._ensure_cli_version()
        cmd = [self.command, "embed", "--stdin", "--json"]
        if self._type_enabled():
            cmd += ["--type", input_type]
        cmd += self._model_args()
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

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key_env: str = "",
        timeout: float = 60.0,
        query_prefix: str = "",
        document_prefix: str = "",
        *,
        credential_resolver: Callable[[str], str | None] | None = None,
    ):
        if not base_url or not model:
            raise ValueError("openai embeddings need embedding_base_url and embedding_model")
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key_env = api_key_env
        self._credential_resolver = credential_resolver
        self.timeout = timeout
        self.model_id = f"openai:{self.base_url}:{model}" + _prefix_suffix(
            query_prefix, document_prefix
        )

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        headers = {"Content-Type": "application/json"}
        key = ""
        if self.api_key_env:
            if self._credential_resolver is None:
                raise RuntimeError("embeddings API key requires a credential resolver")
            key = self._credential_resolver(self.api_key_env) or ""
            if not key:
                raise RuntimeError("embeddings API credential unavailable")
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


def _prefix_suffix(query_prefix: str, document_prefix: str) -> str:
    """Model-id suffix for client-side prefixes, so changing them re-embeds."""
    out = ""
    if query_prefix:
        out += f":qp={query_prefix!r}"
    if document_prefix:
        out += f":dp={document_prefix!r}"
    return out


class SentenceTransformersEmbedder(Embedder):
    def __init__(self, model: str = "", query_prefix: str = "", document_prefix: str = "") -> None:
        self.model = model or "all-MiniLM-L6-v2"
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.model_id = f"st:{self.model}" + _prefix_suffix(query_prefix, document_prefix)
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
    query_prefix: str = "",
    document_prefix: str = "",
    rerank: bool = False,
    rerank_model: str = "",
    rerank_dtype: str = "",
    pooling: str = "",
    max_length: int = 0,
    normalize: bool | None = None,
    rerank_max_length: int = 0,
    credential_resolver: Callable[[str], str | None] | None = None,
) -> Embedder | None:
    """Resolve a configured backend name to an Embedder, or None for lexical-only.

    ``dtype``, ``pooling``, ``max_length``, ``normalize`` and ``rerank*`` apply to
    the ``nachos`` CLI only (``auto`` resolving to it included); a blank ``model`` is the CLI's default
    (MiniLM). ``query_prefix`` / ``document_prefix`` are sent to the nachos
    CLI as overrides of its per-model prefixes (blank = keep the profile's, ``none`` = no
    prefix) and prepended client-side by ``openai`` and ``sentence-transformers``.
    """
    kind = (backend or "auto").strip().lower()
    if kind == "off":
        return None
    prefixes = (_client_prefix(query_prefix), _client_prefix(document_prefix))

    def cli() -> CliEmbedder:
        return CliEmbedder(
            command,
            model,
            dtype=dtype,
            rerank=rerank,
            rerank_model=rerank_model,
            rerank_dtype=rerank_dtype,
            pooling=pooling,
            max_length=max_length,
            normalize=normalize,
            query_prefix=query_prefix,
            document_prefix=document_prefix,
            rerank_max_length=rerank_max_length,
        )

    if kind == "nachos" or (kind == "auto" and shutil.which(command)):
        return cli()
    if kind == "auto":
        return _auto_embedder(model, base_url, api_key_env, prefixes, credential_resolver)
    if kind == "openai":
        return OpenAICompatEmbedder(
            base_url,
            model,
            api_key_env,
            query_prefix=prefixes[0],
            document_prefix=prefixes[1],
            credential_resolver=credential_resolver,
        )
    if kind == "sentence-transformers":
        return SentenceTransformersEmbedder(model, *prefixes)
    raise ValueError(f"Unknown embeddings backend {backend!r}; use one of {EMBEDDING_BACKENDS}")


def _client_prefix(value: str) -> str:
    """Client-side prefix: the ``none`` sentinel means an empty prefix."""
    argument = prefix_argument(value)
    return "" if argument is None else argument


def _auto_embedder(
    model: str,
    base_url: str,
    api_key_env: str,
    prefixes: tuple[str, str],
    credential_resolver: Callable[[str], str | None] | None = None,
) -> Embedder | None:
    """``auto`` without the nachos CLI: sentence-transformers, then OpenAI-compatible."""
    if importlib.util.find_spec("sentence_transformers") is not None:
        return SentenceTransformersEmbedder(model, *prefixes)
    if base_url and model:
        return OpenAICompatEmbedder(
            base_url,
            model,
            api_key_env,
            query_prefix=prefixes[0],
            document_prefix=prefixes[1],
            credential_resolver=credential_resolver,
        )
    return None
