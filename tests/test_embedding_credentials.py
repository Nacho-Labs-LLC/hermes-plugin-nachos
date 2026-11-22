"""Credential boundaries: synthetic keys and loopback-only embedding requests."""

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from nachos_core.embeddings import OpenAICompatEmbedder


@pytest.fixture
def endpoint():
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(self.headers.get("Authorization"))
            payload = json.dumps(
                {"data": [{"index": i, "embedding": [1.0]} for i, _ in enumerate(body["input"])]}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", seen
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_requested_key_without_resolver_never_uses_process_environment(endpoint, monkeypatch):
    url, seen = endpoint
    monkeypatch.setenv("TEST_EMBED_KEY", "synthetic-other-profile")
    embedder = OpenAICompatEmbedder(url, "test-model", api_key_env="TEST_EMBED_KEY")
    with pytest.raises(RuntimeError, match="credential resolver"):
        embedder.embed(["synthetic text"])
    assert seen == []


@pytest.mark.parametrize("backend", ["openai", "auto"])
def test_factory_resolves_credentials_at_each_request(endpoint, monkeypatch, backend):
    from nachos_core.embeddings import make_embedder  # noqa: PLC0415

    url, seen = endpoint
    monkeypatch.setattr("nachos_core.embeddings.shutil.which", lambda _command: None)
    monkeypatch.setattr("nachos_core.embeddings.importlib.util.find_spec", lambda _name: None)
    monkeypatch.setenv("TEST_EMBED_KEY", "synthetic-wrong-profile")
    current = {"TEST_EMBED_KEY": "synthetic-a"}
    embedder = make_embedder(
        backend,
        base_url=url,
        model="test-model",
        api_key_env="TEST_EMBED_KEY",
        credential_resolver=current.get,
    )
    assert seen == []
    assert embedder.embed(["first"]) == [[1.0]]
    current["TEST_EMBED_KEY"] = "synthetic-b"
    assert embedder.embed(["second"]) == [[1.0]]
    assert seen == ["Bearer synthetic-a", "Bearer synthetic-b"]


@pytest.mark.parametrize("missing", [None, ""])
def test_missing_requested_key_does_not_send_unauthenticated_request(endpoint, missing):
    url, seen = endpoint
    embedder = OpenAICompatEmbedder(
        url,
        "test-model",
        api_key_env="TEST_EMBED_KEY",
        credential_resolver=lambda _name: missing,
    )
    with pytest.raises(RuntimeError, match="credential unavailable"):
        embedder.embed(["private text"])
    assert seen == []


def test_unauthenticated_endpoint_does_not_require_or_call_resolver(endpoint):
    url, seen = endpoint

    def unexpected(_name):
        pytest.fail("unauthenticated endpoint must not resolve a credential")

    embedder = OpenAICompatEmbedder(url, "test-model", credential_resolver=unexpected)
    assert embedder.embed(["local text"]) == [[1.0]]
    assert seen == [None]


_HOST_PROBE = r"""
import contextlib
import io
import json
import sys
from pathlib import Path
from agent.secret_scope import (
    UnscopedSecretError, reset_secret_scope, set_multiplex_active, set_secret_scope,
)
from nachos_hermes.memory_provider import NachosMemoryProvider
from nachos_core.store import get_store

url, backend, mode, root = sys.argv[1:]
set_multiplex_active(True)
provider = NachosMemoryProvider()
provider._cfg.update(embeddings=backend, embedding_base_url=url,
                     embedding_model="test-model", embedding_api_key_env="TEST_EMBED_KEY")
# Construct with NO bound scope; credential access must remain lazy.
embedder = provider._make_embedder()
assert embedder is not None
for key in ("synthetic-a", "synthetic-b"):
    token = set_secret_scope({"TEST_EMBED_KEY": key})
    try:
        assert embedder.embed(["scoped text"]) == [[1.0]]
    finally:
        reset_secret_scope(token)
for mapping in ({}, None):
    token = set_secret_scope(mapping)
    try:
        try:
            embedder.embed(["must not leave process"])
        except RuntimeError as exc:
            if mapping is None:
                assert isinstance(exc, UnscopedSecretError)
            else:
                assert "credential unavailable" in str(exc)
        else:
            raise AssertionError("missing credential scope did not fail closed")
    finally:
        reset_secret_scope(token)

# Existing memory service must degrade rather than disabling memory entirely.
from collections import OrderedDict
from nachos_core.index import MemoryIndex
from nachos_core.memory import MemoryService, Scope, PROFILE
path = Path(root) / "lexical.db"
store = get_store("sqlite", path)
for key in ("one", "two"):
    store.put(key, title="same title", summary="same summary", category="test", body="body")
service = MemoryService(OrderedDict([(PROFILE, Scope(PROFILE, store, MemoryIndex(Path(root) / "index.db")))]), embedder=embedder)
try:
    assert service.duplicate_pairs()[0]["match"] == "lexical"
    assert "UnscopedSecretError" in service.embedder_error
finally:
    service.close()

if mode == "audit":
    import memory_audit
    path = Path(root) / "audit.db"
    store = get_store("sqlite", path)
    for key in ("first", "second"):
        store.put(key, title="same title", summary="same summary", category="test", body="body")
    store.close()
    token = set_secret_scope({"TEST_EMBED_KEY": "synthetic-audit"})
    try:
        with contextlib.redirect_stdout(io.StringIO()) as output:
            assert memory_audit.main([
                "--path", str(path), "--embeddings", backend, "--embedding-model", "test-model",
                "--embedding-base-url", url, "--embedding-api-key-env", "TEST_EMBED_KEY",
                "--json", "duplicates",
            ]) == 0
        assert json.loads(output.getvalue())["duplicates"][0]["match"] == "semantic"
    finally:
        reset_secret_scope(token)
print("scoped credentials verified")
"""


@pytest.mark.parametrize("backend", ["openai", "auto"])
@pytest.mark.parametrize("mode", ["provider", "audit"])
def test_real_hermes_request_scopes(endpoint, tmp_path, backend, mode):
    source = os.environ.get("HERMES_SOURCE")
    if not source:
        pytest.skip("set HERMES_SOURCE to a Hermes checkout for host integration")
    assert (Path(source) / "agent" / "secret_scope.py").is_file()
    repo = Path(__file__).resolve().parents[1]
    url, seen = endpoint
    # Deliberate allowlist: never inherit live profile secrets or HOME.
    env = {
        "HOME": str(tmp_path),
        "HERMES_HOME": str(tmp_path / "hermes"),
        "TMPDIR": str(tmp_path),
        "PATH": str(tmp_path),
        "PYTHONPATH": os.pathsep.join((str(repo), str(repo / "tools"), source)),
        "TEST_EMBED_KEY": "synthetic-process-poison",
    }
    result = subprocess.run(
        [sys.executable, "-c", _HOST_PROBE, url, backend, mode, str(tmp_path)],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "scoped credentials verified" in result.stdout
    expected = ["Bearer synthetic-a", "Bearer synthetic-b"]
    if mode == "audit":
        expected.append("Bearer synthetic-audit")
    assert seen == expected


def test_missing_hermes_resolver_fails_closed_and_keeps_lexical_memory(
    endpoint,
    monkeypatch,
    tmp_path,
):
    from collections import OrderedDict  # noqa: PLC0415

    from nachos_core.index import MemoryIndex  # noqa: PLC0415
    from nachos_core.memory import PROFILE, MemoryService, Scope  # noqa: PLC0415
    from nachos_core.store import get_store  # noqa: PLC0415
    from nachos_hermes.config import hermes_credential  # noqa: PLC0415

    monkeypatch.setitem(sys.modules, "agent.secret_scope", None)
    monkeypatch.setenv("TEST_EMBED_KEY", "synthetic-process-poison")
    url, seen = endpoint
    embedder = OpenAICompatEmbedder(
        url,
        "test-model",
        api_key_env="TEST_EMBED_KEY",
        credential_resolver=hermes_credential,
    )
    with pytest.raises(ModuleNotFoundError):
        embedder.embed(["must not leave process"])
    store = get_store("sqlite", tmp_path / "memory.db")
    for key in ("first", "second"):
        store.put(key, title="same", summary="same words", category="test", body="body")
    scope = Scope(PROFILE, store, MemoryIndex(tmp_path / "index.db"))
    service = MemoryService(OrderedDict([(PROFILE, scope)]), embedder=embedder)
    try:
        assert service.duplicate_pairs()[0]["match"] == "lexical"
        assert "ModuleNotFoundError" in service.embedder_error
    finally:
        service.close()
    assert seen == []
