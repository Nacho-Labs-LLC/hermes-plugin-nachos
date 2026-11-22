"""Embedding backends: CLI contract, OpenAI-compatible HTTP, auto-selection."""

import json
import os
import stat
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from nachos_core.embeddings import (
    CliEmbedder,
    Embedder,
    OpenAICompatEmbedder,
    make_embedder,
)

_FAKE_CLI = """#!{python}
import json, sys
assert sys.argv[1:4] == ["embed", "--stdin", "--json"], sys.argv
import os
if os.environ.get("FAKE_ARGS"):
    open(os.environ["FAKE_ARGS"], "w").write(" ".join(sys.argv[4:]))
for line in sys.stdin.read().splitlines():
    print(json.dumps([float(len(line)), 1.0]))
"""


# Serve-capable fake. Appends a line to $FAKE_COUNTER per process start; if
# $FAKE_DIE_AFTER is set, the first process exits after that many requests.
_FAKE_SERVE_CLI = """#!{python}
import json, os, sys
if sys.argv[1] != "serve":
    sys.exit(1)
counter = os.environ["FAKE_COUNTER"]
with open(counter, "a") as fh:
    fh.write("start\\n")
starts = open(counter).read().count("start")
die_after = int(os.environ.get("FAKE_DIE_AFTER", "0")) if starts == 1 else 0
ready = {{"ready": True, "model": "fake"}}
if os.environ.get("FAKE_DIM"):
    ready["dimension"] = int(os.environ["FAKE_DIM"])
if os.environ.get("FAKE_TYPED"):
    ready.update(pooling="mean", queryPrefix="", documentPrefix="")
print(json.dumps(ready), flush=True)
args = sys.argv[2:]
if os.environ.get("FAKE_ARGS"):
    with open(os.environ["FAKE_ARGS"], "w") as fh:
        fh.write(" ".join(args))
served = 0
for line in sys.stdin:
    req = json.loads(line)
    if os.environ.get("FAKE_REQS"):
        with open(os.environ["FAKE_REQS"], "a") as fh:
            fh.write(str(req.get("type")) + "\\n")
    if req["texts"] == ["boom"]:
        print(json.dumps({{"id": req["id"], "error": "bad input"}}), flush=True)
        continue
    vecs = [[float(len(t)), 2.0] for t in req["texts"]]
    print(json.dumps({{"id": req["id"], "vectors": vecs}}), flush=True)
    served += 1
    if die_after and served >= die_after:
        sys.exit(0)
"""


@pytest.fixture
def fake_serve_cli(tmp_path, monkeypatch):
    path = tmp_path / "serve-bin" / "nachos-embeddings"
    path.parent.mkdir()
    path.write_text(_FAKE_SERVE_CLI.format(python=sys.executable))
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    counter = tmp_path / "starts.txt"
    monkeypatch.setenv("FAKE_COUNTER", str(counter))
    return path, counter


def test_serve_worker_is_started_once_across_calls(fake_serve_cli):
    path, counter = fake_serve_cli
    embedder = CliEmbedder(str(path))
    try:
        assert embedder.embed(["abc"]) == [[3.0, 2.0]]
        assert embedder.embed(["a", "bb"]) == [[1.0, 2.0], [2.0, 2.0]]
        assert embedder.embed(["xxxx"]) == [[4.0, 2.0]]
    finally:
        embedder.close()
    assert counter.read_text().count("start") == 1
    assert embedder._worker is None


def test_dimension_is_read_from_the_ready_line(fake_serve_cli, monkeypatch):
    path, _ = fake_serve_cli
    monkeypatch.setenv("FAKE_DIM", "2")
    embedder = CliEmbedder(str(path))
    assert embedder.dimension is None
    assert embedder.embed(["ab"]) == [[2.0, 2.0]]
    assert embedder.dimension == 2
    embedder.close()


def test_vector_length_mismatch_raises(fake_serve_cli, monkeypatch):
    path, _ = fake_serve_cli
    monkeypatch.setenv("FAKE_DIM", "384")
    embedder = CliEmbedder(str(path))
    with pytest.raises(RuntimeError, match="expected 384-dimensional vectors, got 2"):
        embedder.embed(["ab"])
    embedder.close()


def test_missing_dimension_skips_the_check(fake_serve_cli):
    path, _ = fake_serve_cli
    embedder = CliEmbedder(str(path))
    assert embedder.embed(["ab"]) == [[2.0, 2.0]]
    assert embedder.dimension is None
    embedder.close()


def test_dtype_is_passed_to_the_cli_and_kept_in_the_model_id(fake_serve_cli, monkeypatch, tmp_path):
    path, _ = fake_serve_cli
    args_file = tmp_path / "args.txt"
    monkeypatch.setenv("FAKE_ARGS", str(args_file))
    embedder = make_embedder("nachos", command=str(path), model="m", dtype="q8")
    assert embedder.model_id == "nachos:m:q8"
    embedder.embed(["ab"])
    assert args_file.read_text() == "--model m --dtype q8"
    assert make_embedder("nachos", command=str(path)).model_id == "nachos:default"
    embedder.close()


def test_serve_error_reply_raises_without_restart(fake_serve_cli):
    path, counter = fake_serve_cli
    embedder = CliEmbedder(str(path))
    try:
        with pytest.raises(RuntimeError, match="bad input"):
            embedder.embed(["boom"])
        assert embedder.embed(["ok"]) == [[2.0, 2.0]]
    finally:
        embedder.close()
    assert counter.read_text().count("start") == 1


def test_falls_back_to_oneshot_when_serve_is_unsupported(fake_cli):
    embedder = CliEmbedder(str(fake_cli))
    assert embedder.embed(["abc"]) == [[3.0, 1.0]]
    assert embedder._serve_disabled is True
    assert embedder.embed(["ab"]) == [[2.0, 1.0]]


def test_worker_is_restarted_once_after_it_exits(fake_serve_cli, monkeypatch):
    path, counter = fake_serve_cli
    monkeypatch.setenv("FAKE_DIE_AFTER", "1")
    embedder = CliEmbedder(str(path))
    try:
        assert embedder.embed(["abc"]) == [[3.0, 2.0]]
        # The first worker exited after one request; this call restarts it.
        assert embedder.embed(["abcd"]) == [[4.0, 2.0]]
        assert embedder.embed(["a"]) == [[1.0, 2.0]]
    finally:
        embedder.close()
    assert counter.read_text().count("start") == 2
    assert embedder._serve_disabled is False


def test_hung_worker_times_out_and_falls_back(tmp_path):
    hang = tmp_path / "nachos-embeddings"
    hang.write_text(
        f"#!{sys.executable}\nimport sys, time, json\n"
        "if sys.argv[1] == 'serve':\n"
        "    print(json.dumps({'ready': True}), flush=True)\n"
        "    time.sleep(60)\n"
        "for line in sys.stdin.read().splitlines():\n"
        "    print(json.dumps([float(len(line))]))\n"
    )
    hang.chmod(hang.stat().st_mode | stat.S_IEXEC)
    embedder = CliEmbedder(str(hang), request_timeout=0.5)
    try:
        assert embedder.embed(["abc"]) == [[3.0]]
    finally:
        embedder.close()
    assert embedder._serve_disabled is True


@pytest.fixture
def fake_cli(tmp_path, monkeypatch):
    path = tmp_path / "bin" / "nachos-embeddings"
    path.parent.mkdir()
    path.write_text(_FAKE_CLI.format(python=sys.executable))
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{path.parent}{os.pathsep}{os.environ.get('PATH', '')}")
    return path


def test_cli_embedder_round_trips_one_vector_per_line(fake_cli):
    vectors = CliEmbedder(str(fake_cli)).embed(["abc", "multi\nline text"])
    # Newlines are flattened so one input text is always one stdin line.
    assert vectors == [[3.0, 1.0], [15.0, 1.0]]


@pytest.mark.usefixtures("fake_cli")
def test_auto_prefers_the_cli_when_it_is_on_path():
    assert isinstance(make_embedder("auto"), CliEmbedder)


def test_auto_without_any_backend_is_lexical_only(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setattr("importlib.util.find_spec", lambda _name: None)
    assert make_embedder("auto") is None
    assert make_embedder("off") is None


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="Unknown embeddings backend"):
        make_embedder("word2vec")


def test_openai_compatible_endpoint(monkeypatch):
    seen = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen["path"], seen["body"] = self.path, body
            seen["auth"] = self.headers.get("Authorization")
            data = [{"index": i, "embedding": [float(i), 0.5]} for i in range(len(body["input"]))]
            payload = json.dumps({"data": list(reversed(data))}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        monkeypatch.setenv("TEST_EMBED_KEY", "secret-token")
        embedder = OpenAICompatEmbedder(
            f"http://127.0.0.1:{server.server_port}/v1",
            "nomic-embed-text",
            api_key_env="TEST_EMBED_KEY",
        )
        vectors = embedder.embed(["a", "b"])
    finally:
        server.shutdown()

    assert vectors == [[0.0, 0.5], [1.0, 0.5]]  # re-ordered by index
    assert seen["path"] == "/v1/embeddings"
    assert seen["body"] == {"model": "nomic-embed-text", "input": ["a", "b"]}
    assert seen["auth"] == "Bearer secret-token"


def test_openai_backend_requires_url_and_model():
    with pytest.raises(ValueError, match="need embedding_base_url"):
        make_embedder("openai", model="", base_url="")


# -- query / document input types ---------------------------------------------


def _types_seen(tmp_path):
    return (tmp_path / "reqs.txt").read_text().split()


def test_type_is_sent_when_the_ready_line_shows_support(fake_serve_cli, monkeypatch, tmp_path):
    path, _ = fake_serve_cli
    monkeypatch.setenv("FAKE_TYPED", "1")
    monkeypatch.setenv("FAKE_REQS", str(tmp_path / "reqs.txt"))
    embedder = CliEmbedder(str(path))
    try:
        embedder.embed(["a doc"])
        embedder.embed(["a question"], "query")
        embedder.embed(["b"], input_type="document")
    finally:
        embedder.close()
    assert embedder.supports_type is True
    assert _types_seen(tmp_path) == ["document", "query", "document"]


def test_type_is_omitted_for_a_cli_without_type_support(fake_serve_cli, monkeypatch, tmp_path):
    path, _ = fake_serve_cli  # ready line has no pooling / prefix fields (0.6.0 CLI)
    monkeypatch.setenv("FAKE_REQS", str(tmp_path / "reqs.txt"))
    embedder = CliEmbedder(str(path))
    try:
        assert embedder.embed(["q"], "query") == [[1.0, 2.0]]
    finally:
        embedder.close()
    assert embedder.supports_type is False
    assert _types_seen(tmp_path) == ["None"]


def test_send_type_false_keeps_legacy_requests(fake_serve_cli, monkeypatch, tmp_path):
    path, _ = fake_serve_cli
    monkeypatch.setenv("FAKE_TYPED", "1")
    monkeypatch.setenv("FAKE_REQS", str(tmp_path / "reqs.txt"))
    embedder = CliEmbedder(str(path))
    embedder.send_type = False
    try:
        embedder.embed(["q"], "query")
    finally:
        embedder.close()
    assert _types_seen(tmp_path) == ["None"]


def test_oneshot_gets_type_only_when_supported(fake_cli, monkeypatch, tmp_path):
    args_file = tmp_path / "args.txt"
    monkeypatch.setenv("FAKE_ARGS", str(args_file))
    embedder = CliEmbedder(str(fake_cli))
    embedder.embed(["q"], "query")  # serve unsupported, type support unknown: no --type
    assert args_file.read_text() == ""
    embedder.supports_type = True
    embedder.embed(["q"], "query")
    assert args_file.read_text() == "--type query"


def test_invalid_input_type_is_rejected(fake_serve_cli):
    path, _ = fake_serve_cli
    with pytest.raises(ValueError, match="input_type"):
        CliEmbedder(str(path)).embed(["x"], "passage")


class _Recorder(Embedder):
    model_id = "rec"

    def __init__(self, **kw):
        self.__dict__.update(kw)
        self.seen = []

    def _embed_batch(self, texts):
        self.seen.append(texts)
        return [[1.0] for _ in texts]


def test_client_side_prefixes_are_prepended_per_type():
    rec = _Recorder(query_prefix="query: ", document_prefix="passage: ")
    rec.embed(["alpha\nbeta"], "query")
    rec.embed(["gamma"])
    assert rec.seen == [["query: alpha beta"], ["passage: gamma"]]


def test_prefixes_are_part_of_the_model_id_for_st_and_openai():
    plain = make_embedder("sentence-transformers", model="m")
    prefixed = make_embedder(
        "sentence-transformers", model="m", query_prefix="query: ", document_prefix="passage: "
    )
    assert plain.model_id == "st:m"
    assert prefixed.model_id != plain.model_id
    assert "query: " in prefixed.model_id
    assert "passage: " in prefixed.model_id
    oa = make_embedder("openai", model="m", base_url="http://x/v1", document_prefix="d ")
    assert oa.model_id != make_embedder("openai", model="m", base_url="http://x/v1").model_id


def test_nachos_backend_ignores_client_prefixes(fake_serve_cli):
    path, _ = fake_serve_cli
    embedder = make_embedder("nachos", command=str(path), query_prefix="query: ")
    assert embedder.query_prefix == ""
    assert embedder.model_id == "nachos:default"
