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
    parse_cli_version,
)

_FAKE_CLI = """#!{python}
import json, os, sys
if sys.argv[1:2] == ["--version"]:
    ver = os.environ.get("FAKE_VERSION")
    if ver is not None:
        print(ver)
        sys.exit(0)
    sys.exit(1)
assert sys.argv[1:4] == ["embed", "--stdin", "--json"], sys.argv
if os.environ.get("FAKE_ARGS"):
    open(os.environ["FAKE_ARGS"], "w").write(" ".join(sys.argv[4:]))
for line in sys.stdin.read().splitlines():
    print(json.dumps([float(len(line)), 1.0]))
"""


# Serve-capable fake. Appends a line to $FAKE_COUNTER per process start; if
# $FAKE_DIE_AFTER is set, the first process exits after that many requests.
_FAKE_SERVE_CLI = """#!{python}
import json, os, sys
if sys.argv[1:2] == ["--version"]:
    ver = os.environ.get("FAKE_VERSION")
    if ver is not None:
        print(ver)
        sys.exit(0)
    sys.exit(1)
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
    ready.update(
        pooling=os.environ.get("FAKE_POOLING", "mean"),
        queryPrefix="",
        documentPrefix=os.environ.get("FAKE_DOCPREFIX", ""),
    )
if os.environ.get("FAKE_READY_EXTRA"):
    ready.update(json.loads(os.environ["FAKE_READY_EXTRA"]))
args = sys.argv[2:]
if os.environ.get("FAKE_REJECT_RERANK_FLAGS") and "--rerank-model" in args:
    sys.exit(2)  # an older CLI: unknown flag is a usage error before any ready line
if os.environ.get("FAKE_RERANK"):
    ready["rerank"] = {{"model": "fake-reranker"}}
print(json.dumps(ready), flush=True)
if os.environ.get("FAKE_ARGS"):
    with open(os.environ["FAKE_ARGS"], "w") as fh:
        fh.write(" ".join(args))
served = 0
for line in sys.stdin:
    req = json.loads(line)
    if req.get("op") == "rerank":
        if req["query"] == "boom":
            print(json.dumps({{"id": req["id"], "error": "rerank failed"}}), flush=True)
            continue
        scores = [float(p.count("hit")) for p in req["passages"]]
        print(json.dumps({{"id": req["id"], "scores": scores}}), flush=True)
        continue
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
    monkeypatch.setenv("FAKE_VERSION", "0.7.0")  # a current CLI answers --version
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
    monkeypatch.setenv("FAKE_VERSION", "0.7.0")  # a current CLI answers --version
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


def test_openai_compatible_endpoint():
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
        embedder = OpenAICompatEmbedder(
            f"http://127.0.0.1:{server.server_port}/v1",
            "nomic-embed-text",
            api_key_env="TEST_EMBED_KEY",
            credential_resolver={"TEST_EMBED_KEY": "secret-token"}.get,
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


@pytest.mark.parametrize(
    ("env", "suffix"),
    [
        ({}, ""),  # 0.6.0 CLI: no type fields at all
        ({"FAKE_TYPED": "1"}, ""),  # type-aware but mean pooling, no prefix: vectors unchanged
        ({"FAKE_TYPED": "1", "FAKE_POOLING": "cls"}, ":pooling=cls"),
        ({"FAKE_TYPED": "1", "FAKE_DOCPREFIX": "passage: "}, ":dp='passage: '"),
    ],
)
def test_model_id_reflects_settings_that_change_stored_vectors(
    fake_serve_cli, monkeypatch, env, suffix
):
    path, _ = fake_serve_cli
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    embedder = CliEmbedder(str(path), "m", dtype="q8")
    assert embedder.model_id == "nachos:m:q8"  # before the worker is up
    embedder.warm_up()
    assert embedder.model_id == "nachos:m:q8" + suffix
    embedder.close()


def test_warm_up_without_a_working_cli_does_not_raise(tmp_path):
    embedder = CliEmbedder(str(tmp_path / "missing-cli"))
    embedder.warm_up()
    assert embedder.model_id == "nachos:default"


# -- reranking over the serve protocol ---------------------------------------------


def test_rerank_flags_are_passed_and_scores_returned(fake_serve_cli, monkeypatch, tmp_path):
    path, _ = fake_serve_cli
    args_file = tmp_path / "args.txt"
    monkeypatch.setenv("FAKE_ARGS", str(args_file))
    monkeypatch.setenv("FAKE_RERANK", "1")
    embedder = make_embedder(
        "nachos", command=str(path), rerank=True, rerank_model="r-model", rerank_dtype="q8"
    )
    try:
        assert embedder.rerank("q", ["a hit hit", "none", "one hit"]) == [2.0, 0.0, 1.0]
        assert embedder.supports_rerank is True
        assert embedder.embed(["abc"]) == [[3.0, 2.0]]  # embedding still works on the same worker
    finally:
        embedder.close()
    assert args_file.read_text() == "--rerank-model r-model --rerank-dtype q8"


def test_rerank_uses_the_default_model_when_none_is_configured(
    fake_serve_cli, monkeypatch, tmp_path
):
    path, _ = fake_serve_cli
    args_file = tmp_path / "args.txt"
    monkeypatch.setenv("FAKE_ARGS", str(args_file))
    monkeypatch.setenv("FAKE_RERANK", "1")
    embedder = CliEmbedder(str(path), rerank=True)
    embedder.embed(["x"])  # a round trip guarantees the fake has written its args file
    embedder.close()
    assert args_file.read_text() == "--rerank-model jinaai/jina-reranker-v1-tiny-en"


def test_rerank_is_off_and_flagless_when_not_requested(fake_serve_cli, monkeypatch, tmp_path):
    path, _ = fake_serve_cli
    args_file = tmp_path / "args.txt"
    monkeypatch.setenv("FAKE_ARGS", str(args_file))
    embedder = CliEmbedder(str(path))
    assert embedder.rerank("q", ["a hit"]) is None
    embedder.embed(["x"])
    embedder.close()
    assert args_file.read_text() == ""


def test_ready_line_without_rerank_falls_back_silently_and_logs_once(fake_serve_cli, caplog):
    path, _ = fake_serve_cli  # worker accepts the flags but announces no "rerank"
    embedder = CliEmbedder(str(path), rerank=True)
    try:
        with caplog.at_level("WARNING", logger="nachos_core.embeddings"):
            assert embedder.rerank("q", ["a hit"]) is None
            assert embedder.rerank("q", ["a hit"]) is None
        assert embedder.supports_rerank is False
        assert embedder.embed(["ab"]) == [[2.0, 2.0]]
    finally:
        embedder.close()
    assert sum("reranking disabled" in r.message for r in caplog.records) == 1


def test_old_cli_that_rejects_the_flags_still_embeds_in_serve_mode(
    fake_serve_cli, monkeypatch, caplog
):
    path, counter = fake_serve_cli
    monkeypatch.setenv("FAKE_REJECT_RERANK_FLAGS", "1")
    embedder = CliEmbedder(str(path), rerank=True)
    try:
        with caplog.at_level("WARNING", logger="nachos_core.embeddings"):
            assert embedder.embed(["abc"]) == [[3.0, 2.0]]
        assert embedder.rerank("q", ["x"]) is None
        assert embedder._serve_disabled is False
    finally:
        embedder.close()
    assert counter.read_text().count("start") == 2  # rejected launch, then the plain one
    assert sum("reranking disabled" in r.message for r in caplog.records) == 1


def test_rerank_error_reply_raises_but_keeps_the_worker(fake_serve_cli, monkeypatch):
    path, _ = fake_serve_cli
    monkeypatch.setenv("FAKE_RERANK", "1")
    embedder = CliEmbedder(str(path), rerank=True)
    try:
        with pytest.raises(RuntimeError, match="rerank failed"):
            embedder.rerank("boom", ["a"])
        assert embedder.rerank("q", ["a hit"]) == [1.0]
    finally:
        embedder.close()


def test_overrides_are_passed_to_serve_and_oneshot(fake_serve_cli, fake_cli, monkeypatch, tmp_path):
    path, _ = fake_serve_cli
    args_file = tmp_path / "args.txt"
    monkeypatch.setenv("FAKE_ARGS", str(args_file))
    embedder = CliEmbedder(
        str(path),
        "org/m",
        dtype="q4",
        pooling="cls",
        max_length=128,
        normalize=False,
        query_prefix="Q: ",
        document_prefix="none",
    )
    try:
        embedder.embed(["ab"])
        parts = args_file.read_text().split(" --")
        assert "pooling cls" in parts
        assert "max-length 128" in parts
        assert "no-normalize" in parts
        assert "query-prefix Q: " in parts  # leading/trailing spaces survive
        assert any(
            p.startswith("document-prefix") and p.strip() == "document-prefix" for p in parts
        )
    finally:
        embedder.close()
    # The one-shot fallback sends the same model flags.
    oneshot = CliEmbedder(str(fake_cli), "org/m", pooling="mean", normalize=True)
    assert oneshot._model_args() == ["--model", "org/m", "--pooling", "mean", "--normalize"]


def test_unset_overrides_send_nothing_extra():
    embedder = CliEmbedder("x", "m")
    assert embedder._model_args() == ["--model", "m"]
    assert CliEmbedder("x")._model_args() == []


def test_prefix_none_disables_and_blank_keeps_the_profile():
    off = CliEmbedder("x", query_prefix="none", document_prefix="  NONE ")
    assert off._model_args() == ["--query-prefix", "", "--document-prefix", ""]
    assert CliEmbedder("x", query_prefix="", document_prefix="")._model_args() == []


def test_ready_line_overrides_change_the_model_id(fake_serve_cli, monkeypatch):
    path, _ = fake_serve_cli
    monkeypatch.setenv("FAKE_TYPED", "1")
    monkeypatch.setenv("FAKE_READY_EXTRA", json.dumps({"maxLength": 256, "normalize": False}))
    embedder = CliEmbedder(str(path), "m", dtype="q8", max_length=256, normalize=False)
    try:
        embedder.warm_up()
        assert embedder.model_id == "nachos:m:q8:ml=256:norm=off"
    finally:
        embedder.close()


def test_blank_model_is_the_backend_default_and_explicit_model_dtype_are_used():
    assert make_embedder("nachos").model_id == "nachos:default"
    assert make_embedder("nachos", model="org/custom").model_id == "nachos:org/custom"
    both = make_embedder("nachos", model="org/custom", dtype="q4")
    assert both.model_id == "nachos:org/custom:q4"


def test_rerank_max_length_is_passed_to_serve(fake_serve_cli, monkeypatch, tmp_path):
    path, _ = fake_serve_cli
    args_file = tmp_path / "args.txt"
    monkeypatch.setenv("FAKE_ARGS", str(args_file))
    monkeypatch.setenv("FAKE_RERANK", "1")
    embedder = CliEmbedder(
        str(path), rerank=True, rerank_model="r", rerank_dtype="q8", rerank_max_length=2048
    )
    try:
        embedder.warm_up()
        assert "--rerank-max-length 2048" in args_file.read_text()
    finally:
        embedder.close()


def test_none_prefix_means_no_prefix_for_client_side_backends():
    st = make_embedder("sentence-transformers", model="")
    assert st.model_id == "st:all-MiniLM-L6-v2"
    openai = make_embedder(
        "openai",
        model="nomic-embed-text",
        base_url="http://h/v1",
        query_prefix="none",
        document_prefix="passage: ",
    )
    assert openai.query_prefix == ""  # the 'none' sentinel means no prefix here too
    assert openai.document_prefix == "passage: "


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("0.6.0", (0, 6, 0)),
        ("v0.5.2\n", (0, 5, 2)),
        ("nachos-embeddings 0.6.1-beta.1", (0, 6, 1)),
        ("", None),
        ("usage: nachos-embeddings\n  version 0.6.0", None),
        ("not a version", None),
    ],
)
def test_parse_cli_version(raw, expected):
    assert parse_cli_version(raw) == expected


def test_older_cli_version_degrades_gracefully_and_warns_once(
    fake_serve_cli, monkeypatch, tmp_path, caplog
):
    path, _ = fake_serve_cli
    args_file = tmp_path / "args.txt"
    reqs_file = tmp_path / "reqs.txt"
    monkeypatch.setenv("FAKE_VERSION", "nachos-embeddings 0.5.2")
    monkeypatch.setenv("FAKE_TYPED", "1")
    monkeypatch.setenv("FAKE_RERANK", "1")
    monkeypatch.setenv("FAKE_ARGS", str(args_file))
    monkeypatch.setenv("FAKE_REQS", str(reqs_file))
    embedder = CliEmbedder(
        str(path),
        "org/m",
        dtype="q8",
        rerank=True,
        pooling="cls",
        max_length=256,
        normalize=False,
        query_prefix="Q: ",
        document_prefix="D: ",
    )
    try:
        with caplog.at_level("WARNING", logger="nachos_core.embeddings"):
            assert embedder.embed(["hello"], input_type="query") == [[5.0, 2.0]]
            assert embedder.embed(["world"], input_type="document") == [[5.0, 2.0]]
            assert embedder.rerank("q", ["hit"]) is None
        assert embedder.cli_version == (0, 5, 2)
        assert embedder.supports_type is False
        assert embedder.supports_rerank is False
        assert args_file.read_text() == "--model org/m --dtype q8"
        assert reqs_file.read_text().splitlines() == ["None", "None"]
    finally:
        embedder.close()
    warnings = [r.message for r in caplog.records if "older than 0.7.0" in r.message]
    assert len(warnings) == 1


def test_older_cli_ready_line_version_disables_types_and_rerank(
    fake_serve_cli, monkeypatch, tmp_path, caplog
):
    path, _ = fake_serve_cli
    reqs_file = tmp_path / "reqs.txt"
    monkeypatch.setenv("FAKE_TYPED", "1")
    monkeypatch.setenv("FAKE_RERANK", "1")
    monkeypatch.setenv("FAKE_READY_EXTRA", json.dumps({"version": "0.5.9"}))
    monkeypatch.setenv("FAKE_REQS", str(reqs_file))
    embedder = CliEmbedder(str(path), rerank=True)
    try:
        with caplog.at_level("WARNING", logger="nachos_core.embeddings"):
            assert embedder.embed(["abc"], input_type="query") == [[3.0, 2.0]]
            assert embedder.rerank("q", ["hit"]) is None
        assert embedder.cli_version == (0, 5, 9)
        assert embedder.supports_type is False
        assert embedder.supports_rerank is False
        assert reqs_file.read_text().splitlines() == ["None"]
    finally:
        embedder.close()
    assert sum("older than 0.7.0" in r.message for r in caplog.records) == 1
    assert not any("reranking disabled" in r.message for r in caplog.records)


def test_older_oneshot_cli_strips_0_6_override_flags(fake_cli, monkeypatch, tmp_path, caplog):
    args_file = tmp_path / "args.txt"
    monkeypatch.setenv("FAKE_VERSION", "0.5.0")
    monkeypatch.setenv("FAKE_ARGS", str(args_file))
    embedder = CliEmbedder(str(fake_cli), "org/m", dtype="q8", pooling="cls", max_length=128)
    with caplog.at_level("WARNING", logger="nachos_core.embeddings"):
        assert embedder.embed(["abc"]) == [[3.0, 1.0]]
        assert embedder.embed(["de"]) == [[2.0, 1.0]]
    assert args_file.read_text() == "--model org/m --dtype q8"
    assert sum("older than 0.7.0" in r.message for r in caplog.records) == 1


def test_cli_version_0_6_0_keeps_full_features(fake_serve_cli, monkeypatch, tmp_path, caplog):
    path, _ = fake_serve_cli
    args_file = tmp_path / "args.txt"
    reqs_file = tmp_path / "reqs.txt"
    monkeypatch.setenv("FAKE_VERSION", "0.7.0")
    monkeypatch.setenv("FAKE_TYPED", "1")
    monkeypatch.setenv("FAKE_RERANK", "1")
    monkeypatch.setenv("FAKE_ARGS", str(args_file))
    monkeypatch.setenv("FAKE_REQS", str(reqs_file))
    embedder = CliEmbedder(str(path), "org/m", rerank=True, pooling="cls", max_length=256)
    try:
        with caplog.at_level("WARNING", logger="nachos_core.embeddings"):
            assert embedder.embed(["abc"], input_type="query") == [[3.0, 2.0]]
            assert embedder.rerank("q", ["no", "hit hit"]) == [0.0, 2.0]
        assert embedder.cli_version == (0, 7, 0)
        assert embedder.supports_type is True
        assert embedder.supports_rerank is True
        assert "--pooling cls" in args_file.read_text()
        assert "--max-length 256" in args_file.read_text()
        assert "--rerank-model" in args_file.read_text()
        assert reqs_file.read_text().splitlines() == ["query"]
    finally:
        embedder.close()
    assert not caplog.records


def test_cli_without_version_flag_is_treated_as_older_and_warns_once(
    fake_serve_cli, monkeypatch, tmp_path, caplog
):
    path, _ = fake_serve_cli
    args_file = tmp_path / "args.txt"
    reqs_file = tmp_path / "reqs.txt"
    monkeypatch.delenv("FAKE_VERSION")  # the fake exits 1 on --version, like a 0.5.x CLI
    monkeypatch.setenv("FAKE_TYPED", "1")
    monkeypatch.setenv("FAKE_RERANK", "1")
    monkeypatch.setenv("FAKE_ARGS", str(args_file))
    monkeypatch.setenv("FAKE_REQS", str(reqs_file))
    embedder = CliEmbedder(str(path), "org/m", dtype="q8", rerank=True, pooling="cls")
    try:
        with caplog.at_level("WARNING", logger="nachos_core.embeddings"):
            assert embedder.embed(["abc"], input_type="query") == [[3.0, 2.0]]
            assert embedder.embed(["de"], input_type="document") == [[2.0, 2.0]]
            assert embedder.rerank("q", ["hit"]) is None
        assert embedder.cli_version is None
        assert embedder.supports_type is False
        assert embedder.supports_rerank is False
        assert args_file.read_text() == "--model org/m --dtype q8"
        assert reqs_file.read_text().splitlines() == ["None", "None"]
    finally:
        embedder.close()
    assert sum("older than 0.7.0" in r.message for r in caplog.records) == 1


def test_ready_line_version_overrides_a_failed_version_probe(fake_serve_cli, monkeypatch, tmp_path):
    path, _ = fake_serve_cli
    reqs_file = tmp_path / "reqs.txt"
    monkeypatch.delenv("FAKE_VERSION")
    monkeypatch.setenv("FAKE_TYPED", "1")
    monkeypatch.setenv("FAKE_READY_EXTRA", json.dumps({"version": "0.7.0"}))
    monkeypatch.setenv("FAKE_REQS", str(reqs_file))
    embedder = CliEmbedder(str(path))
    try:
        assert embedder.embed(["abc"], input_type="query") == [[3.0, 2.0]]
        assert embedder.cli_version == (0, 7, 0)
        assert embedder.supports_type is True
        assert reqs_file.read_text().splitlines() == ["query"]
    finally:
        embedder.close()


def test_missing_cli_binary_is_not_mistaken_for_an_old_cli(tmp_path, caplog):
    embedder = CliEmbedder(str(tmp_path / "no-such-nachos-embeddings"), timeout=2.0)
    with (
        caplog.at_level("WARNING", logger="nachos_core.embeddings"),
        pytest.raises(Exception),  # noqa: B017, PT011 - any backend failure; the service degrades
    ):
        embedder.embed(["abc"])
    assert embedder.cli_version is None
    assert not any("older than" in r.message for r in caplog.records)
