"""Presets, setting resolution, and the ``python -m nachos_hermes.models`` command."""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import pytest

from nachos_core import reindex
from nachos_core.embeddings import CliEmbedder, Embedder
from nachos_core.index import MemoryIndex, index_path_for
from nachos_core.memory import PROFILE, MemoryService, Scope
from nachos_core.models import PRESET_CHOICES, PRESETS, prefix_argument, resolve_model
from nachos_core.store import get_store
from nachos_hermes import models as models_cli
from nachos_hermes.commands import status_text
from nachos_hermes.config import (
    AUXILIARY_TASKS,
    DEFAULTS,
    apply_config,
    auxiliary_models,
    make_embedder_from_config,
    profile_config,
)


class TestPresets:
    def test_three_presets_with_pinned_q8_models(self) -> None:
        assert set(PRESETS) == {"small", "balanced", "quality"}
        assert PRESET_CHOICES == ("", "small", "balanced", "quality")
        assert PRESETS["small"].model == "Xenova/all-MiniLM-L6-v2"
        assert PRESETS["quality"].model == "Alibaba-NLP/gte-modernbert-base"
        assert all(
            p.dtype == "q8" and p.download_mb > 0 and p.dimension > 0 for p in PRESETS.values()
        )

    def test_explicit_settings_beat_the_preset(self) -> None:
        assert resolve_model("", "", "") == ("", "")
        assert resolve_model("small", "", "") == ("Xenova/all-MiniLM-L6-v2", "q8")
        assert resolve_model("small", "", "fp16") == ("Xenova/all-MiniLM-L6-v2", "fp16")
        # A custom model never inherits the preset's dtype (it may not ship that file).
        assert resolve_model("quality", "org/m", "") == ("org/m", "")
        assert resolve_model("quality", "org/m", "q4") == ("org/m", "q4")
        assert resolve_model("nope", "", "") == ("", "")
        assert resolve_model(" Quality ", "", "") == (PRESETS["quality"].model, "q8")

    def test_prefix_argument(self) -> None:
        assert prefix_argument("") is None
        assert prefix_argument("query: ") == "query: "
        assert prefix_argument("none") == ""
        assert prefix_argument(" NONE ") == ""


class TestConfig:
    def test_new_keys_have_defaults_and_validate_choices(self) -> None:
        cfg = dict(DEFAULTS)
        assert cfg["embedding_preset"] == ""
        assert cfg["embedding_max_length"] == 0
        apply_config(
            cfg,
            {
                "embedding_preset": "quality",
                "embedding_pooling": "cls",
                "embedding_max_length": 256,
                "rerank_max_length": 2048,
                "reindex_budget_ms": 100,
            },
        )
        assert cfg["embedding_preset"] == "quality"
        assert cfg["embedding_pooling"] == "cls"
        assert cfg["embedding_max_length"] == 256
        assert cfg["rerank_max_length"] == 2048
        assert cfg["reindex_budget_ms"] == 100
        apply_config(cfg, {"embedding_preset": "huge", "embedding_pooling": "max"})
        assert cfg["embedding_preset"] == "quality"  # invalid choices are ignored
        assert cfg["embedding_pooling"] == "cls"

    def test_cfg_resolves_to_an_embedder(self) -> None:
        cfg = dict(DEFAULTS)
        cfg.update(
            embeddings="nachos",
            embedding_preset="balanced",
            embedding_pooling="cls",
            embedding_max_length=64,
            embedding_query_prefix="none",
            rerank="nachos",
            rerank_max_length=2048,
        )
        embedder = make_embedder_from_config(cfg)
        assert isinstance(embedder, CliEmbedder)
        assert embedder.model_id == f"nachos:{PRESETS['balanced'].model}:q8"
        assert embedder._model_args()[-2:] == ["--query-prefix", ""]
        assert embedder.rerank_max_length == 2048
        cfg["embeddings"] = "off"
        assert make_embedder_from_config(cfg) is None

    def test_profile_config_reads_config_json(self, tmp_path: Path) -> None:
        (tmp_path / "nachos").mkdir()
        (tmp_path / "nachos" / "config.json").write_text(
            json.dumps({"embedding_preset": "small", "embeddings": "nachos"})
        )
        cfg = profile_config(tmp_path)
        assert cfg["embedding_preset"] == "small"
        assert profile_config(tmp_path / "missing")["embedding_preset"] == ""

    def test_auxiliary_models_are_named_and_default_to_unset_without_hermes(self) -> None:
        assert set(AUXILIARY_TASKS) == {"nachos_capture", "nachos_summary"}
        assert auxiliary_models() == {"nachos_capture": None, "nachos_summary": None}


class _Words(Embedder):
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t)), 1.0] for t in texts]


def _profile_with_entries(home: Path, count: int, model_id: str) -> None:
    (home / "nachos").mkdir(parents=True, exist_ok=True)
    path = home / "nachos" / "memory.db"
    scope = Scope(PROFILE, get_store("sqlite", path), MemoryIndex(index_path_for(path)))
    service = MemoryService(OrderedDict([(PROFILE, scope)]), embedder=_Words(model_id))
    for i in range(count):
        service.put(
            title=f"Entry number {i}",
            body=f"A reasonably long body for entry {i} so it is not low signal at all.",
            summary=f"summary of entry number {i}",
        )
    service.close()


class TestCommand:
    def test_list_shows_presets_settings_and_auxiliary_llms(self, tmp_path: Path, capsys) -> None:
        (tmp_path / "nachos").mkdir()
        (tmp_path / "nachos" / "config.json").write_text(
            json.dumps({"embedding_preset": "quality", "rerank_model": "org/r"})
        )
        assert models_cli.main(["list", "--profile-home", str(tmp_path), "--json"]) == 0
        report = json.loads(capsys.readouterr().out)
        assert set(report["presets"]) == {"small", "balanced", "quality"}
        assert report["settings"]["embedding_preset"] == "quality"
        assert report["settings"]["rerank_model"] == "org/r"
        assert report["effective_embedding"] == {
            "model": "Alibaba-NLP/gte-modernbert-base",
            "dtype": "q8",
        }
        assert set(report["auxiliary_llm"]) == {"nachos_capture", "nachos_summary"}
        assert models_cli.main(["list"]) == 0  # works without a profile too
        text = capsys.readouterr().out
        assert "auxiliary.nachos_capture.model" in text
        assert "auxiliary.nachos_summary.model" in text
        assert "balanced" in text

    def test_status_reports_coverage_and_flags_a_needed_reindex(
        self, tmp_path: Path, capsys
    ) -> None:
        _profile_with_entries(tmp_path, 4, "nachos:Xenova/all-MiniLM-L6-v2:q8")
        (tmp_path / "nachos" / "config.json").write_text(
            json.dumps({"embeddings": "nachos", "embedding_preset": "small"})
        )
        assert models_cli.main(["status", "--profile-home", str(tmp_path), "--json"]) == 0
        ok = json.loads(capsys.readouterr().out)
        assert ok["entries"] == 4
        assert ok["coverage"] == 1.0
        assert ok["needs_reindex"] is False

        (tmp_path / "nachos" / "config.json").write_text(
            json.dumps({"embeddings": "nachos", "embedding_preset": "quality"})
        )
        assert models_cli.main(["status", "--profile-home", str(tmp_path), "--json"]) == 0
        switched = json.loads(capsys.readouterr().out)
        assert switched["coverage"] == 0.0
        assert switched["needs_reindex"] is True
        assert switched["stale_models"] == ["nachos:Xenova/all-MiniLM-L6-v2:q8"]
        assert models_cli.main(["status", "--profile-home", str(tmp_path)]) == 0
        assert "reindex" in capsys.readouterr().out

    def test_status_is_read_only(self, tmp_path: Path) -> None:
        _profile_with_entries(tmp_path, 2, "nachos:m")
        db = tmp_path / "nachos" / "memory.index.db"
        before = db.stat().st_mtime_ns
        models_cli.main(["status", "--profile-home", str(tmp_path)])
        assert db.stat().st_mtime_ns == before

    def test_embeddings_off_never_needs_reindex(self, tmp_path: Path, capsys) -> None:
        _profile_with_entries(tmp_path, 2, "nachos:m")
        (tmp_path / "nachos" / "config.json").write_text(json.dumps({"embeddings": "off"}))
        models_cli.main(["status", "--profile-home", str(tmp_path), "--json"])
        assert json.loads(capsys.readouterr().out)["needs_reindex"] is False

    def test_reindex_runs_the_configured_embedder_to_completion(
        self, tmp_path: Path, capsys, monkeypatch
    ) -> None:
        _profile_with_entries(tmp_path, 30, "old-model")
        monkeypatch.setattr(
            models_cli, "make_embedder_from_config", lambda _cfg: _Words("new-model")
        )
        assert models_cli.main(["reindex", "--profile-home", str(tmp_path)]) == 0
        report = json.loads(capsys.readouterr().out)
        assert report["state"] == "ready"
        scope = report["scopes"][PROFILE]
        assert scope["embedded"] == 30
        assert scope["other_models"] == {}

    def test_errors_are_reported_as_json(self, tmp_path: Path, capsys) -> None:
        assert models_cli.main(["status", "--profile-home", "relative/path"]) == 2
        assert "absolute" in capsys.readouterr().out
        assert models_cli.main(["status"]) == 2
        assert "required" in capsys.readouterr().out
        assert models_cli.main(["status", "--profile-home", str(tmp_path)]) == 2
        assert "no store" in capsys.readouterr().out
        (tmp_path / "nachos").mkdir()
        (tmp_path / "nachos" / "memory.db").write_text("")
        (tmp_path / "nachos" / "config.json").write_text(json.dumps({"embeddings": "off"}))
        assert models_cli.main(["reindex", "--profile-home", str(tmp_path)]) == 2
        assert "off or unavailable" in capsys.readouterr().out

    def test_pytest_collects_with_raises(self) -> None:
        with pytest.raises(SystemExit):
            models_cli.main(["bogus"])


class TestStatusText:
    def test_status_shows_preset_and_switch_progress(self, tmp_path: Path) -> None:
        count = reindex.BULK_ENTRIES + 8
        _profile_with_entries(tmp_path, count, "old-model")
        path = tmp_path / "nachos" / "memory.db"
        scope = Scope(PROFILE, get_store("sqlite", path), MemoryIndex(index_path_for(path)))
        service = MemoryService(
            OrderedDict([(PROFILE, scope)]), embedder=_Words("new-model"), reindex_budget_s=0.0
        )
        cfg = dict(DEFAULTS, embedding_preset="quality")
        service.search("entry number")  # one bounded step after the switch
        text = status_text(service, cfg, None, "s", 0)
        assert "preset      : quality" in text
        assert f"/{count} with the configured model; recall is lexical until done" in text
        service.reindex()
        done = status_text(service, cfg, None, "s", 0)
        assert "re-embedding" not in done
        service.close()
