"""Setup and profile-isolation contracts for the packaged Nachos provider."""

from pathlib import Path

from nachos_hermes.memory_provider import NachosMemoryProvider


def test_setup_config_is_profile_scoped_and_reloaded(tmp_path: Path):
    """Setup writes only to the target profile and initialize consumes it."""
    profile_home = tmp_path / "profile"
    provider = NachosMemoryProvider()

    provider.save_config(
        {
            "store": "flatfile",
            "scorer": "lexical",
            "prefetch_top_n": 7,
            "prefetch_char_budget": 2400,
            "manifest_char_budget": 1800,
        },
        str(profile_home),
    )

    config_file = profile_home / "nachos" / "config.json"
    assert config_file.is_file()
    assert not (tmp_path / ".hermes" / "nachos" / "config.json").exists()

    restored = NachosMemoryProvider()
    restored.initialize("session-1", hermes_home=str(profile_home))

    assert restored._cfg["store"] == "flatfile"
    assert restored._cfg["prefetch_top_n"] == 7
    assert restored._cfg["prefetch_char_budget"] == 2400
    assert restored._cfg["manifest_char_budget"] == 1800


def test_setup_schema_describes_all_non_secret_provider_settings():
    """The setup UI exposes every persisted Nachos setting without secrets."""
    schema = NachosMemoryProvider().get_config_schema()

    assert {field["key"] for field in schema} == {
        "store",
        "scorer",
        "semantic_provider",
        "prefetch_top_n",
        "prefetch_char_budget",
        "manifest_char_budget",
    }
    assert all(not field.get("secret") for field in schema)
