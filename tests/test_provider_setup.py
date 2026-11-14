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
    assert restored._cfg["manifest_char_budget"] == 1800


def test_setup_schema_describes_all_non_secret_provider_settings():
    """The setup UI exposes every persisted Nachos setting without secrets."""
    schema = NachosMemoryProvider().get_config_schema()

    assert {field["key"] for field in schema} == {
        "store",
        "manifest_char_budget",
        "embeddings",
        "embedding_model",
        "embedding_dtype",
        "embedding_base_url",
        "embedding_api_key_env",
        "duplicate_threshold",
        "shared_store",
        "shared_store_path",
        "redact_secrets",
        "checkpoint_max_age_days",
        "capture",
    }
    assert all("default" in field for field in schema)
    assert all(not field.get("secret") for field in schema)
