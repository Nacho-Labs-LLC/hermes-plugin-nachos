"""Characterize capture's disclosed model inputs without any external model call."""

from nachos_core import capture
from nachos_hermes.memory_provider import NachosMemoryProvider


def test_capture_transmits_known_memory_and_user_facts_before_insertion_approval(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(NachosMemoryProvider, "_start_capture_thread", lambda _self: None)
    provider = NachosMemoryProvider()
    provider.save_config({"embeddings": "off", "capture": "review"}, str(tmp_path))
    provider.initialize("disclosure", hermes_home=str(tmp_path), agent_context="primary")
    provider._service.put(
        title="Release procedure",
        summary="Run release checks before publishing the package",
        category="procedure",
        body="Run the complete release checks before publishing the package to the catalog.",
    )
    user_file = tmp_path / "memories" / "USER.md"
    user_file.parent.mkdir()
    user_text = "\n§\n".join(f"Synthetic user fact {i:03d}" for i in range(101))
    user_file.write_text(user_text, encoding="utf-8")
    messages = [
        message
        for _ in range(4)
        for message in (
            {"role": "user", "content": "Synthetic conversation detail. " * 30},
            {"role": "assistant", "content": "Synthetic conversation response. " * 30},
        )
    ]
    calls = []

    def record_request(prompt):
        calls.append(prompt)
        return "[]"

    provider.on_session_end(messages)
    assert calls == []
    assert provider.process_capture(record_request)["jobs"] == 1
    assert len(calls) == 1
    outbound = calls[0][1]["content"]
    assert "Release procedure" in outbound
    assert "Run release checks before publishing the package" in outbound
    assert "Synthetic conversation detail" in outbound
    assert "Synthetic user fact 000" in outbound
    assert "Synthetic user fact 099" in outbound
    assert "Synthetic user fact 100" not in outbound
    assert user_file.read_text(encoding="utf-8") == user_text
    assert len(provider._service.profile.store.list()) == 1


def test_capture_prompt_caps_known_memory_at_200_entries():
    manifest = [
        (str(i), f"Known title {i:03d}", f"Known summary {i:03d}", "test") for i in range(201)
    ]
    outbound = capture.build_prompt([], manifest)[1]["content"]
    assert "Known title 199: Known summary 199" in outbound
    assert "Known title 200" not in outbound


def test_capture_off_does_not_call_the_model(tmp_path, monkeypatch):
    monkeypatch.setattr(NachosMemoryProvider, "_start_capture_thread", lambda _self: None)
    provider = NachosMemoryProvider()
    provider.save_config({"embeddings": "off", "capture": "off"}, str(tmp_path))
    provider.initialize("disabled", hermes_home=str(tmp_path), agent_context="primary")
    calls = []
    result = provider.process_capture(lambda prompt: calls.append(prompt) or "[]")
    assert result == {"jobs": 0, "candidates": 0, "retry": 0}
    assert calls == []
