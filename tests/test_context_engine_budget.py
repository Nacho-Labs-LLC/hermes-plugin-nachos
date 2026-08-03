"""Contract tests for the Hermes-to-Nachos compression budget handoff."""

import importlib.util
import sys
import types
from pathlib import Path

ENGINE_PATH = (
    Path(__file__).resolve().parents[1]
    / "plugins"
    / "context_engine"
    / "nachos"
    / "__init__.py"
)


def _load_engine_module(monkeypatch):
    """Load the plugin with a minimal Hermes ContextEngine contract."""
    agent_module = types.ModuleType("agent")
    context_engine_module = types.ModuleType("agent.context_engine")

    class ContextEngine:
        pass

    context_engine_module.ContextEngine = ContextEngine
    agent_module.context_engine = context_engine_module
    monkeypatch.setitem(sys.modules, "agent", agent_module)
    monkeypatch.setitem(sys.modules, "agent.context_engine", context_engine_module)

    spec = importlib.util.spec_from_file_location("nachos_context_engine_test", ENGINE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_valid_host_budget_explicitly_accepts_the_contract(monkeypatch):
    module = _load_engine_module(monkeypatch)
    engine = module.NachosContextEngine()
    engine.update_model("test-main", 1_000_000)

    accepted = engine.set_compression_budget(
        800_000, 400_000, reason="model_init"
    )

    assert accepted is True
    assert engine._effective_context_length() == 800_000
    assert engine.threshold_tokens == 400_000


def test_invalid_host_budget_is_explicitly_rejected(monkeypatch):
    module = _load_engine_module(monkeypatch)
    engine = module.NachosContextEngine()
    engine.update_model("test-main", 1_000_000)

    accepted = engine.set_compression_budget(0, 400_000, reason="bad-input")

    assert accepted is False
    assert engine._effective_context_length() == 1_000_000
    assert engine.threshold_tokens == 0
