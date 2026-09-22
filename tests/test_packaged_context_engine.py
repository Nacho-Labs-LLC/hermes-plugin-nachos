"""Public-package contracts for the dogfooded Nachos context engine."""

from nachos_hermes.context_engine import NachosContextEngine


def test_packaged_context_engine_derives_a_working_trigger_from_model_capacity():
    """A package install exposes the same pressure behavior used in dogfood."""
    engine = NachosContextEngine()

    engine.update_model("test-model", context_length=100_000)

    assert engine.name == "nachos"
    assert engine.threshold_tokens == 75_000
    assert engine.should_compress(75_000)
