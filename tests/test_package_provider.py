"""Tests for Nachos' versioned Hermes memory-provider package boundary."""

from nachos_hermes.memory_provider import NachosMemoryProvider


def test_system_prompt_always_declares_nachos_memory_contract():
    """The active backend remains explicit even before the store is initialized."""
    prompt = NachosMemoryProvider().system_prompt_block()

    assert "Nachos is the active, authoritative durable-memory provider" in prompt
    assert "nachos_memory_recall" in prompt
    assert "memory.memory_enabled: false" in prompt