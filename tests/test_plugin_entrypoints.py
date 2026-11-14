"""Distribution metadata contracts for Nachos' plugin entry points."""

import importlib.metadata


def test_memory_entry_point_loads_a_callable():
    entry_point = next(
        item
        for item in importlib.metadata.entry_points(group="hermes_agent.memory_providers")
        if item.name == "nachos"
    )

    assert callable(entry_point.load())
