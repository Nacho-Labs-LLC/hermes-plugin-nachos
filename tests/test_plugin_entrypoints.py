"""Distribution metadata contracts for Nachos' plugin entry points."""

import importlib.metadata


def test_context_entry_point_loads_a_module_with_register():
    """Hermes' generic plugin loader imports a module then resolves register()."""
    entry_point = next(
        item
        for item in importlib.metadata.entry_points(group="hermes_agent.plugins")
        if item.name == "nachos-context"
    )

    module = entry_point.load()

    assert callable(getattr(module, "register", None))
