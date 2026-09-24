"""Catalog-root registration contracts for Nachos."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class Collector:
    def __init__(self):
        self.memory = []
        self.context = []
        self.commands = []

    def register_memory_provider(self, provider):
        self.memory.append(provider)

    def register_context_engine(self, engine):
        self.context.append(engine)

    def register_command(self, name, *_args, **_kwargs):
        self.commands.append(name)


def test_catalog_root_registers_memory_and_context():
    """A catalog directory install exposes both supported Nachos layers."""
    module_path = ROOT / "__init__.py"
    spec = importlib.util.spec_from_file_location("nachos_catalog_root", module_path)

    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    collector = Collector()
    module.register(collector)

    assert [provider.name for provider in collector.memory] == ["nachos"]
    assert [engine.name for engine in collector.context] == ["nachos"]
    assert {"nachos-memory-status", "nachos-snapshot"} <= set(collector.commands)
