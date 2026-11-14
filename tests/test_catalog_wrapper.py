"""Catalog-root registration contracts for Nachos."""

import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class Collector:
    def __init__(self):
        self.memory = []
        self.commands = []

    def register_memory_provider(self, provider):
        self.memory.append(provider)

    def register_command(self, name, *_args, **_kwargs):
        self.commands.append(name)


def test_catalog_root_registers_memory_only():
    """A catalog directory install exposes the memory provider."""
    module_path = ROOT / "__init__.py"
    spec = importlib.util.spec_from_file_location("nachos_catalog_root", module_path)

    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    collector = Collector()
    module.register(collector)

    assert [provider.name for provider in collector.memory] == ["nachos"]
    assert "nachos-memory-status" in set(collector.commands)


def test_catalog_root_imports_without_plugin_root_on_sys_path():
    """The directory loader can import the root package without a path shim."""
    script = f"""
import importlib.util
import pathlib
import sys

root = pathlib.Path(r'''{ROOT}''')
python_root = pathlib.Path(sys.base_prefix)
sys.path = [
    entry for entry in sys.path if pathlib.Path(entry).is_relative_to(python_root)
]
spec = importlib.util.spec_from_file_location(
    'nachos_catalog_root', root / '__init__.py', submodule_search_locations=[str(root)]
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
"""
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", script],
        cwd=ROOT.parent,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_dev_dependency_group_allows_hermes_pytest_major():
    """Catalog dependency preparation remains compatible with Hermes' pytest 9."""
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert '"pytest>=8,<10"' in pyproject
