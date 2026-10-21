"""Real Hermes directory discovery; CI supplies a pinned HERMES_SOURCE checkout."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("kind", ["memory", "context"])
def test_catalog_directory_is_discovered_and_loaded(tmp_path, kind):
    source = os.environ.get("HERMES_SOURCE")
    if not source:
        pytest.skip("set HERMES_SOURCE to run the real-Hermes integration gate")
    source = Path(source).resolve()
    assert (source / "plugins" / "memory" / "__init__.py").is_file()

    home = tmp_path / "home"
    installed = home / "plugins" / "nachos"
    installed.mkdir(parents=True)
    for name in ("__init__.py", "plugin.yaml"):
        shutil.copy2(ROOT / name, installed / name)
    for name in ("nachos_core", "nachos_hermes"):
        shutil.copytree(ROOT / name, installed / name, ignore=shutil.ignore_patterns("__pycache__"))
    (home / "config.yaml").write_text("plugins:\n  isolation: in_process\n", encoding="utf-8")

    script = r'''
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
home, kind = Path(sys.argv[2]), sys.argv[3]
installed = home / "plugins" / "nachos"
# Exclude bundled providers and pip entry points so neither can mask a broken
# catalog directory. Discovery heuristics and directory loaders remain untouched.
if kind == "memory":
    from plugins import memory as loader
    from agent.memory_provider import MemoryProvider as Base
    loader._MEMORY_PLUGINS_DIR = home / "empty-bundled-memory"
    loader._iter_entry_points = lambda: []
    assert "nachos" in loader.list_memory_provider_names(), "catalog memory provider not discovered"
    assert loader.find_provider_dir("nachos") == installed
    discovered = loader.discover_memory_providers()
    loaded = loader.load_memory_provider("nachos", register_skills=False)
else:
    from plugins import context_engine as loader
    from agent.context_engine import ContextEngine as Base
    loader._CONTEXT_ENGINE_PLUGINS_DIR = home / "empty-bundled-context"
    assert loader.find_engine_dir("nachos") == installed, "catalog context engine not discovered"
    discovered = loader.discover_context_engines()
    loaded = loader.load_context_engine("nachos")
assert any(name == "nachos" and available for name, _, available in discovered), discovered
assert isinstance(loaded, Base), (kind, loaded)
assert loaded.name == "nachos"
assert type(loaded).__module__.startswith("nachos_hermes."), type(loaded)
import nachos_hermes
assert Path(nachos_hermes.__file__).parent == installed / "nachos_hermes"
print(kind + ": catalog directory discovered and loaded")
'''
    env = {key: value for key, value in os.environ.items() if not key.startswith("HERMES_")}
    env.update(HERMES_HOME=str(home), HOME=str(home), HERMES_ENABLE_PROJECT_PLUGINS="0")
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(source), str(home), kind],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
