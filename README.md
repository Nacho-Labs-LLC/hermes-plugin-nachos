# Nachos for Hermes Agent

Nachos is a durable-memory provider for [Hermes Agent](https://hermes-agent.nousresearch.com). It replaces full-file memory injection with a bounded, three-tier assembly:

1. **Manifest** — a compact table of contents is present every turn.
2. **Prefetch** — relevant entry bodies are ranked and injected within a bounded budget.
3. **Recall** — the agent retrieves full entries only when needed.

The package is local-first: SQLite and flat-file stores use the standard library, lexical ranking is the default, and no LLM call occurs in the hot path.

## Install

Install Nachos into the same Python environment that runs Hermes:

```bash
python -m pip install "git+https://github.com/Nacho-Labs-LLC/hermes-plugin-nachos.git@v0.5.1"
```

For a reproducible production deployment, pin a full 40-character commit SHA instead of a tag.

Select Nachos for the active Hermes profile:

```bash
hermes config set memory.provider nachos
hermes config set memory.memory_enabled false
```

Start a new session or restart the profile gateway after switching providers. Nachos injects an explicit durable-memory contract that directs the agent to `nachos_memory_recall`, `nachos_memory_put`, and `nachos_memory_remove`.

## Configuration

Run `hermes memory setup` to configure the packaged provider. Settings are stored at `$HERMES_HOME/nachos/config.json`, so each Hermes profile remains isolated.

| Setting | Default | Meaning |
| --- | --- | --- |
| `store` | `sqlite` | Local durable store: `sqlite` or hand-editable `flatfile`. |
| `scorer` | `lexical` | Ranking method: `lexical` or optional `semantic`. |
| `semantic_provider` | `nachos` | Optional semantic backend: `nachos`, `sentence-transformers`, or `openai`. |
| `prefetch_top_n` | `5` | Maximum entries preloaded for a turn. |
| `prefetch_char_budget` | `1500` | Maximum characters injected by prefetch. |
| `manifest_char_budget` | `1200` | Target budget for the always-on manifest. |

Existing `nachos.memory` values in `config.yaml` remain supported for backwards compatibility. Profile-scoped setup values take precedence.

## Tools

- `nachos_memory_recall` — fetch an entry by key or search matching entries.
- `nachos_memory_put` — add or update an entry.
- `nachos_memory_remove` — delete an entry.
- `/nachos-memory-status` — display provider status.
- `/nachos-memory-list` — render the current manifest.

## Development

```bash
uv sync --group dev
.venv/Scripts/python.exe -m pytest tests -q  # Windows
.venv/Scripts/python.exe -m ruff check .
.venv/Scripts/python.exe -m build
```

The release gate builds a wheel, installs it into a clean environment, loads the `hermes_agent.memory_providers` entry point, and exercises provider registration.

## Nachos Context Engine

Nachos Context is a supported, dogfooded context engine that applies zone-based compaction, preserves tool-call/result pairs, and captures conversation snapshots before aggressive compaction. The same package installs its `nachos-context` Hermes plugin entry point.

Enable it for the active profile, then select it:

```bash
hermes plugins enable nachos-context
hermes config set context.engine nachos
```

Start a new session after enabling it. Context settings remain under the existing `nachos.compaction` and `nachos.snapshots` configuration sections.

## Experimental policy layer

The YAML policy layer is still experimental. It is not yet a supported installation artifact because it needs profile-safe configuration and packaging before it can be recommended to others.

## License

MIT
