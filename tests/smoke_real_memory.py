"""Smoke test — build a manifest from the real ~/.hermes/ directory.

Run with:
    ~/DEV/hermes-agent/.venv/bin/python tests/smoke_real_memory.py

Useful for visually verifying what the manifest will look like in
production before wiring the plugin into Hermes.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.hermes_memory import HermesMemoryReader
from nachos_core.assembler import AssembleParams, PromptAssembler
from nachos_core.manifest import build_manifest, render_manifest


def main():
    home = Path.home() / ".hermes"
    print(f"Reading memory from: {home}\n")

    reader = HermesMemoryReader(hermes_home=home)

    # 1. Show raw entry counts
    entries = reader.list_entries(limit=1000)
    print(f"Total entries discovered: {len(entries)}")
    by_kind = {}
    for e in entries:
        by_kind[e["kind"]] = by_kind.get(e["kind"], 0) + 1
    for kind, count in sorted(by_kind.items(), key=lambda kv: -kv[1]):
        print(f"  {kind}: {count}")
    print()

    # 2. Show recent topics
    topics = reader.list_recent_topics(limit=5)
    print(f"Recent topics found: {len(topics)}")
    for t in topics:
        print(f"  • {t['topic']} ({t['session_age']})")
    print()

    # 3. Build and render the manifest
    manifest = build_manifest(reader)
    rendered = render_manifest(manifest)
    print("=" * 70)
    print("RENDERED MANIFEST (this would be injected into the system prompt)")
    print("=" * 70)
    print(rendered if rendered else "(empty)")
    print()

    # 4. Run it through the full assembler with a mock base prompt
    a = PromptAssembler()
    params = AssembleParams(
        base_prompt="You are Hermes Agent.",
        memory_manifest=rendered,
        include_memory_instructions=True,
    )
    full_prompt, report = a.assemble(params)

    print("=" * 70)
    print("PROMPT REPORT")
    print("=" * 70)
    print(f"total_chars:  {report.total_chars}")
    print(f"total_tokens: {report.total_tokens}")
    print(f"sections:")
    for s in report.sections:
        print(f"  • {s.name:25s} chars={s.size_chars:6d}  "
              f"tokens={s.size_tokens or 0:5d}  "
              f"hash={s.hash[:8]}...  source={s.source}")
    print()


if __name__ == "__main__":
    main()
