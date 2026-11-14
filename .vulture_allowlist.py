# Vulture allowlist: names that ARE used, but only through a host contract that static
# analysis cannot see. Vulture runs at min_confidence 80 (see [tool.vulture] in
# pyproject.toml), which already ignores the 60%-confidence "unused method/attribute"
# reports for MemoryProvider hooks, dataclass fields and sqlite row_factory, so only the
# 100%-confidence item below needs listing.

# MemoryProvider.on_session_switch(rewound=...): Hermes passes this keyword by name.
rewound
