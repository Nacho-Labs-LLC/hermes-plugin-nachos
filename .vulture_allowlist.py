# Vulture allowlist — names that ARE used, but only via host contract
# (Hermes calls these by name through the MemoryProvider ABC / plugin
# registration / dataclass serialization), so static analysis can't see
# the use. Referencing them here suppresses the false positives.
#
# Usage: vulture nachos_core plugins tools .vulture_allowlist.py --min-confidence 70

# MemoryProvider ABC overrides (called by the Hermes MemoryManager)
_p.is_available
_p.initialize
_p.system_prompt_block
_p.prefetch
_p.queue_prefetch
_p.get_tool_schemas
_p.handle_tool_call
_p.on_memory_write
_p.shutdown
metadata          # on_memory_write signature param (ABC contract)

# Plugin entry point (called by the plugin loader)
register

# Dataclass fields consumed via serialization / host reads
utilization_ratio
before_size
after_size
created_at
message_count
confidence
source_session
extracted_at
tags

# SnapshotStore methods (called by the context engine at runtime)
save
load
rotate

# sqlite3 connection tuning
row_factory
