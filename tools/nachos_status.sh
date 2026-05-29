#!/usr/bin/env bash
# nachos_status.sh — one-screen Nachos soak-monitoring summary
#
# Usage: ~/DEV/hermes-plugin-nachos/tools/nachos_status.sh
# Bookmark with: alias nachos-status='bash ~/DEV/hermes-plugin-nachos/tools/nachos_status.sh'

set -euo pipefail

NACHOS_DIR="${HERMES_HOME:-$HOME/.hermes}/nachos"
LOG_FILE="${HERMES_HOME:-$HOME/.hermes}/logs/agent.log"
PY="${HERMES_AGENT_PY:-$HOME/DEV/hermes-agent/.venv/bin/python}"

# Color helpers (no-op if not a tty)
if [ -t 1 ]; then
    BOLD='\033[1m'; DIM='\033[2m'; GRN='\033[32m'; YEL='\033[33m'
    RED='\033[31m'; CYA='\033[36m'; RST='\033[0m'
else
    BOLD=''; DIM=''; GRN=''; YEL=''; RED=''; CYA=''; RST=''
fi

hr() { printf "${DIM}─────────────────────────────────────────────────────────────────${RST}\n"; }
hdr() { printf "\n${BOLD}${CYA}▸ %s${RST}\n" "$1"; }

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

printf "${BOLD}🧀 Nachos Soak Status${RST}  ${DIM}$(date '+%Y-%m-%d %H:%M:%S')${RST}\n"
hr

# ---------------------------------------------------------------------------
# 1. Plugin activation
# ---------------------------------------------------------------------------

hdr "Plugin activation"
# Use Python to resolve the EFFECTIVE config (handles YAML duplicate-key,
# merge order, etc.) instead of grepping the raw file.
activation=$("$PY" - <<EOF 2>/dev/null
try:
    import sys
    sys.path.insert(0, "$HOME/DEV/hermes-agent")
    from hermes_cli.config import load_config
    cfg = load_config()
    eng = (cfg.get("context", {}) or {}).get("engine", "")
    prov = (cfg.get("memory", {}) or {}).get("provider", "")
    print(f"engine={eng!r}")
    print(f"provider={prov!r}")
except Exception as e:
    print(f"error={e!r}")
EOF
)
eng_line=$(printf "%s\n" "$activation" | sed -n 's/^engine=//p')
prov_line=$(printf "%s\n" "$activation" | sed -n 's/^provider=//p')
err_line=$(printf "%s\n" "$activation" | sed -n 's/^error=//p')

if [ -n "$err_line" ]; then
    printf "  ${YEL}⚠${RST}  could not resolve config: %s\n" "$err_line"
elif [ "$eng_line" = "'nachos'" ]; then
    printf "  ${GRN}✓${RST} context.engine: nachos\n"
else
    printf "  ${RED}✗${RST} context.engine: %s (expected 'nachos')\n" "$eng_line"
fi

if [ "$prov_line" = "'nachos'" ]; then
    printf "  ${GRN}✓${RST} memory.provider: nachos\n"
else
    printf "  ${RED}✗${RST} memory.provider: %s (expected 'nachos')\n" "$prov_line"
fi

# Check if memory plugin has been activated in the current log (not just config)
recently_activated=$(grep "Memory provider 'nachos' activated" "${LOG_FILE:-$HOME/.hermes/logs/agent.log}" 2>/dev/null | tail -1)
if [ -n "$recently_activated" ]; then
    activated_ts=$(echo "$recently_activated" | awk '{print $1, $2}')
    printf "  ${GRN}✓${RST} last activated: %s\n" "$activated_ts"
else
    printf "  ${YEL}⚠${RST}  memory provider not yet activated — restart Hermes to activate\n"
fi

# ---------------------------------------------------------------------------
# 2. Fact store
# ---------------------------------------------------------------------------

hdr "Fact store"
FACTS_FILE="$NACHOS_DIR/facts.jsonl"
if [ -f "$FACTS_FILE" ]; then
    total=$(wc -l < "$FACTS_FILE" | tr -d ' ')
    size=$(du -h "$FACTS_FILE" | awk '{print $1}')
    printf "  total facts:  ${BOLD}%s${RST}  ${DIM}(%s)${RST}\n" "$total" "$size"

    if [ "$total" -gt 0 ]; then
        # Breakdown by kind + confidence histogram
        "$PY" - <<EOF 2>/dev/null || true
import json, sys, collections
kinds = collections.Counter()
confs = []
sessions = set()
recent_objects = []
with open("$FACTS_FILE") as f:
    for line in f:
        try:
            r = json.loads(line)
        except Exception:
            continue
        kinds[r.get("kind", "unknown")] += 1
        confs.append(r.get("confidence", 0))
        if r.get("source_session"):
            sessions.add(r["source_session"])
        recent_objects.append((r.get("extracted_at", ""), r))

print(f"  by kind:      ", end="")
print(", ".join(f"{k}={v}" for k, v in kinds.most_common()))

if confs:
    avg = sum(confs) / len(confs)
    lo = min(confs); hi = max(confs)
    bins = {"0.6-0.7": 0, "0.7-0.8": 0, "0.8-0.9": 0, "0.9-1.0": 0}
    for c in confs:
        if c < 0.7: bins["0.6-0.7"] += 1
        elif c < 0.8: bins["0.7-0.8"] += 1
        elif c < 0.9: bins["0.8-0.9"] += 1
        else: bins["0.9-1.0"] += 1
    print(f"  confidence:   avg={avg:.2f} min={lo:.2f} max={hi:.2f}")
    print(f"  histogram:    " + " ".join(f"{k}={v}" for k, v in bins.items()))

print(f"  sessions:     {len(sessions)} contributed")

# Last 5 facts (most recent first by extracted_at)
recent_objects.sort(key=lambda t: t[0], reverse=True)
print()
print("  Most recent facts:")
for ts, f in recent_objects[:5]:
    sub = f.get("subject", "?")[:30]
    pred = f.get("predicate", "?")[:25]
    obj = f.get("object", "?")[:50]
    conf = f.get("confidence", 0)
    print(f"    [{conf:.2f}] {sub} {pred} {obj}")
EOF
    fi
else
    printf "  ${DIM}(no facts.jsonl yet — extraction has not run)${RST}\n"
fi

# ---------------------------------------------------------------------------
# 3. Snapshots
# ---------------------------------------------------------------------------

hdr "Snapshots"
SNAP_DIR="$NACHOS_DIR/snapshots"
if [ -d "$SNAP_DIR" ]; then
    total_snaps=$(find "$SNAP_DIR" -name "*.json.gz" 2>/dev/null | wc -l | tr -d ' ')
    total_size=$(du -sh "$SNAP_DIR" 2>/dev/null | awk '{print $1}')
    sessions=$(find "$SNAP_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d ' ')
    printf "  sessions:     %s\n" "$sessions"
    printf "  total snaps:  ${BOLD}%s${RST}  ${DIM}(%s)${RST}\n" "$total_snaps" "$total_size"

    # Per-session breakdown if more than one
    if [ "$sessions" -gt 0 ]; then
        echo
        echo "  Per session (newest first):"
        for d in $(find "$SNAP_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -10); do
            sid=$(basename "$d")
            count=$(find "$d" -name "*.json.gz" 2>/dev/null | wc -l | tr -d ' ')
            sz=$(du -sh "$d" 2>/dev/null | awk '{print $1}')
            printf "    %-40s %s snaps  ${DIM}(%s)${RST}\n" "$sid" "$count" "$sz"
        done
    fi
else
    printf "  ${DIM}(none — no compaction has hit aggressive zone yet)${RST}\n"
fi

# ---------------------------------------------------------------------------
# 4. PromptReports
# ---------------------------------------------------------------------------

hdr "PromptReports"
REPORTS_DIR="$NACHOS_DIR/reports"
if [ -d "$REPORTS_DIR" ]; then
    files=$(find "$REPORTS_DIR" -name "*.jsonl" 2>/dev/null)
    file_count=$(echo "$files" | grep -c . || true)
    if [ "$file_count" -gt 0 ]; then
        total_lines=$(cat $files 2>/dev/null | wc -l | tr -d ' ')
        total_size=$(du -sh "$REPORTS_DIR" 2>/dev/null | awk '{print $1}')
        printf "  sessions:     %s files\n" "$file_count"
        printf "  total turns:  ${BOLD}%s${RST}  ${DIM}(%s)${RST}\n" "$total_lines" "$total_size"

        # Most recent file + last reported turn
        newest=$(ls -t "$REPORTS_DIR"/*.jsonl 2>/dev/null | head -1)
        if [ -n "$newest" ]; then
            sid=$(basename "$newest" .jsonl)
            last_line=$(tail -1 "$newest" 2>/dev/null)
            if [ -n "$last_line" ]; then
                stats=$("$PY" -c "
import json
r = json.loads('''$last_line''')
print(f\"chars={r.get('total_chars','?')} tokens={r.get('total_tokens','?')} sections={len(r.get('sections',[]))}\")
" 2>/dev/null || echo "(parse failed)")
                printf "  most recent:  %s  ${DIM}%s${RST}\n" "$sid" "$stats"
            fi
        fi
    else
        printf "  ${DIM}(no reports yet)${RST}\n"
    fi
else
    printf "  ${DIM}(no reports dir — provider not yet initialized)${RST}\n"
fi

# ---------------------------------------------------------------------------
# 5. Recent log activity
# ---------------------------------------------------------------------------

hdr "Recent log activity"
if [ -f "$LOG_FILE" ]; then
    matches=$(grep -ic "nachos" "$LOG_FILE" 2>/dev/null || echo 0)
    printf "  total log mentions: ${BOLD}%s${RST}\n" "$matches"
    if [ "$matches" -gt 0 ]; then
        echo
        echo "  Last 8 Nachos log lines:"
        grep -i "nachos" "$LOG_FILE" 2>/dev/null | tail -8 | sed 's/^/    /'
    fi
else
    printf "  ${DIM}(agent.log not found — Hermes may not be running)${RST}\n"
fi

# ---------------------------------------------------------------------------
# 6. Health check
# ---------------------------------------------------------------------------

hdr "Health check"
warnings=0
errors=0

# Anything saying extraction failed?
if [ -f "$LOG_FILE" ]; then
    fail=$(grep -ic "Nachos extraction.*failed\|Nachos.*disabled" "$LOG_FILE" 2>/dev/null | tr -d ' \n' || echo 0)
    fail=${fail:-0}
    if [ "$fail" -gt 0 ] 2>/dev/null; then
        printf "  ${YEL}⚠${RST}  %s extraction-related warnings in log\n" "$fail"
        warnings=$((warnings + 1))
    fi
fi

# Has extraction ever succeeded?
if [ -f "$LOG_FILE" ]; then
    success=$(grep -c "Nachos extraction.*kept=" "$LOG_FILE" 2>/dev/null | tr -d ' \n' || echo 0)
    success=${success:-0}
    if [ "$success" -gt 0 ] 2>/dev/null; then
        printf "  ${GRN}✓${RST}  extraction has succeeded %s times\n" "$success"
    else
        printf "  ${YEL}⚠${RST}  extraction has never succeeded yet (start a session and /reset)\n"
        warnings=$((warnings + 1))
    fi
fi

# Snapshot dir size warning
if [ -d "$SNAP_DIR" ]; then
    bytes=$(du -sk "$SNAP_DIR" 2>/dev/null | awk '{print $1}')
    if [ "${bytes:-0}" -gt 102400 ]; then
        printf "  ${YEL}⚠${RST}  snapshots dir > 100MB — consider rotation tuning\n"
        warnings=$((warnings + 1))
    fi
fi

if [ "$warnings" -eq 0 ] && [ "$errors" -eq 0 ]; then
    printf "  ${GRN}all checks pass${RST}\n"
fi

echo
hr
echo
