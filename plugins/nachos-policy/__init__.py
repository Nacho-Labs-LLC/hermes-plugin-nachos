"""
nachos-policy — Hermes plugin
==============================
Python port of the Nachos Cheese policy engine.

Registers a pre_tool_call hook that evaluates every tool invocation against
YAML-based policy rules before execution.  Opt-in: only activates when
  nachos:
    layers:
      policy: true
is set in ~/.hermes/config.yaml.

Default effect is deny, but the bundled standard.yaml ships with an
explicit allow-* rule so enabling the plugin doesn't break anything.

Failure-open guarantee: any unhandled exception inside evaluate() is caught,
a warning is logged, and the tool call is allowed through.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger("nachos-policy")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_OPERATORS = frozenset(
    ["equals", "not_equals", "in", "not_in", "contains", "matches", "starts_with", "ends_with"]
)
VALID_EFFECTS = frozenset(["allow", "deny"])

WATCH_INTERVAL = 5  # seconds between mtime polls
REGEX_MAX_LEN = 200  # ReDoS guard — matches TS original


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PolicyCondition:
    field: str
    operator: str
    value: Any  # str | list[str] | int | bool


@dataclass
class PolicyRule:
    id: str
    priority: int
    match: dict  # keys: resource, action, resourceId (all optional)
    effect: str  # 'allow' | 'deny'
    description: str = ""
    conditions: list[PolicyCondition] = field(default_factory=list)
    reason: str = ""


@dataclass
class PolicyDocument:
    version: str
    rules: list[PolicyRule]
    metadata: dict = field(default_factory=dict)


@dataclass
class PolicyValidationError:
    file: str
    message: str
    rule_id: str = ""
    field: str = ""


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


def _validate_document(doc: Any, filename: str) -> list[PolicyValidationError]:
    errors: list[PolicyValidationError] = []

    if not isinstance(doc, dict):
        errors.append(PolicyValidationError(file=filename, message="Policy document must be a dict"))
        return errors

    if not doc.get("version"):
        errors.append(PolicyValidationError(file=filename, message="Missing required field: version", field="version"))

    rules = doc.get("rules")
    if rules is None or not isinstance(rules, list):
        errors.append(PolicyValidationError(file=filename, message="Missing or invalid rules array", field="rules"))
        return errors  # can't continue

    seen_ids: set[str] = set()
    for i, rule in enumerate(rules):
        rule_errors = _validate_rule(rule, filename, i)
        errors.extend(rule_errors)

        if isinstance(rule, dict):
            rid = rule.get("id")
            if isinstance(rid, str):
                if rid in seen_ids:
                    errors.append(
                        PolicyValidationError(file=filename, rule_id=rid, message=f"Duplicate rule ID: {rid}", field="id")
                    )
                seen_ids.add(rid)

    return errors


def _validate_rule(rule: Any, filename: str, index: int) -> list[PolicyValidationError]:
    errors: list[PolicyValidationError] = []
    if not isinstance(rule, dict):
        errors.append(PolicyValidationError(file=filename, message=f"Rule at index {index} must be a dict"))
        return errors

    rule_id = rule.get("id") if isinstance(rule.get("id"), str) else f"rule-{index}"

    if not isinstance(rule.get("id"), str) or not rule.get("id"):
        errors.append(PolicyValidationError(file=filename, rule_id=rule_id, message="Rule must have a string id", field="id"))

    priority = rule.get("priority")
    if not isinstance(priority, (int, float)):
        errors.append(PolicyValidationError(file=filename, rule_id=rule_id, message="Rule must have a numeric priority", field="priority"))
    elif not (isinstance(priority, float) and priority != priority) and priority >= 0:  # finite non-negative
        pass  # ok
    else:
        errors.append(
            PolicyValidationError(file=filename, rule_id=rule_id, message=f"Priority must be finite non-negative, got: {priority}", field="priority")
        )

    match = rule.get("match")
    if not isinstance(match, dict):
        errors.append(PolicyValidationError(file=filename, rule_id=rule_id, message="Rule must have a match object", field="match"))
    else:
        errors.extend(_validate_match(match, filename, rule_id))

    effect = rule.get("effect")
    if effect not in VALID_EFFECTS:
        errors.append(
            PolicyValidationError(
                file=filename,
                rule_id=rule_id,
                message=f"Rule must have a valid effect ({', '.join(sorted(VALID_EFFECTS))})",
                field="effect",
            )
        )

    conditions = rule.get("conditions")
    if conditions is not None:
        if not isinstance(conditions, list):
            errors.append(PolicyValidationError(file=filename, rule_id=rule_id, message="Conditions must be a list", field="conditions"))
        else:
            for ci, cond in enumerate(conditions):
                errors.extend(_validate_condition(cond, filename, rule_id, ci))

    return errors


def _validate_match(match: dict, filename: str, rule_id: str) -> list[PolicyValidationError]:
    errors: list[PolicyValidationError] = []
    # resource / action / resourceId are all optional; if present must be str or list[str]
    for key in ("resource", "action", "resourceId"):
        val = match.get(key)
        if val is None:
            continue
        items = val if isinstance(val, list) else [val]
        for item in items:
            if not isinstance(item, str):
                errors.append(
                    PolicyValidationError(
                        file=filename, rule_id=rule_id, message=f"match.{key} values must be strings", field=f"match.{key}"
                    )
                )
    return errors


def _validate_condition(cond: Any, filename: str, rule_id: str, index: int) -> list[PolicyValidationError]:
    errors: list[PolicyValidationError] = []
    if not isinstance(cond, dict):
        errors.append(PolicyValidationError(file=filename, rule_id=rule_id, message=f"Condition {index} must be a dict"))
        return errors

    if not isinstance(cond.get("field"), str) or not cond.get("field"):
        errors.append(
            PolicyValidationError(
                file=filename, rule_id=rule_id, message=f"Condition {index} must have a string field", field=f"conditions[{index}].field"
            )
        )

    op = cond.get("operator")
    if op not in VALID_OPERATORS:
        errors.append(
            PolicyValidationError(
                file=filename,
                rule_id=rule_id,
                message=f"Condition {index} must have a valid operator ({', '.join(sorted(VALID_OPERATORS))})",
                field=f"conditions[{index}].operator",
            )
        )

    if cond.get("value") is None and "value" not in cond:
        errors.append(
            PolicyValidationError(
                file=filename, rule_id=rule_id, message=f"Condition {index} must have a value", field=f"conditions[{index}].value"
            )
        )

    return errors


# ---------------------------------------------------------------------------
# Document parser — raw YAML dict -> typed dataclasses
# ---------------------------------------------------------------------------


def _parse_document(raw: dict, filename: str) -> PolicyDocument:
    """Convert validated raw YAML dict to a PolicyDocument."""
    rules: list[PolicyRule] = []
    for r in raw.get("rules", []):
        conditions: list[PolicyCondition] = []
        for c in r.get("conditions") or []:
            conditions.append(PolicyCondition(field=c["field"], operator=c["operator"], value=c["value"]))
        rules.append(
            PolicyRule(
                id=r["id"],
                description=r.get("description", ""),
                priority=int(r["priority"]),
                match=dict(r.get("match") or {}),
                conditions=conditions,
                effect=r["effect"],
                reason=r.get("reason", ""),
            )
        )
    return PolicyDocument(
        version=str(raw.get("version", "1.0")),
        metadata=dict(raw.get("metadata") or {}),
        rules=rules,
    )


# ---------------------------------------------------------------------------
# Policy Engine
# ---------------------------------------------------------------------------


class PolicyEngine:
    """
    Synchronous, thread-safe policy engine.

    evaluate() is the hot path — no I/O, no locks during the read.
    The rules list is replaced atomically via assignment (CPython GIL
    guarantees list reference assignment is atomic).
    """

    def __init__(
        self,
        policies_dir: str | os.PathLike,
        default_effect: str = "deny",
        enable_hot_reload: bool = True,
    ) -> None:
        self._policies_dir = Path(policies_dir)
        self._default_effect = default_effect
        self._enable_hot_reload = enable_hot_reload

        # Sorted rules — replaced atomically on reload
        self._rules: list[PolicyRule] = []
        # Snapshot of mtimes used to detect changes
        self._mtimes: dict[str, float] = {}
        self._last_reload: float | None = None
        self._validation_errors: list[PolicyValidationError] = []

        # Stats
        self._eval_count = 0
        self._eval_total_ms = 0.0
        self._stats_lock = threading.Lock()

        # Initial load
        self.load()

        # Hot-reload watcher
        if enable_hot_reload:
            t = threading.Thread(target=self._watch_thread, daemon=True, name="nachos-policy-watcher")
            t.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> None:
        """
        Atomic load: read all *.yaml/*.yml from policies_dir, validate every
        document.  If ANY document has a validation error the entire batch is
        rejected and the previous ruleset is kept intact.
        """
        policies_dir = self._policies_dir

        if not policies_dir.exists():
            log.warning("nachos-policy: policies dir does not exist: %s — using default %s", policies_dir, self._default_effect)
            return

        yaml_files = sorted(f for f in policies_dir.iterdir() if f.suffix in (".yaml", ".yml"))

        if not yaml_files:
            log.warning("nachos-policy: no policy files found in %s", policies_dir)
            return

        candidate_docs: list[PolicyDocument] = []
        all_errors: list[PolicyValidationError] = []
        new_mtimes: dict[str, float] = {}

        for fpath in yaml_files:
            try:
                raw_text = fpath.read_text(encoding="utf-8")
                raw = yaml.safe_load(raw_text)
            except Exception as exc:
                all_errors.append(
                    PolicyValidationError(file=fpath.name, message=f"Failed to load: {exc}")
                )
                continue

            errors = _validate_document(raw, fpath.name)
            if errors:
                all_errors.extend(errors)
                continue

            candidate_docs.append(_parse_document(raw, fpath.name))
            new_mtimes[str(fpath)] = fpath.stat().st_mtime

        if all_errors:
            for e in all_errors:
                log.error("nachos-policy: validation error [%s] rule=%s  %s", e.file, e.rule_id or "-", e.message)
            if self._last_reload is not None:
                log.warning("nachos-policy: rejecting reload — keeping previous ruleset intact")
            else:
                log.warning("nachos-policy: initial load has errors — using default %s", self._default_effect)
            self._validation_errors = all_errors
            return  # keep previous _rules

        # Build sorted rule list from all docs (highest priority first)
        all_rules: list[PolicyRule] = []
        seen_ids: set[str] = set()
        for doc in candidate_docs:
            for rule in doc.rules:
                if rule.priority < 0:
                    log.warning("nachos-policy: rule %s has negative priority — skipped", rule.id)
                    continue
                if rule.id in seen_ids:
                    log.warning("nachos-policy: duplicate rule ID %s — keeping first occurrence", rule.id)
                    continue
                seen_ids.add(rule.id)
                all_rules.append(rule)

        all_rules.sort(key=lambda r: r.priority, reverse=True)

        # Atomic replace
        self._rules = all_rules
        self._mtimes = new_mtimes
        self._validation_errors = []
        self._last_reload = time.monotonic()

        log.info(
            "nachos-policy: loaded %d rule(s) from %d file(s)",
            len(all_rules),
            len(candidate_docs),
        )

    def evaluate(
        self,
        tool_name: str,
        tool_args: dict,
        context: dict | None = None,
    ) -> tuple[bool, str]:
        """
        Evaluate whether tool_name with tool_args is allowed.

        Returns (allowed: bool, reason: str).

        Failure-open: any unexpected exception logs a warning and returns
        (True, "policy evaluation error — fail open").
        """
        t0 = time.perf_counter()
        try:
            result = self._evaluate_inner(tool_name, tool_args, context or {})
        except Exception as exc:  # pylint: disable=broad-except
            log.warning("nachos-policy: evaluation error (fail open): %s", exc, exc_info=True)
            return (True, "policy evaluation error — fail open")
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            with self._stats_lock:
                self._eval_count += 1
                self._eval_total_ms += elapsed_ms

        return result

    def get_stats(self) -> dict:
        with self._stats_lock:
            avg = self._eval_total_ms / self._eval_count if self._eval_count else 0.0
            return {
                "rules_active": len(self._rules),
                "evaluations_total": self._eval_count,
                "avg_evaluation_ms": round(avg, 4),
                "last_reload": self._last_reload,
                "validation_errors": len(self._validation_errors),
            }

    # ------------------------------------------------------------------
    # Internal evaluation
    # ------------------------------------------------------------------

    def _evaluate_inner(
        self,
        tool_name: str,
        tool_args: dict,
        context: dict,
    ) -> tuple[bool, str]:
        """
        Core evaluation: iterate rules (highest priority first), return on
        first match.  Builds a lightweight request dict analogous to
        SecurityRequest.
        """
        # Build request context that conditions can inspect
        request = {
            "tool_name": tool_name,
            "tool_args": tool_args,
            **context,
        }

        rules = self._rules  # local ref — safe under GIL

        for rule in rules:
            if self._matches_rule(rule, tool_name, tool_args, request):
                allowed = rule.effect == "allow"
                reason = rule.reason if not allowed else f"allowed by rule {rule.id}"
                log.debug(
                    "nachos-policy: tool=%s rule=%s priority=%d effect=%s",
                    tool_name, rule.id, rule.priority, rule.effect,
                )
                return (allowed, reason)

        # No match — default effect
        if self._default_effect == "allow":
            return (True, "default allow")
        return (False, "No policy rule matched — default deny applied")

    def _matches_rule(
        self,
        rule: PolicyRule,
        tool_name: str,
        tool_args: dict,
        request: dict,
    ) -> bool:
        if not self._matches_criteria(rule.match, tool_name):
            return False
        for cond in rule.conditions:
            if not self._matches_condition(cond, tool_name, tool_args, request):
                return False
        return True

    def _matches_criteria(self, match: dict, tool_name: str) -> bool:
        """
        Match criteria maps directly onto tool calls:
          resource  — 'tool' matches all tool calls; others never match
          action    — 'execute' or 'call' match all tool calls; others never match
          resourceId — must equal tool_name (exact or list membership)
        """
        # resource
        resource = match.get("resource")
        if resource is not None:
            resources = resource if isinstance(resource, list) else [resource]
            # In the Hermes context every hook invocation IS a tool call.
            # Map 'tool' -> always matches.  Other resource types never match.
            if not any(r == "tool" for r in resources):
                return False

        # action
        action = match.get("action")
        if action is not None:
            actions = action if isinstance(action, list) else [action]
            # 'execute' and 'call' are the natural action for tool invocations
            tool_actions = {"execute", "call"}
            if not any(a in tool_actions for a in actions):
                return False

        # resourceId — exact tool name match
        resource_id = match.get("resourceId")
        if resource_id is not None:
            ids = resource_id if isinstance(resource_id, list) else [resource_id]
            if tool_name not in ids:
                return False

        return True

    def _matches_condition(
        self,
        cond: PolicyCondition,
        tool_name: str,
        tool_args: dict,
        request: dict,
    ) -> bool:
        actual = self._get_field_value(cond.field, tool_name, tool_args, request)
        expected = cond.value
        op = cond.operator

        if op == "equals":
            return actual == expected

        if op == "not_equals":
            return actual != expected

        if op == "in":
            if not isinstance(expected, list):
                return False
            return str(actual) in [str(v) for v in expected]

        if op == "not_in":
            if not isinstance(expected, list):
                return False
            return str(actual) not in [str(v) for v in expected]

        if op == "contains":
            if not isinstance(actual, str) or not isinstance(expected, str):
                return False
            return expected in actual

        if op == "matches":
            if not isinstance(actual, str) or not isinstance(expected, str):
                return False
            # ReDoS guard
            if len(expected) > REGEX_MAX_LEN:
                log.warning("nachos-policy: regex pattern exceeds %d chars — rejected", REGEX_MAX_LEN)
                return False
            # Nested quantifier guard: (a+)+, (a*)*, etc.
            if re.search(r"([+*}])\s*\)[\s]*[+*{]", expected):
                log.warning("nachos-policy: regex contains nested quantifiers — rejected: %.80s", expected)
                return False
            try:
                return bool(re.search(expected, actual))
            except re.error:
                return False

        if op == "starts_with":
            if not isinstance(actual, str) or not isinstance(expected, str):
                return False
            return actual.startswith(expected)

        if op == "ends_with":
            if not isinstance(actual, str) or not isinstance(expected, str):
                return False
            return actual.endswith(expected)

        return False

    def _get_field_value(
        self,
        field_path: str,
        tool_name: str,
        tool_args: dict,
        request: dict,
    ) -> Any:
        """
        Resolve a field path against the tool call context.

        Supported short-hands:
          tool_name          -> the tool being called
          tool_args.<key>    -> a top-level key in tool_args
          metadata.<key>     -> alias for tool_args.<key>
          context.<key>      -> extra context passed by caller

        Fallback: dot-notation traversal of the merged request dict.
        """
        if field_path == "tool_name":
            return tool_name

        if field_path.startswith("tool_args."):
            key = field_path[len("tool_args."):]
            return tool_args.get(key)

        if field_path.startswith("metadata."):
            key = field_path[len("metadata."):]
            # Check tool_args first (natural mapping), then context
            return tool_args.get(key, request.get(key))

        if field_path.startswith("context."):
            key = field_path[len("context."):]
            return request.get(key)

        # Generic dot-notation traversal over the merged request dict
        parts = field_path.split(".")
        value: Any = request
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return None
        return value

    # ------------------------------------------------------------------
    # Hot-reload watcher
    # ------------------------------------------------------------------

    def _watch_thread(self) -> None:
        """Daemon thread: poll file mtimes every WATCH_INTERVAL seconds."""
        log.debug("nachos-policy: watcher started (poll every %ds)", WATCH_INTERVAL)
        while True:
            time.sleep(WATCH_INTERVAL)
            try:
                self._check_and_reload()
            except Exception as exc:  # pylint: disable=broad-except
                log.warning("nachos-policy: watcher error: %s", exc)

    def _check_and_reload(self) -> None:
        if not self._policies_dir.exists():
            return
        yaml_files = [f for f in self._policies_dir.iterdir() if f.suffix in (".yaml", ".yml")]
        current_mtimes = {str(f): f.stat().st_mtime for f in yaml_files}
        if current_mtimes != self._mtimes:
            log.info("nachos-policy: policy file change detected — reloading")
            self.load()


# ---------------------------------------------------------------------------
# Hermes plugin register()
# ---------------------------------------------------------------------------


def register(ctx: Any) -> None:
    """
    Hermes plugin entry point.

    Reads ~/.hermes/config.yaml.  Only activates if:
      nachos:
        layers:
          policy: true
    """
    config_path = Path.home() / ".hermes" / "config.yaml"
    nachos_config: dict = {}
    if config_path.exists():
        try:
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            nachos_config = raw.get("nachos") or {}
        except Exception as exc:
            log.warning("nachos-policy: could not read config: %s", exc)

    layers = nachos_config.get("layers") or {}
    if not layers.get("policy", False):
        log.debug("nachos-policy: layer disabled (nachos.layers.policy is not true) — skipping registration")
        return

    policy_cfg = nachos_config.get("policy") or {}
    default_effect = policy_cfg.get("default_effect", "deny")
    enable_hot_reload = policy_cfg.get("hot_reload", True)
    policies_dir_str = policy_cfg.get("policies_dir") or str(Path.home() / ".hermes" / "nachos" / "policies")
    policies_dir = Path(policies_dir_str).expanduser()

    log.info("nachos-policy: activating — policies_dir=%s  default_effect=%s", policies_dir, default_effect)

    engine = PolicyEngine(
        policies_dir=policies_dir,
        default_effect=default_effect,
        enable_hot_reload=enable_hot_reload,
    )

    # ------------------------------------------------------------------
    # pre_tool_call hook
    # ------------------------------------------------------------------

    def pre_tool_call(tool_name: str, args: dict = None, tool_args: dict = None, **kwargs: Any):
        """Block tool execution if policy denies it.

        Returns {'action': 'block', 'message': reason} to block, None to allow.
        Failure-open: any exception logs a warning and returns None.
        """
        try:
            effective_args = args if args is not None else (tool_args or {})
            context = kwargs.get("context") or {}
            allowed, reason = engine.evaluate(tool_name, effective_args, context)
            if not allowed:
                return {
                    "action": "block",
                    "message": f"[nachos-policy] Tool '{tool_name}' blocked by policy: {reason}",
                }
        except Exception as exc:  # pylint: disable=broad-except
            log.warning("nachos-policy: pre_tool_call hook error (fail open): %s", exc)
        return None

    ctx.register_hook("pre_tool_call", pre_tool_call)
    log.info("nachos-policy: pre_tool_call hook registered")

    # ------------------------------------------------------------------
    # nachos_policy_check tool — dry-run evaluation
    # ------------------------------------------------------------------

    def nachos_policy_check(tool_name: str, tool_args: dict | None = None, context: dict | None = None) -> dict:
        """
        Dry-run policy evaluation.  Returns a dict with:
          allowed (bool), reason (str), stats (dict)

        Example usage from Hermes:
          nachos_policy_check(tool_name='terminal', tool_args={'command': 'rm -rf /'})
        """
        allowed, reason = engine.evaluate(tool_name, tool_args or {}, context or {})
        return {
            "allowed": allowed,
            "reason": reason,
            "stats": engine.get_stats(),
        }

    # Register as a plugin tool if the plugin context supports it
    if hasattr(ctx, "register_tool"):
        ctx.register_tool(
            name="nachos_policy_check",
            func=nachos_policy_check,
            description=(
                "Dry-run nachos policy check. Pass tool_name and optional tool_args/context "
                "to see whether the policy engine would allow or deny the call."
            ),
        )
        log.info("nachos-policy: nachos_policy_check tool registered")
    else:
        log.debug("nachos-policy: ctx has no register_tool — nachos_policy_check not exposed as tool")
