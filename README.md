nachos-policy
=============

Hermes plugin that gates every tool call through a YAML-based policy engine
ported from the Nachos Cheese security layer.  Default effect is deny, but
the bundled standard.yaml ships with an explicit allow-all rule so enabling
the plugin does not break your workflow.


Enabling the plugin
-------------------

1. Copy (or symlink) this directory into your Hermes plugin path:

     ~/.hermes/plugins/nachos-policy/

   or add the parent directory to your plugin search path in config.yaml.

2. Add the opt-in flag to ~/.hermes/config.yaml:

     nachos:
       layers:
         policy: true

   Without this flag the plugin loads but registers nothing — zero overhead.

3. Optional config knobs (all under nachos.policy):

     nachos:
       layers:
         policy: true
       policy:
         policies_dir: ~/.hermes/nachos/policies   # default
         default_effect: deny                       # default
         hot_reload: true                           # default

4. On first run the engine loads every *.yaml / *.yml file from policies_dir.
   The bundled standard.yaml is copied there automatically only if you do it
   manually — the plugin does not auto-copy files to avoid surprises.
   You should copy policies/standard.yaml yourself:

     mkdir -p ~/.hermes/nachos/policies
     cp plugins/nachos-policy/policies/standard.yaml ~/.hermes/nachos/policies/


How rules work
--------------

A policy document is a YAML file with a version and a rules list.  Rules are
evaluated in descending priority order — the first matching rule wins.

Rule schema:

  - id: 'unique-rule-id'            # required, must be unique across all files
    description: 'Human note'       # optional
    priority: 500                   # required, integer >= 0, higher = evaluated first
    match:
      resource: 'tool'              # optional — 'tool' matches all Hermes tool calls
      action: 'execute'             # optional — 'execute' or 'call' match tool calls
      resourceId: 'terminal'        # optional — exact tool name, or list of names
    conditions:                     # optional, all must match (AND)
      - field: 'tool_name'
        operator: 'equals'
        value: 'terminal'
    effect: 'allow'                 # required — 'allow' or 'deny'
    reason: 'Human-readable why'    # shown to user on deny

Condition operators:
  equals, not_equals, in, not_in, contains, matches (regex), starts_with, ends_with

Field paths available in conditions:
  tool_name              — name of the tool being called
  tool_args.<key>        — any top-level key in the tool's argument dict
  metadata.<key>         — alias for tool_args.<key>
  context.<key>          — extra context passed by the caller


Example: deny the terminal tool
--------------------------------

Create ~/.hermes/nachos/policies/deny-terminal.yaml:

  version: '1.0'

  metadata:
    name: 'Block raw terminal'
    description: 'Require explicit approval before running shell commands'

  rules:
    - id: 'deny-terminal-tool'
      description: 'Block terminal / shell execution'
      priority: 1000
      match:
        resource: 'tool'
        resourceId:
          - 'terminal'
          - 'shell'
          - 'bash'
      effect: 'deny'
      reason: >
        Direct shell execution is blocked by policy.
        Ask your administrator to allow specific commands.

This rule fires before standard-allow-all-tools (priority 100) and denies
calls to terminal, shell, and bash unconditionally.


Dry-run testing
---------------

If the plugin registers the nachos_policy_check tool you can ask Hermes:

  nachos_policy_check(tool_name='terminal', tool_args={'command': 'ls'})

Returns: { "allowed": bool, "reason": str, "stats": { ... } }


Hot reload
----------

The engine polls the policies directory every 5 seconds.  Edit a YAML file
and it is picked up automatically.  If any file fails validation the entire
reload is rejected and the previous valid ruleset is kept intact (atomic
reload).  Validation errors are logged at ERROR level.


Failure-open guarantee
----------------------

If the policy engine itself throws an unexpected exception during evaluation
(a bug, not a deny decision), it logs a WARNING and allows the tool call
through.  Policy bugs never silently kill tool execution.
