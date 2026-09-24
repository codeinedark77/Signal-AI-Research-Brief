#!/usr/bin/env python3
"""Structural validation for signal_workflow.json.

This is NOT a substitute for a real n8n import -- it can't know whether
"n8n-nodes-base.googleCalendar" takes exactly these parameter names in
your installed version. What it DOES catch, mechanically and for real:
dangling connection references, duplicate node ids/names, orphaned nodes,
and malformed positions -- the class of error that comes from hand-editing
JSON rather than from a specific node's schema drifting between versions.

Exits 0 and prints "OK" on success; exits 1 and prints every problem found
otherwise (does not stop at the first one).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REQUIRED_NODE_KEYS = {"id", "name", "type", "typeVersion", "position", "parameters"}


def validate(workflow: dict) -> list[str]:
    errors: list[str] = []

    for key in ("name", "nodes", "connections"):
        if key not in workflow:
            errors.append(f"Missing required top-level key: {key!r}")
    if errors:
        return errors  # can't check anything else meaningfully without these

    nodes = workflow["nodes"]
    connections = workflow["connections"]

    if not isinstance(nodes, list) or not nodes:
        errors.append("'nodes' must be a non-empty list")
        return errors

    names_seen: dict[str, int] = {}
    ids_seen: dict[str, int] = {}
    node_names = set()

    for i, node in enumerate(nodes):
        missing = REQUIRED_NODE_KEYS - node.keys()
        if missing:
            errors.append(f"nodes[{i}] ({node.get('name', '?')!r}) missing keys: {sorted(missing)}")
            continue

        name, node_id = node["name"], node["id"]
        node_names.add(name)
        names_seen[name] = names_seen.get(name, 0) + 1
        ids_seen[node_id] = ids_seen.get(node_id, 0) + 1

        pos = node["position"]
        if not (isinstance(pos, list) and len(pos) == 2 and all(isinstance(p, (int, float)) for p in pos)):
            errors.append(f"nodes[{i}] ({name!r}) has malformed position: {pos!r}")

        if not isinstance(node["parameters"], dict):
            errors.append(f"nodes[{i}] ({name!r}) 'parameters' must be an object")

    for name, count in names_seen.items():
        if count > 1:
            errors.append(f"Duplicate node name: {name!r} appears {count} times")
    for node_id, count in ids_seen.items():
        if count > 1:
            errors.append(f"Duplicate node id: {node_id!r} appears {count} times")

    # Connections: every source and target must be a real node name.
    targets_referenced = set()
    for source_name, outputs in connections.items():
        if source_name not in node_names:
            errors.append(f"connections references unknown source node: {source_name!r}")
            continue
        main_outputs = outputs.get("main", [])
        for branch in main_outputs:
            for edge in branch:
                target_name = edge.get("node")
                if target_name not in node_names:
                    errors.append(
                        f"connections[{source_name!r}] targets unknown node: {target_name!r}"
                    )
                else:
                    targets_referenced.add(target_name)

    # Orphan check: every node except trigger-type nodes should be reachable
    # as a connection target (a node nobody points to, and which isn't a
    # trigger, is either dead or a forgotten wire).
    trigger_like = {n["name"] for n in nodes if "trigger" in n["type"].lower()}
    unreachable = node_names - targets_referenced - trigger_like
    if unreachable:
        errors.append(f"Node(s) with no incoming connection and not a trigger: {sorted(unreachable)}")

    # Every non-terminal node should have at least one outgoing connection,
    # otherwise the pipeline silently dead-ends there.
    nodes_with_outgoing = set(connections.keys())
    terminal_candidates = node_names - nodes_with_outgoing
    if len(terminal_candidates) == 0:
        errors.append("No terminal node found -- every node has an outgoing connection (cycle?)")
    elif len(terminal_candidates) > 1:
        errors.append(
            f"More than one node has no outgoing connection: {sorted(terminal_candidates)} "
            "(fine if intentional, flagged because it usually isn't)"
        )

    return errors


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "n8n/signal_workflow.json")
    workflow = json.loads(path.read_text())
    errors = validate(workflow)
    if errors:
        print(f"FAILED: {len(errors)} problem(s) found in {path}")
        for e in errors:
            print(f"  - {e}")
        return 1
    print(f"OK: {path} passes structural validation "
          f"({len(workflow['nodes'])} nodes, {len(workflow['connections'])} wired outputs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
