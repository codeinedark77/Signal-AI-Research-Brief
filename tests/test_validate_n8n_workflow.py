from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from validate_n8n_workflow import validate  # noqa: E402

VALID = {
    "name": "test",
    "nodes": [
        {
            "id": "1",
            "name": "Trigger",
            "type": "n8n-nodes-base.scheduleTrigger",
            "typeVersion": 1.2,
            "position": [0, 0],
            "parameters": {},
        },
        {
            "id": "2",
            "name": "Do Thing",
            "type": "n8n-nodes-base.httpRequest",
            "typeVersion": 4.2,
            "position": [200, 0],
            "parameters": {},
        },
    ],
    "connections": {"Trigger": {"main": [[{"node": "Do Thing", "type": "main", "index": 0}]]}},
}


def test_real_workflow_file_is_valid():
    path = Path(__file__).parent.parent / "n8n" / "signal_workflow.json"
    workflow = json.loads(path.read_text())
    assert validate(workflow) == []


def test_real_evening_capture_workflow_file_is_valid():
    path = Path(__file__).parent.parent / "n8n" / "signal_evening_capture.json"
    workflow = json.loads(path.read_text())
    assert validate(workflow) == []


def test_valid_fixture_has_no_errors():
    assert validate(VALID) == []


def test_catches_dangling_connection_target():
    broken = json.loads(json.dumps(VALID))
    broken["connections"]["Trigger"]["main"][0][0]["node"] = "Nonexistent Node"
    errors = validate(broken)
    assert any("unknown node" in e for e in errors)


def test_catches_duplicate_node_name():
    broken = json.loads(json.dumps(VALID))
    broken["nodes"][1]["name"] = "Trigger"  # collides with nodes[0]
    errors = validate(broken)
    assert any("Duplicate node name" in e for e in errors)


def test_catches_duplicate_node_id():
    broken = json.loads(json.dumps(VALID))
    broken["nodes"][1]["id"] = "1"  # collides with nodes[0]
    errors = validate(broken)
    assert any("Duplicate node id" in e for e in errors)


def test_catches_orphaned_node():
    broken = json.loads(json.dumps(VALID))
    broken["nodes"].append(
        {
            "id": "3",
            "name": "Orphan",
            "type": "n8n-nodes-base.code",
            "typeVersion": 2,
            "position": [400, 0],
            "parameters": {},
        }
    )
    errors = validate(broken)
    assert any("no incoming connection" in e and "Orphan" in e for e in errors)


def test_catches_missing_required_node_key():
    broken = json.loads(json.dumps(VALID))
    del broken["nodes"][0]["typeVersion"]
    errors = validate(broken)
    assert any("missing keys" in e for e in errors)


def test_catches_malformed_position():
    broken = json.loads(json.dumps(VALID))
    broken["nodes"][0]["position"] = "not-a-list"
    errors = validate(broken)
    assert any("malformed position" in e for e in errors)


def test_catches_missing_top_level_key():
    broken = json.loads(json.dumps(VALID))
    del broken["connections"]
    errors = validate(broken)
    assert any("Missing required top-level key" in e for e in errors)
