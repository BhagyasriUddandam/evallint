"""Tests for evallint.migrate.

Migration is optional -- a version-1 file is valid forever -- so the bar here is
that it must be SAFE rather than merely useful. Two properties matter most:

  1. **The data is unchanged.** Field names and the version line change; nothing
     else does. Proven by reloading the output and comparing every field.
  2. **A failure writes nothing.** Every refusal is checked to leave the
     destination absent and the source untouched.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evallint.io import load, load_with_mapping
from evallint.migrate import MigrationError, migrate_file, to_payload
from evallint.schema import SCHEMA_VERSION

GSM8K = [
    {"question": "Natalia sold 48 clips in April, half as many in May. Total?",
     "answer": "72"},
    {"question": "Weng earns $12/hour and sat 50 minutes. How much?", "answer": "10"},
]


def write_jsonl(path: Path, records: list[dict]) -> Path:
    path.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
    )
    return path


# --------------------------------------------------------------------------
# The point of the feature: an inferred mapping becomes permanent
# --------------------------------------------------------------------------


def test_migration_makes_an_inferred_mapping_permanent(tmp_path):
    """GSM8K needs --map (or inference) on every run. After migrating, the
    canonical names are in the file and the mapping is the identity."""
    source = write_jsonl(tmp_path / "gsm8k.jsonl", GSM8K)
    dest = tmp_path / "gsm8k_v2.jsonl"

    report = migrate_file(source, dest)
    assert report.renamed == {"input": "question", "expected": "answer"}
    assert report.cases == 2
    assert report.verified

    _, mapping = load_with_mapping(dest)
    assert mapping.is_identity()


def test_the_migrated_file_declares_the_version(tmp_path):
    source = write_jsonl(tmp_path / "in.jsonl", GSM8K)
    dest = tmp_path / "out.jsonl"
    migrate_file(source, dest)

    first_line = dest.read_text(encoding="utf-8").splitlines()[0]
    assert json.loads(first_line) == {"evallint_schema": SCHEMA_VERSION}
    assert load(dest).schema_version == SCHEMA_VERSION


def test_every_field_survives_the_round_trip(tmp_path):
    """The whole rich schema in, the whole rich schema out."""
    record = {
        "id": "refund",
        "system": "You are a support agent.",
        "messages": [
            {"role": "user", "content": "My order never arrived."},
            {"role": "assistant",
             "tool_calls": [{"name": "lookup", "arguments": {"u": "7"}, "id": "c1"}]},
            {"role": "tool", "content": "lost", "tool_call_id": "c1", "name": "lookup"},
        ],
        "tools": [{"name": "lookup", "description": "Find an order"},
                  {"name": "start_return"}],
        "expected_tool_calls": [{"name": "start_return", "arguments": {"o": "1"}}],
        "expected": {"action": "start_return"},
        "acceptable": [{"action": "escalate"}],
        "expected_schema": {"type": "object", "required": ["action"]},
        "rubric": {"criteria": [{"name": "policy", "weight": 3},
                                {"name": "empathy"}], "scale": [1, 5]},
        "label": "support",
        "ticket": "T-991",
    }
    source = tmp_path / "rich.json"
    source.write_text(json.dumps({"evallint_schema": 2, "cases": [record]}))
    dest = tmp_path / "rich_v2.json"

    report = migrate_file(source, dest)
    assert set(report.v2_fields_used) == {
        "messages", "system", "acceptable", "expected_schema", "rubric",
        "tools", "expected_tool_calls",
    }
    assert load(source).cases[0] == load(dest).cases[0]


def test_a_simple_case_is_not_turned_into_a_conversation(tmp_path):
    """Deliberate. The conversion would be lossless but would make a simple
    file harder to read for nothing, and the four-field form stays
    first-class."""
    source = write_jsonl(tmp_path / "in.jsonl", [{"id": "a", "input": "2+2?"}])
    dest = tmp_path / "out.jsonl"
    migrate_file(source, dest)

    case_line = json.loads(dest.read_text(encoding="utf-8").splitlines()[1])
    assert case_line == {"id": "a", "input": "2+2?"}
    assert "messages" not in case_line


def test_migration_resolves_an_ambiguous_turns_list(tmp_path):
    """MT-Bench's list-of-strings has to be INTERPRETED on every load, which is
    reported as a note each time. Migrating writes explicit role objects, so the
    interpretation happens once and the note stops."""
    source = write_jsonl(
        tmp_path / "mt.jsonl",
        [{"question_id": 81, "turns": ["Write a post.", "Rewrite it."]}],
    )
    dest = tmp_path / "mt_v2.json"

    report = migrate_file(source, dest)
    assert any("successive USER turns" in n for n in report.load_notes)

    reloaded = load(dest)
    assert reloaded.load_notes == ()
    assert reloaded.cases[0].input == load(source).cases[0].input


def test_an_explicit_field_map_is_honoured(tmp_path):
    source = write_jsonl(
        tmp_path / "hella.jsonl",
        [{"ctx": "A man opens a door", "label": 2, "activity_label": "doors"}],
    )
    dest = tmp_path / "out.jsonl"
    report = migrate_file(source, dest, field_map={"label": "activity_label"})
    assert report.renamed["label"] == "activity_label"
    assert load(dest).cases[0].label == "doors"


def test_metadata_is_preserved_not_dropped(tmp_path):
    source = write_jsonl(
        tmp_path / "in.jsonl", [{"id": "a", "input": "q", "ticket": "T-1", "n": 3}]
    )
    dest = tmp_path / "out.jsonl"
    migrate_file(source, dest)
    assert load(dest).cases[0].metadata == {"ticket": "T-1", "n": 3}


def test_to_payload_shape(tmp_path):
    eval_set = load(write_jsonl(tmp_path / "in.jsonl", [{"id": "a", "input": "q"}]))
    payload = to_payload(eval_set)
    assert payload["evallint_schema"] == SCHEMA_VERSION
    assert payload["cases"] == [{"id": "a", "input": "q"}]


# --------------------------------------------------------------------------
# Refusals. Each one must leave nothing behind.
# --------------------------------------------------------------------------


def test_an_existing_destination_is_not_overwritten_by_default(tmp_path):
    source = write_jsonl(tmp_path / "in.jsonl", GSM8K)
    dest = tmp_path / "out.jsonl"
    dest.write_text("PRECIOUS", encoding="utf-8")

    with pytest.raises(MigrationError, match="already exists"):
        migrate_file(source, dest)
    assert dest.read_text(encoding="utf-8") == "PRECIOUS"

    migrate_file(source, dest, overwrite=True)
    assert dest.read_text(encoding="utf-8") != "PRECIOUS"


def test_migrating_onto_the_source_is_refused(tmp_path):
    """The original has to survive if anything goes wrong, so in-place is out."""
    source = write_jsonl(tmp_path / "in.jsonl", GSM8K)
    with pytest.raises(MigrationError, match="destination is the source"):
        migrate_file(source, source, overwrite=True)
    assert len(load(source)) == 2


def test_csv_is_not_a_valid_destination(tmp_path):
    source = write_jsonl(tmp_path / "in.jsonl", GSM8K)
    dest = tmp_path / "out.csv"
    with pytest.raises(MigrationError, match="CSV cannot hold"):
        migrate_file(source, dest)
    assert not dest.exists()


def test_an_unreadable_source_writes_nothing(tmp_path):
    source = tmp_path / "broken.jsonl"
    source.write_text('{"id": "a", "input": ', encoding="utf-8")
    dest = tmp_path / "out.jsonl"
    with pytest.raises(MigrationError, match="invalid JSON"):
        migrate_file(source, dest)
    assert not dest.exists()


def test_verification_failure_writes_nothing(tmp_path, monkeypatch):
    """The safety net itself. If serialisation ever loses a field, migration
    must refuse rather than write a quietly corrupt file -- so the failure path
    is exercised by breaking as_dict on purpose."""
    import evallint.schema as schema_module

    source = write_jsonl(tmp_path / "in.jsonl", [{"id": "a", "input": "q",
                                                 "label": "cls"}])
    dest = tmp_path / "out.jsonl"

    def lossy(self):
        return {"id": self.id, "input": self.input}  # drops the label

    monkeypatch.setattr(schema_module.EvalCase, "as_dict", lossy)
    with pytest.raises(MigrationError) as exc:
        migrate_file(source, dest)
    assert "case 'a' changed during migration" in str(exc.value)
    assert "label" in str(exc.value)
    assert not dest.exists()


def test_report_renders_the_facts(tmp_path):
    source = write_jsonl(tmp_path / "gsm8k.jsonl", GSM8K)
    dest = tmp_path / "out.jsonl"
    text = migrate_file(source, dest).render()
    assert "question -> input" in text
    assert "2 case(s), schema version 2 declared" in text
    assert "verified" in text
