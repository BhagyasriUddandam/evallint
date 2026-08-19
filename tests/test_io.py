"""Tests for evallint.io.

Two things are being proven here:
  1. Every supported format loads into the same EvalSet.
  2. Malformed input fails loudly, with the file and location in the message.

(2) matters as much as (1): a loader that silently drops a bad row would make
every downstream check quietly wrong about how big the eval set is.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evallint.io import LoadError, load
from evallint.schema import EvalCase, EvalSet

EXAMPLE_SET = Path(__file__).resolve().parents[1] / "examples" / "sample_evalset.jsonl"

GOOD_RECORDS = [
    {"id": "a1", "input": "What is 2 + 2?", "expected": "4", "label": "math"},
    {"id": "a2", "input": "Capital of France?", "expected": "Paris", "label": "geo"},
]


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def jsonl(records: list[dict]) -> str:
    return "\n".join(json.dumps(r) for r in records) + "\n"


# --------------------------------------------------------------------------
# Happy path: one dataset, three file formats, same result
# --------------------------------------------------------------------------


def test_loads_jsonl(tmp_path: Path) -> None:
    eval_set = load(write(tmp_path / "cases.jsonl", jsonl(GOOD_RECORDS)))

    assert isinstance(eval_set, EvalSet)
    assert len(eval_set) == 2
    assert [c.id for c in eval_set] == ["a1", "a2"]
    assert eval_set.cases[0].input == "What is 2 + 2?"
    assert eval_set.cases[0].expected == "4"
    assert eval_set.cases[0].label == "math"
    assert eval_set.source == str(tmp_path / "cases.jsonl")


def test_loads_ndjson_extension(tmp_path: Path) -> None:
    eval_set = load(write(tmp_path / "cases.ndjson", jsonl(GOOD_RECORDS)))
    assert len(eval_set) == 2


def test_loads_json_list(tmp_path: Path) -> None:
    eval_set = load(write(tmp_path / "cases.json", json.dumps(GOOD_RECORDS)))
    assert [c.id for c in eval_set] == ["a1", "a2"]


def test_loads_json_object_with_cases_key(tmp_path: Path) -> None:
    payload = {"name": "my eval", "cases": GOOD_RECORDS}
    eval_set = load(write(tmp_path / "cases.json", json.dumps(payload)))
    assert [c.id for c in eval_set] == ["a1", "a2"]


def test_loads_csv(tmp_path: Path) -> None:
    text = (
        "id,input,expected,label\n"
        "a1,What is 2 + 2?,4,math\n"
        "a2,Capital of France?,Paris,geo\n"
    )
    eval_set = load(write(tmp_path / "cases.csv", text))

    assert [c.id for c in eval_set] == ["a1", "a2"]
    assert eval_set.cases[1].expected == "Paris"
    assert eval_set.cases[1].label == "geo"


def test_all_three_formats_produce_the_same_cases(tmp_path: Path) -> None:
    """The point of io.py: format is an implementation detail below the schema."""
    from_jsonl = load(write(tmp_path / "c.jsonl", jsonl(GOOD_RECORDS)))
    from_json = load(write(tmp_path / "c.json", json.dumps(GOOD_RECORDS)))
    from_csv = load(
        write(
            tmp_path / "c.csv",
            "id,input,expected,label\n"
            "a1,What is 2 + 2?,4,math\n"
            "a2,Capital of France?,Paris,geo\n",
        )
    )

    def comparable(s: EvalSet) -> list[tuple]:
        return [(c.id, c.input, c.expected, c.label) for c in s]

    assert comparable(from_jsonl) == comparable(from_json) == comparable(from_csv)


# --------------------------------------------------------------------------
# Field handling
# --------------------------------------------------------------------------


def test_missing_ids_are_generated_by_position(tmp_path: Path) -> None:
    records = [{"input": "first"}, {"input": "second"}]
    eval_set = load(write(tmp_path / "cases.jsonl", jsonl(records)))
    assert [c.id for c in eval_set] == ["case_1", "case_2"]


def test_unknown_fields_are_kept_as_metadata(tmp_path: Path) -> None:
    records = [{"id": "a1", "input": "hi", "difficulty": "hard", "author": "bhagi"}]
    eval_set = load(write(tmp_path / "cases.jsonl", jsonl(records)))

    assert eval_set.cases[0].metadata == {"difficulty": "hard", "author": "bhagi"}


def test_optional_fields_default_to_none(tmp_path: Path) -> None:
    eval_set = load(write(tmp_path / "cases.jsonl", jsonl([{"id": "a1", "input": "hi"}])))
    case = eval_set.cases[0]

    assert case.expected is None
    assert case.label is None
    assert case.metadata == {}


def test_numeric_label_is_normalised_to_string(tmp_path: Path) -> None:
    """1 and "1" must not count as two different classes in the imbalance check."""
    records = [{"id": "a1", "input": "hi", "label": 1}, {"id": "a2", "input": "yo", "label": "1"}]
    eval_set = load(write(tmp_path / "cases.jsonl", jsonl(records)))

    assert [c.label for c in eval_set] == ["1", "1"]


def test_non_string_expected_is_allowed(tmp_path: Path) -> None:
    """Ground truth is not always text: booleans and lists are legitimate."""
    records = [
        {"id": "a1", "input": "Is the sky blue?", "expected": True},
        {"id": "a2", "input": "Name two colours", "expected": ["red", "blue"]},
    ]
    eval_set = load(write(tmp_path / "cases.jsonl", jsonl(records)))

    assert eval_set.cases[0].expected is True
    assert eval_set.cases[1].expected == ["red", "blue"]


def test_empty_csv_cells_become_none(tmp_path: Path) -> None:
    """CSV has no null, so a blank cell must mean absent, not empty string."""
    text = "id,input,expected,label\na1,What is 2 + 2?,,\n"
    eval_set = load(write(tmp_path / "cases.csv", text))

    assert eval_set.cases[0].expected is None
    assert eval_set.cases[0].label is None


def test_blank_lines_in_jsonl_are_skipped(tmp_path: Path) -> None:
    text = json.dumps(GOOD_RECORDS[0]) + "\n\n" + json.dumps(GOOD_RECORDS[1]) + "\n\n"
    eval_set = load(write(tmp_path / "cases.jsonl", text))
    assert len(eval_set) == 2


# --------------------------------------------------------------------------
# Malformed input must fail clearly (and say where)
# --------------------------------------------------------------------------


def test_missing_file_errors(tmp_path: Path) -> None:
    with pytest.raises(LoadError, match="no such file"):
        load(tmp_path / "nope.jsonl")


def test_unsupported_extension_errors(tmp_path: Path) -> None:
    with pytest.raises(LoadError, match="unsupported file type"):
        load(write(tmp_path / "cases.txt", "whatever"))


def test_invalid_json_line_reports_the_line_number(tmp_path: Path) -> None:
    text = json.dumps(GOOD_RECORDS[0]) + "\n{not valid json\n"
    with pytest.raises(LoadError, match=r"cases\.jsonl line 2: invalid JSON"):
        load(write(tmp_path / "cases.jsonl", text))


def test_missing_input_field_reports_the_location(tmp_path: Path) -> None:
    records = [{"id": "a1", "input": "fine"}, {"id": "a2", "expected": "oops"}]
    with pytest.raises(LoadError, match=r"line 2: missing required field 'input'"):
        load(write(tmp_path / "cases.jsonl", jsonl(records)))


def test_empty_input_is_rejected(tmp_path: Path) -> None:
    records = [{"id": "a1", "input": "   "}]
    with pytest.raises(LoadError, match="input must be a non-empty string"):
        load(write(tmp_path / "cases.jsonl", jsonl(records)))


def test_non_string_input_is_rejected(tmp_path: Path) -> None:
    records = [{"id": "a1", "input": 42}]
    with pytest.raises(LoadError, match="input must be a string, got int"):
        load(write(tmp_path / "cases.jsonl", jsonl(records)))


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    """Two rows claiming the same id would make every report ambiguous."""
    records = [{"id": "a1", "input": "one"}, {"id": "a1", "input": "two"}]
    with pytest.raises(LoadError, match="case ids must be unique, repeated: a1"):
        load(write(tmp_path / "cases.jsonl", jsonl(records)))


def test_empty_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(LoadError, match="at least one case"):
        load(write(tmp_path / "cases.jsonl", ""))


def test_empty_csv_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(LoadError, match="no header row"):
        load(write(tmp_path / "cases.csv", ""))


def test_json_object_without_cases_key_errors(tmp_path: Path) -> None:
    with pytest.raises(LoadError, match="must have a 'cases' key"):
        load(write(tmp_path / "cases.json", json.dumps({"rows": GOOD_RECORDS})))


def test_json_scalar_payload_errors(tmp_path: Path) -> None:
    with pytest.raises(LoadError, match="expected a list of cases"):
        load(write(tmp_path / "cases.json", json.dumps("just a string")))


def test_non_object_record_errors(tmp_path: Path) -> None:
    with pytest.raises(LoadError, match="item 1: expected an object, got str"):
        load(write(tmp_path / "cases.json", json.dumps(["not a case"])))


def test_list_label_is_rejected(tmp_path: Path) -> None:
    records = [{"id": "a1", "input": "hi", "label": ["a", "b"]}]
    with pytest.raises(LoadError, match="label must be a string or number, got list"):
        load(write(tmp_path / "cases.jsonl", jsonl(records)))


# --------------------------------------------------------------------------
# The shipped example loads, and still contains the flaws it claims to
# --------------------------------------------------------------------------


def test_example_evalset_loads(tmp_path: Path) -> None:
    eval_set = load(EXAMPLE_SET)

    assert len(eval_set) == 20
    assert all(isinstance(c, EvalCase) for c in eval_set)
    assert all(c.label for c in eval_set)


def test_example_evalset_still_contains_its_planted_flaws() -> None:
    """Guards examples/README.md against drift: if someone 'cleans up' the
    sample set, the checks would have nothing to catch and the demo would
    silently become useless."""
    eval_set = load(EXAMPLE_SET)
    by_id = {c.id: c for c in eval_set}

    # planted exact duplicate
    assert by_id["billing_004"].input == by_id["billing_014"].input

    # planted class imbalance
    counts: dict[str, int] = {}
    for case in eval_set:
        counts[case.label] = counts.get(case.label, 0) + 1
    assert counts == {"billing": 14, "password_reset": 4, "refund": 1, "bug_report": 1}


# --------------------------------------------------------------------------
# UTF-8 BOM — the Excel / PowerShell export case
# --------------------------------------------------------------------------


def test_bom_csv_keeps_the_real_ids(tmp_path) -> None:
    """A BOM'd CSV used to load "successfully" with its id column named
    "﻿id", so real ids were demoted to metadata and every case was
    renamed case_1, case_2... Findings then named cases that did not exist in
    the user's file. It never errored, which made it worse than a crash."""
    p = tmp_path / "excel.csv"
    p.write_bytes(b"\xef\xbb\xbfid,input,expected,label\na,hello,x,l1\nb,world,y,l2\n")

    eval_set = load(p)

    assert [c.id for c in eval_set] == ["a", "b"]
    assert all(c.metadata == {} for c in eval_set), "no BOM'd key leaked to metadata"
    assert eval_set.cases[0].label == "l1"


def test_bom_jsonl_loads(tmp_path) -> None:
    p = tmp_path / "bom.jsonl"
    p.write_bytes(b'\xef\xbb\xbf{"id":"a","input":"x"}\n')

    assert [c.id for c in load(p)] == ["a"]


def test_bom_json_loads(tmp_path) -> None:
    p = tmp_path / "bom.json"
    p.write_bytes(b'\xef\xbb\xbf[{"id":"a","input":"x"}]')

    assert [c.id for c in load(p)] == ["a"]


def test_files_without_a_bom_are_unaffected(tmp_path) -> None:
    """utf-8-sig must be a no-op on ordinary files."""
    p = tmp_path / "plain.csv"
    p.write_text("id,input\na,hello\n", encoding="utf-8")

    assert [c.id for c in load(p)] == ["a"]
