"""Turning raw records into cases, with errors a person can act on.

schema.py holds the RULES. This module applies them to untrusted data from a
file and reports every violation with the exact place it happened.

## Two things this does that first-failure validation cannot

**It collects.** A 5000-case file with a mistake in 40 cases should take one run
to diagnose, not 40. Every issue found is accumulated and reported together.

**It says where.** Each issue carries a JSON path — ``cases[3].messages[1].role``
— the value that was wrong, and what was allowed instead. Near-miss keys get a
suggestion, because ``expected_schemas`` silently becoming metadata is worse
than being told it is not a field.

## The version is what decides how strict to be

This is the whole job of ``evallint_schema``, and it is deliberately narrow:

  * **Undeclared (version 1)** — lenient. Any key this build does not recognise
    stays in ``metadata``, exactly as it always has. A ``messages`` column that
    does not parse as a conversation is DEMOTED to metadata with a note, not
    rejected. An existing dataset must keep loading, and a dataset that happens
    to have a column called ``messages`` holding something else is a real thing.
  * **Declared version 2** — strict. The user has said "this file follows the
    rich schema", so a ``messages`` value that is not a conversation is an
    error, and a key that is one edit away from a real field is an error rather
    than silently-ignored metadata.

The version therefore never gates FEATURES. Rich fields are read whenever they
are present and well-formed, because dropping a user's conversation into
metadata in order to be pedantic about a header line would be the silent data
loss this package exists to complain about. Declaring the version buys stricter
errors, and refuses a future version outright instead of misreading it.
"""

from __future__ import annotations

import difflib
import json
from dataclasses import dataclass, field, replace
from typing import Any

from .schema import (
    SUPPORTED_SCHEMA_VERSIONS,
    Criterion,
    EvalCase,
    Message,
    Role,
    Rubric,
    SchemaError,
    ToolCall,
    ToolSpec,
    derive_input,
)

__all__ = [
    "CANONICAL_FIELDS",
    "MAX_REPORTED_ISSUES",
    "ParsedCases",
    "SchemaValidationError",
    "ValidationIssue",
    "check_version",
    "parse_case",
    "parse_cases",
]

#: Every key this build reads as a field rather than as metadata. Anything else
#: is the user's own bookkeeping and is preserved untouched.
CANONICAL_FIELDS = (
    "id",
    "input",
    "expected",
    "label",
    "messages",
    "system",
    "acceptable",
    "expected_schema",
    "rubric",
    "tools",
    "expected_tool_calls",
)

#: How many issues an error message prints. The rest are counted, and all of
#: them stay on the exception for a caller that wants them. A wall of 4000 lines
#: is not more useful than 20 and a total.
MAX_REPORTED_ISSUES = 20

#: How similar an unknown key must be to a real field before we suggest it.
#: 0.82 accepts 'expected_schemas' -> 'expected_schema' and 'sytem' -> 'system'
#: while rejecting 'tool' -> 'tools' ... which it does NOT, and that is fine:
#: a false suggestion costs a moment's reading, a missed one costs a debugging
#: session. Tuned on the near-misses in tests/test_validation_errors.py.
SUGGESTION_CUTOFF = 0.82


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One specific thing wrong, and where."""

    path: str
    message: str
    hint: str | None = None
    #: Which record this issue belongs to, for counting affected cases. Carried
    #: rather than parsed back out of `path`: a source file named `f.jsonl`
    #: makes `path.split(".")[0]` return `f` for every issue in the file.
    case: str | None = None

    def render(self) -> str:
        text = f"  {self.path}: {self.message}"
        if self.hint:
            text += f"\n      {self.hint}"
        return text


class SchemaValidationError(SchemaError):
    """One or more records could not be read as cases.

    Subclasses SchemaError, which subclasses ValueError, so existing callers
    that catch either keep working unchanged.
    """

    def __init__(
        self,
        issues: list[ValidationIssue] | tuple[ValidationIssue, ...],
        *,
        source: str | None = None,
    ) -> None:
        self.issues = tuple(issues)
        self.source = source
        super().__init__(self._render())

    def _render(self) -> str:
        n = len(self.issues)
        where = f"{self.source}: " if self.source else ""
        head = f"{where}{n} schema problem{'s' if n != 1 else ''}"
        cases = {i.case for i in self.issues if i.case is not None}
        if len(cases) > 1:
            head += f" in {len(cases)} cases"
        lines = [head, ""]
        lines.extend(i.render() for i in self.issues[:MAX_REPORTED_ISSUES])
        if n > MAX_REPORTED_ISSUES:
            lines.append(f"  ... and {n - MAX_REPORTED_ISSUES} more")
        return "\n".join(lines)


@dataclass(slots=True)
class ParsedCases:
    """The result of reading a batch of records.

    ``notes`` records decisions that were not errors but that the user must be
    able to see — above all, a field DEMOTED to metadata. A demotion that
    nobody is told about is indistinguishable from the field never having been
    written.
    """

    cases: list[EvalCase] = field(default_factory=list)
    issues: list[ValidationIssue] = field(default_factory=list)
    #: (path, reason) pairs, collapsed into `notes` by `parse_cases`. Stored
    #: apart from their location so that one reason affecting 80 cases prints
    #: once with a count, rather than 80 near-identical lines nobody reads.
    observations: list[tuple[str, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def parse_cases(
    records: list[tuple[str, Any]],
    *,
    strict: bool,
    source: str | None = None,
) -> ParsedCases:
    """Read every record, collecting issues rather than stopping at the first.

    Args:
        records: ``(where, record)`` pairs; ``where`` is a human-readable
            location such as ``"sample.jsonl line 4"``, used in issue paths so
            the message points at a place in the user's file.
        strict: True when the file declared schema version 2. See the module
            docstring for exactly what it changes.
        source: file name, for the error header.

    Raises:
        SchemaValidationError: if any record produced an issue. Every issue
            found across every record is on the exception.
    """
    out = ParsedCases()
    for position, (where, record) in enumerate(records, start=1):
        parse_case(
            record,
            where=where,
            position=position,
            strict=strict,
            into=out,
        )
    if out.issues:
        raise SchemaValidationError(out.issues, source=source)
    out.notes = _collapse(out.observations)
    return out


def _collapse(observations: list[tuple[str, str]]) -> list[str]:
    """One line per distinct reason, with a count and the first place it hit."""
    grouped: dict[str, list[str]] = {}
    for path, reason in observations:
        grouped.setdefault(reason, []).append(path)
    notes = []
    for reason, paths in grouped.items():
        if len(paths) == 1:
            notes.append(f"{paths[0]}: {reason}")
        else:
            notes.append(f"{reason} — {len(paths)} cases, first at {paths[0]}")
    return notes


def parse_case(
    record: Any,
    *,
    where: str,
    position: int,
    strict: bool,
    into: ParsedCases,
) -> None:
    """Read one record, appending its case or its issues to ``into``."""
    if not isinstance(record, dict):
        into.issues.append(
            ValidationIssue(
                where, f"expected an object, got {type(record).__name__}",
                case=where,
            )
        )
        return

    data = dict(record)  # copy: canonical fields are popped, remainder is metadata
    issues: list[ValidationIssue] = []

    raw_id = data.pop("id", None)
    case_id = f"case_{position}" if raw_id is None else str(raw_id)

    if strict:
        _flag_near_miss_keys(data, where, issues)

    had_messages = "messages" in data
    messages = _parse_messages(data, where, issues, into, strict)
    tools = _parse_tools(data, where, issues, into, strict)
    expected_calls = _parse_tool_calls(
        data.pop("expected_tool_calls", None),
        f"{where}.expected_tool_calls",
        issues,
        into,
        strict,
        restore=(data, "expected_tool_calls"),
    )
    rubric = _parse_rubric(data, where, issues, into, strict)
    acceptable = _parse_acceptable(data.pop("acceptable", None), where, issues)
    expected_schema = _parse_expected_schema(data, where, issues, into, strict)
    system = data.pop("system", None)
    if system is not None and not isinstance(system, str):
        issues.append(
            ValidationIssue(
                f"{where}.system",
                f"must be a string, got {type(system).__name__}",
                hint="a system prompt is one block of text; put per-case "
                "settings in metadata instead",
            )
        )
        system = None

    raw_input = data.pop("input", None)
    derived = False
    if raw_input is None:
        if messages:
            raw_input = derive_input(messages)
            derived = True
            if not raw_input.strip():
                issues.append(
                    ValidationIssue(
                        f"{where}.messages",
                        "no turn produced any text, so there is nothing to "
                        "give a model",
                    )
                )
        elif not had_messages:
            issues.append(
                ValidationIssue(
                    where,
                    "missing required field 'input'",
                    hint="every case needs 'input', or 'messages' for a "
                    "multi-turn case",
                )
            )
        # else: 'messages' was there and did not parse. That failure is already
        # reported above, and "missing required field 'input'" on top of it
        # describes a consequence rather than a cause.

    if issues:
        # One stamping point for every field-level issue, so no parser has to
        # remember to pass the location through.
        into.issues.extend(replace(i, case=where) for i in issues)
        return

    try:
        into.cases.append(
            EvalCase(
                id=case_id,
                input=raw_input,
                expected=data.pop("expected", None),
                label=data.pop("label", None),
                metadata=data,
                messages=messages,
                system=system,
                acceptable=acceptable,
                expected_schema=expected_schema,
                rubric=rubric,
                tools=tools,
                expected_tool_calls=expected_calls,
                input_is_derived=derived,
            )
        )
    except SchemaError as exc:
        # The rules in schema.py do not know where the data came from, so the
        # location is attached here rather than duplicating the rule.
        into.issues.append(ValidationIssue(where, str(exc), case=where))


# ---------------------------------------------------------------- field parsers


def _demote(
    into: ParsedCases,
    issues: list[ValidationIssue],
    *,
    path: str,
    reason: str,
    strict: bool,
    hint: str | None = None,
    restore: tuple[dict[str, Any], str] | None = None,
    raw: Any = None,
) -> None:
    """Reject in strict mode; keep as metadata with a note otherwise.

    The lenient branch is what lets an existing dataset with a column called
    ``messages`` holding something else keep loading. The note is what stops
    that from being silent.

    ``restore`` puts the raw value back into the record so it genuinely lands
    in metadata. Without it the note would claim the value was kept while the
    parser had already popped it -- a message that is wrong in the one direction
    that matters, since the user would stop looking for their data.
    """
    if strict:
        issues.append(ValidationIssue(path, reason, hint=hint))
        return
    into.observations.append(
        (
            path,
            f"{reason} — kept as metadata, not read as a field. "
            'Declare "evallint_schema": 2 to make this an error.',
        )
    )
    if restore is not None:
        data, key = restore
        data[key] = raw


def _as_structure(
    value: Any,
    *,
    path: str,
    into: ParsedCases,
    issues: list[ValidationIssue],
    strict: bool,
    expect: str,
    restore: tuple[dict[str, Any], str] | None = None,
    raw: Any = None,
) -> Any:
    """Accept JSON-in-a-string, because CSV has no other way to nest.

    A CSV cell holding ``[{"role": "user", ...}]`` is the only way to put a
    conversation in a spreadsheet, so a string that parses as JSON is read as
    the structure it encodes. A string that does not parse is left alone for
    the caller to reject or demote.
    """
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text[0] not in "[{":
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        _demote(
            into,
            issues,
            path=path,
            reason=f"looks like JSON but does not parse ({exc.msg} at "
            f"position {exc.pos}); expected {expect}",
            strict=strict, restore=restore, raw=raw,
        )
        return None


def _parse_messages(
    data: dict[str, Any],
    where: str,
    issues: list[ValidationIssue],
    into: ParsedCases,
    strict: bool,
) -> tuple[Message, ...]:
    raw_value = data.pop("messages", None)
    if raw_value is None:
        return ()
    path = f"{where}.messages"
    keep = (data, "messages")
    value = _as_structure(
        raw_value, path=path, into=into, issues=issues, strict=strict,
        expect="a list of {role, content} objects", restore=keep, raw=raw_value,
    )
    if value is None:
        return ()
    if not isinstance(value, list):
        _demote(
            into, issues, path=path, strict=strict,
            reason=f"must be a list of turns, got {type(value).__name__}",
            hint='e.g. [{"role": "user", "content": "..."}]',
            restore=keep, raw=raw_value,
        )
        return ()
    if not value:
        _demote(
            into, issues, path=path, strict=strict,
            reason="is an empty list, so the case has no conversation",
            restore=keep, raw=raw_value,
        )
        return ()

    if all(isinstance(item, str) and item.strip() for item in value):
        # MT-Bench and several HuggingFace sets store a conversation as a plain
        # list of strings: ["first question", "follow-up"]. Those are successive
        # USER turns -- the model's replies are not in the file, because the
        # point is to generate them.
        #
        # This is an INTERPRETATION, and a list of strings could in principle be
        # an alternating transcript instead. So it is applied and then reported,
        # the same way an inferred column mapping is: a note is emitted every
        # time, in strict mode too, because a reading that changes the flattened
        # text must be visible rather than merely defensible.
        into.observations.append(
            (
                path,
                "a list of plain strings was read as successive USER turns "
                "(the MT-Bench shape). If they are alternating "
                "user/assistant turns, rewrite them as "
                '{"role": ..., "content": ...} objects.',
            )
        )
        return tuple(Message(role=Role.USER, content=item) for item in value)

    messages: list[Message] = []
    for index, raw in enumerate(value):
        item_path = f"{path}[{index}]"
        if not isinstance(raw, dict):
            _demote(
                into, issues, path=item_path, strict=strict,
                reason=f"must be an object, got {type(raw).__name__}",
                hint='e.g. {"role": "user", "content": "..."}',
                restore=keep, raw=raw_value,
            )
            return ()
        turn = dict(raw)
        role = turn.pop("role", None)
        if role is None:
            _demote(
                into, issues, path=item_path, strict=strict,
                reason="has no 'role'",
                hint="allowed roles: " + ", ".join(r.value for r in Role),
                restore=keep, raw=raw_value,
            )
            return ()
        if isinstance(role, str) and role.strip().lower() not in set(Role):
            issues.append(
                ValidationIssue(
                    f"{item_path}.role",
                    f"{role!r} is not a valid role; allowed: "
                    + ", ".join(r.value for r in Role),
                    hint=_did_you_mean(role, [r.value for r in Role]),
                )
            )
            continue

        calls = _parse_tool_calls(
            turn.pop("tool_calls", None), f"{item_path}.tool_calls",
            issues, into, strict,
        )
        # Content may legitimately be a list of parts in provider formats. Join
        # the text parts rather than rejecting: refusing a real-world shape we
        # can read is worse than reading it.
        content = turn.pop("content", "")
        if isinstance(content, list):
            content = "\n".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        try:
            messages.append(
                Message(
                    role=role,
                    content="" if content is None else content,
                    tool_calls=calls,
                    tool_call_id=turn.pop("tool_call_id", None),
                    name=turn.pop("name", None),
                )
            )
        except SchemaError as exc:
            issues.append(ValidationIssue(item_path, str(exc)))
    return tuple(messages)


def _parse_tool_calls(
    value: Any,
    path: str,
    issues: list[ValidationIssue],
    into: ParsedCases,
    strict: bool,
    restore: tuple[dict[str, Any], str] | None = None,
) -> tuple[ToolCall, ...]:
    if value is None:
        return ()
    raw_value = value
    value = _as_structure(
        value, path=path, into=into, issues=issues, strict=strict,
        expect="a list of {name, arguments} objects",
        restore=restore, raw=raw_value,
    )
    if value is None:
        return ()
    if not isinstance(value, list):
        _demote(
            into, issues, path=path, strict=strict,
            reason=f"must be a list of calls, got {type(value).__name__}",
            hint='e.g. [{"name": "get_weather", "arguments": {"city": "Oslo"}}]',
            restore=restore, raw=raw_value,
        )
        return ()

    calls: list[ToolCall] = []
    for index, raw in enumerate(value):
        item_path = f"{path}[{index}]"
        if not isinstance(raw, dict):
            _demote(
                into, issues, path=item_path, strict=strict,
                reason=f"must be an object, got {type(raw).__name__}",
            )
            continue
        call = dict(raw)
        # OpenAI nests the call under 'function'; accept it so a transcript
        # copied out of a provider response loads without hand-editing.
        if "function" in call and isinstance(call["function"], dict):
            nested = call.pop("function")
            call.setdefault("name", nested.get("name"))
            call.setdefault("arguments", nested.get("arguments"))
        arguments = call.get("arguments")
        if isinstance(arguments, str):
            # Providers serialise arguments as a JSON string. Parse it, but do
            # not invent a dict if it is not one: an unparseable argument blob
            # is a real problem worth naming.
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                issues.append(
                    ValidationIssue(
                        f"{item_path}.arguments",
                        f"is a string that does not parse as JSON ({exc.msg})",
                    )
                )
                continue
        try:
            calls.append(
                ToolCall(
                    name=call.get("name"),
                    arguments={} if arguments is None else arguments,
                    id=call.get("id"),
                )
            )
        except SchemaError as exc:
            issues.append(ValidationIssue(item_path, str(exc)))
    return tuple(calls)


def _parse_tools(
    data: dict[str, Any],
    where: str,
    issues: list[ValidationIssue],
    into: ParsedCases,
    strict: bool,
) -> tuple[ToolSpec, ...]:
    raw_value = data.pop("tools", None)
    if raw_value is None:
        return ()
    path = f"{where}.tools"
    keep = (data, "tools")
    value = _as_structure(
        raw_value, path=path, into=into, issues=issues, strict=strict,
        expect="a list of {name, description, parameters} objects",
        restore=keep, raw=raw_value,
    )
    if value is None:
        return ()
    if not isinstance(value, list):
        _demote(
            into, issues, path=path, strict=strict,
            reason=f"must be a list of tools, got {type(value).__name__}",
            hint='e.g. [{"name": "search", "parameters": {...}}]',
            restore=keep, raw=raw_value,
        )
        return ()

    specs: list[ToolSpec] = []
    for index, raw in enumerate(value):
        item_path = f"{path}[{index}]"
        if isinstance(raw, str):
            # A bare name is a common shorthand and unambiguous, so accept it.
            raw = {"name": raw}
        if not isinstance(raw, dict):
            _demote(
                into, issues, path=item_path, strict=strict,
                reason=f"must be an object or a name, got {type(raw).__name__}",
            )
            continue
        spec = dict(raw)
        if "function" in spec and isinstance(spec["function"], dict):
            spec = dict(spec["function"])
        try:
            specs.append(
                ToolSpec(
                    name=spec.get("name"),
                    description=spec.get("description"),
                    parameters=spec.get("parameters") or spec.get("input_schema") or {},
                )
            )
        except SchemaError as exc:
            issues.append(ValidationIssue(item_path, str(exc)))
    return tuple(specs)


def _parse_rubric(
    data: dict[str, Any],
    where: str,
    issues: list[ValidationIssue],
    into: ParsedCases,
    strict: bool,
) -> Rubric | None:
    raw_value = data.pop("rubric", None)
    if raw_value is None:
        return None
    path = f"{where}.rubric"
    keep = (data, "rubric")
    value = _as_structure(
        raw_value, path=path, into=into, issues=issues, strict=strict,
        expect="an object with a 'criteria' list",
        restore=keep, raw=raw_value,
    )
    if value is None:
        return None

    if isinstance(value, list):
        # A bare list of criteria is the obvious shorthand for a rubric with
        # nothing but criteria, so read it rather than making the user nest it.
        value = {"criteria": value}
    if not isinstance(value, dict):
        _demote(
            into, issues, path=path, strict=strict,
            reason=f"must be an object with a 'criteria' list, got "
            f"{type(value).__name__}",
            hint='e.g. {"criteria": [{"name": "accuracy", "weight": 2}]}',
            restore=keep, raw=raw_value,
        )
        return None

    raw_criteria = value.get("criteria")
    if raw_criteria is None:
        _demote(
            into, issues, path=path, strict=strict,
            reason="has no 'criteria'",
            hint='e.g. {"criteria": [{"name": "accuracy"}]}',
            restore=keep, raw=raw_value,
        )
        return None
    if not isinstance(raw_criteria, list):
        issues.append(
            ValidationIssue(
                f"{path}.criteria",
                f"must be a list, got {type(raw_criteria).__name__}",
            )
        )
        return None

    criteria: list[Criterion] = []
    for index, raw in enumerate(raw_criteria):
        item_path = f"{path}.criteria[{index}]"
        if isinstance(raw, str):
            raw = {"name": raw}  # bare name shorthand, same reasoning as tools
        if not isinstance(raw, dict):
            issues.append(
                ValidationIssue(
                    item_path,
                    f"must be an object or a name, got {type(raw).__name__}",
                )
            )
            continue
        try:
            criteria.append(
                Criterion(
                    name=raw.get("name"),
                    description=raw.get("description"),
                    weight=raw.get("weight", 1.0),
                )
            )
        except SchemaError as exc:
            issues.append(ValidationIssue(item_path, str(exc)))

    if not criteria:
        return None
    try:
        return Rubric(
            criteria=tuple(criteria),
            scale=value.get("scale"),
            aggregate=value.get("aggregate", "weighted_mean"),
        )
    except SchemaError as exc:
        issues.append(ValidationIssue(path, str(exc)))
        return None


def _parse_acceptable(
    value: Any, where: str, issues: list[ValidationIssue]
) -> tuple[Any, ...]:
    if value is None:
        return ()
    path = f"{where}.acceptable"
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            try:
                value = json.loads(text)
            except json.JSONDecodeError:
                # A string that merely starts with '[' is more likely a real
                # answer than broken JSON, so keep it as a single answer.
                return (value,)
        else:
            # One acceptable answer written as a bare string. Wrapping it is
            # unambiguous, and iterating a string into characters would be a
            # catastrophic misread.
            return (value,)
    if isinstance(value, (list, tuple, set)):
        return tuple(value)
    if isinstance(value, dict):
        issues.append(
            ValidationIssue(
                path,
                "must be a list of answers, got an object",
                hint="an object is one structured answer, so put it in a "
                'list: "acceptable": [{...}]',
            )
        )
        return ()
    return (value,)


def _parse_expected_schema(
    data: dict[str, Any],
    where: str,
    issues: list[ValidationIssue],
    into: ParsedCases,
    strict: bool,
) -> dict[str, Any] | None:
    raw_value = data.pop("expected_schema", None)
    if raw_value is None:
        return None
    path = f"{where}.expected_schema"
    keep = (data, "expected_schema")
    value = _as_structure(
        raw_value, path=path, into=into, issues=issues, strict=strict,
        expect="a JSON Schema object", restore=keep, raw=raw_value,
    )
    if value is None:
        return None
    if not isinstance(value, dict):
        _demote(
            into, issues, path=path, strict=strict,
            reason=f"must be a JSON Schema object, got {type(value).__name__}",
            hint='e.g. {"type": "object", "required": ["total"]}',
            restore=keep, raw=raw_value,
        )
        return None
    return value


# ------------------------------------------------------------------- near misses


def _flag_near_miss_keys(
    data: dict[str, Any], where: str, issues: list[ValidationIssue]
) -> None:
    """In strict mode, a key one edit from a real field is a mistake, not metadata.

    Only in strict mode. Under version 1 an unrecognised key has always been
    kept as metadata, and enough real datasets have columns like ``labels`` or
    ``inputs`` that turning those into errors would break files this package
    promises to keep reading.
    """
    for key in list(data):
        if key in CANONICAL_FIELDS:
            continue
        close = difflib.get_close_matches(
            key, CANONICAL_FIELDS, n=1, cutoff=SUGGESTION_CUTOFF
        )
        if close:
            issues.append(
                ValidationIssue(
                    f"{where}.{key}",
                    f"is not a field, and would be kept as metadata",
                    hint=f"did you mean '{close[0]}'? If '{key}' really is your "
                    "own metadata, nest it under \"metadata\" to silence this.",
                )
            )


def _did_you_mean(value: Any, options: list[str]) -> str | None:
    if not isinstance(value, str):
        return None
    close = difflib.get_close_matches(value.lower(), options, n=1, cutoff=0.6)
    return f"did you mean {close[0]!r}?" if close else None


def check_version(declared: Any, *, source: str) -> int:
    """Validate a declared ``evallint_schema`` value.

    Raises:
        SchemaValidationError: the value is not a version this build reads.
            Refusing by name is the point: a version-3 file misread under
            version-2 rules would produce a confident report about data that
            was parsed wrong.
    """
    if isinstance(declared, bool) or not isinstance(declared, int):
        raise SchemaValidationError(
            [
                ValidationIssue(
                    "evallint_schema",
                    f"must be an integer version, got "
                    f"{type(declared).__name__} ({declared!r})",
                    hint="supported: "
                    + ", ".join(str(v) for v in SUPPORTED_SCHEMA_VERSIONS),
                )
            ],
            source=source,
        )
    if declared not in SUPPORTED_SCHEMA_VERSIONS:
        from . import __version__

        raise SchemaValidationError(
            [
                ValidationIssue(
                    "evallint_schema",
                    f"this file declares schema version {declared}, which "
                    f"evallint {__version__} cannot read",
                    hint="supported versions: "
                    + ", ".join(str(v) for v in SUPPORTED_SCHEMA_VERSIONS)
                    + ". Upgrade evallint, or lower the declared version if "
                    "the file does not actually use newer features.",
                )
            ],
            source=source,
        )
    return declared
