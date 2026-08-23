"""The internal representation of an eval set.

Everything upstream (io.py) converts a file into these types; everything
downstream (the checks) reads only these types. That boundary is the whole
point of this module: a check should never know or care whether the data came
from CSV or JSONL, or whether it was written in the four-field format or the
full chat format.

Validation lives here rather than in io.py so that a hand-constructed EvalCase
is held to exactly the same rules as one loaded from disk. File-aware errors
with JSON paths and multi-error collection live in validation.py, which wraps
these rules rather than duplicating them.

## The four-field minimum is still the whole schema for most users

    {"id": "1", "input": "2+2?", "expected": "4", "label": "arith"}

Only ``id`` and ``input`` are required, and nothing below changes that. Every
field added in schema version 2 is optional and defaults to empty, so a v1
file, a v1 hand-built EvalCase, and a v1 call to ``EvalCase(id=..., input=...)``
all behave exactly as they did before.

## Why ``input`` stayed a required string

Every check in this package reads ``case.input`` as text — `len(case.input)` in
imbalance, `normalise(case.input)` in five leakage detectors, embedding in
redundancy. Making it optional or polymorphic would mean editing all of them,
and a chat case would then be invisible to every check that was not updated.

So for a multi-turn case ``input`` is DERIVED: the conversation flattened to
text by :func:`derive_input`. Every existing check then works on chat data
unchanged — cross-turn leakage, redundancy between conversations, length
statistics — with no edits to any check.

Derived data must never masquerade as user data, so ``input_is_derived`` records
it, and the flattening rule is documented on :func:`derive_input` and tested for
round-trip stability. A single user turn flattens to its own content verbatim,
which is what makes ``{"input": "2+2?"}`` and its one-message chat equivalent
produce byte-identical text.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

__all__ = [
    "SCHEMA_VERSION",
    "SUPPORTED_SCHEMA_VERSIONS",
    "Criterion",
    "EvalCase",
    "EvalSet",
    "Message",
    "Role",
    "Rubric",
    "SchemaError",
    "ToolCall",
    "ToolSpec",
    "derive_input",
]

#: The newest schema version this build understands. Version 1 is the original
#: four-field format and is never deprecated.
SCHEMA_VERSION = 2

#: Every version this build can read. A file declaring anything else is refused
#: by name rather than parsed on a guess.
SUPPORTED_SCHEMA_VERSIONS = (1, 2)


class SchemaError(ValueError):
    """A case or set violates the schema.

    Subclasses ValueError because that is what a caller would naturally catch
    for "you gave me bad data".
    """


class Role(StrEnum):
    """Who produced a message.

    A closed set on purpose. An open string field turns ``"assistent"`` into a
    turn that no consumer recognises and nothing rejects — a silent hole in the
    conversation, which is the failure mode this package exists to report.
    """

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A call to a tool: the name, the arguments, and the id that pairs it with
    its result.

    Used in two places with two different meanings, which the field name makes
    explicit at the point of use:

      * ``Message.tool_calls`` — a call the assistant actually made, i.e. part
        of the conversation being replayed.
      * ``EvalCase.expected_tool_calls`` — a call the assistant SHOULD make,
        i.e. ground truth.
    """

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    id: str | None = None

    def __post_init__(self) -> None:
        _require_nonempty_str(self.name, "tool call name")
        if not isinstance(self.arguments, dict):
            raise SchemaError(
                f"tool call '{self.name}': arguments must be an object, got "
                f"{type(self.arguments).__name__}"
            )
        if self.id is not None and not isinstance(self.id, str):
            raise SchemaError(
                f"tool call '{self.name}': id must be a string, got "
                f"{type(self.id).__name__}"
            )

    def signature(self) -> str:
        """``name(arg=value, ...)`` with keys sorted, for text comparison.

        Sorted so that two calls differing only in key order compare equal.
        Used by :func:`derive_input` and available to anything that needs a
        stable textual form of a call.
        """
        args = ", ".join(
            f"{k}={json.dumps(self.arguments[k], sort_keys=True, default=str)}"
            for k in sorted(self.arguments)
        )
        return f"{self.name}({args})"

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.name}
        if self.arguments:
            out["arguments"] = dict(self.arguments)
        if self.id is not None:
            out["id"] = self.id
        return out


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """A tool the model was allowed to use on this case.

    ``parameters`` is a JSON Schema object and is stored, not interpreted. This
    package does not validate arguments against it: doing so would mean
    shipping a JSON Schema implementation, and being wrong about a spec would
    produce confident findings about correct data.
    """

    name: str
    description: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_nonempty_str(self.name, "tool name")
        if self.description is not None and not isinstance(self.description, str):
            raise SchemaError(
                f"tool '{self.name}': description must be a string, got "
                f"{type(self.description).__name__}"
            )
        if not isinstance(self.parameters, dict):
            raise SchemaError(
                f"tool '{self.name}': parameters must be a JSON Schema object, "
                f"got {type(self.parameters).__name__}"
            )

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.name}
        if self.description is not None:
            out["description"] = self.description
        if self.parameters:
            out["parameters"] = dict(self.parameters)
        return out


@dataclass(frozen=True, slots=True)
class Message:
    """One turn of a conversation.

    Attributes:
        role: Who spoke. See :class:`Role`.
        content: The text of the turn. May be empty ONLY when the turn carries
            tool calls instead, which is how an assistant turn that only calls
            a tool is represented.
        tool_calls: Calls the assistant made in this turn.
        tool_call_id: On a ``tool`` turn, which call this is the result of.
        name: The tool's name on a ``tool`` turn, or a speaker name where a
            format supplies one.
    """

    role: Role
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.role, Role):
            if isinstance(self.role, str):
                try:
                    object.__setattr__(self, "role", Role(self.role.strip().lower()))
                except ValueError as exc:
                    raise SchemaError(
                        f"{self.role!r} is not a valid role; allowed: "
                        + ", ".join(r.value for r in Role)
                    ) from exc
            else:
                raise SchemaError(
                    f"role must be a string, got {type(self.role).__name__}"
                )

        if not isinstance(self.content, str):
            raise SchemaError(
                f"{self.role.value} message: content must be a string, got "
                f"{type(self.content).__name__}"
            )

        object.__setattr__(self, "tool_calls", tuple(self.tool_calls))
        for position, call in enumerate(self.tool_calls):
            if not isinstance(call, ToolCall):
                raise SchemaError(
                    f"{self.role.value} message: tool_calls[{position}] is a "
                    f"{type(call).__name__}, expected ToolCall"
                )

        for name, value in (("tool_call_id", self.tool_call_id), ("name", self.name)):
            if value is not None and not isinstance(value, str):
                raise SchemaError(
                    f"{self.role.value} message: {name} must be a string, got "
                    f"{type(value).__name__}"
                )

        if not self.content.strip() and not self.tool_calls:
            # An empty turn contributes nothing to the flattened input, so it
            # would silently shrink the case rather than fail. Tool-call-only
            # assistant turns are the one legitimate empty-content case.
            raise SchemaError(
                f"{self.role.value} message has neither content nor tool_calls, "
                "so it carries no information"
            )

        if self.tool_calls and self.role is not Role.ASSISTANT:
            raise SchemaError(
                f"only an assistant message can make tool calls, but a "
                f"{self.role.value} message has {len(self.tool_calls)}"
            )

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"role": self.role.value}
        if self.content:
            out["content"] = self.content
        if self.tool_calls:
            out["tool_calls"] = [c.as_dict() for c in self.tool_calls]
        if self.tool_call_id is not None:
            out["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            out["name"] = self.name
        return out


@dataclass(frozen=True, slots=True)
class Criterion:
    """One line of a rubric.

    ``weight`` is relative, not a share: weights of 3 and 1 mean the same thing
    as 0.75 and 0.25. The raw number the user wrote is stored unchanged, and
    :meth:`Rubric.weight_shares` derives the shares. Rewriting a user's numbers
    on load would make the file and the report disagree.
    """

    name: str
    description: str | None = None
    weight: float = 1.0

    def __post_init__(self) -> None:
        _require_nonempty_str(self.name, "criterion name")
        if self.description is not None and not isinstance(self.description, str):
            raise SchemaError(
                f"criterion '{self.name}': description must be a string, got "
                f"{type(self.description).__name__}"
            )
        if isinstance(self.weight, bool) or not isinstance(self.weight, (int, float)):
            raise SchemaError(
                f"criterion '{self.name}': weight must be a number, got "
                f"{type(self.weight).__name__}"
            )
        if self.weight < 0:
            raise SchemaError(
                f"criterion '{self.name}': weight must not be negative, got "
                f"{self.weight}"
            )
        object.__setattr__(self, "weight", float(self.weight))

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.name}
        if self.description is not None:
            out["description"] = self.description
        if self.weight != 1.0:
            out["weight"] = self.weight
        return out


@dataclass(frozen=True, slots=True)
class Rubric:
    """Criteria a grader should apply to this case.

    A rubric is an INPUT TO A JUDGE, not ground truth. Storing one here does not
    make the resulting scores reliable, and nothing in this package treats it as
    if it did — a rubric with a judge behind it still needs
    :class:`~evallint.checks.EvaluatorReliability` to say whether the grading is
    reproducible. That caveat is in the docs and in the CLI note; it is
    mentioned here because this is where someone reads the field and assumes
    otherwise.
    """

    criteria: tuple[Criterion, ...]
    scale: tuple[float, float] | None = None
    aggregate: str = "weighted_mean"

    def __post_init__(self) -> None:
        object.__setattr__(self, "criteria", tuple(self.criteria))
        if not self.criteria:
            raise SchemaError("a rubric must have at least one criterion")
        for position, criterion in enumerate(self.criteria):
            if not isinstance(criterion, Criterion):
                raise SchemaError(
                    f"criteria[{position}] is a {type(criterion).__name__}, "
                    "expected Criterion"
                )
        names = [c.name for c in self.criteria]
        repeated = sorted({n for n in names if names.count(n) > 1})
        if repeated:
            # Two criteria with one name cannot be reported separately, and
            # their weights would silently add.
            raise SchemaError(
                "criterion names must be unique within a rubric, repeated: "
                + ", ".join(repeated)
            )

        if self.scale is not None:
            scale = tuple(self.scale)
            if len(scale) != 2:
                raise SchemaError(
                    f"scale must be [low, high], got {len(scale)} value(s)"
                )
            low, high = scale
            for bound in (low, high):
                if isinstance(bound, bool) or not isinstance(bound, (int, float)):
                    raise SchemaError(
                        f"scale bounds must be numbers, got "
                        f"{type(bound).__name__}"
                    )
            if not low < high:
                raise SchemaError(f"scale low must be below high, got [{low}, {high}]")
            object.__setattr__(self, "scale", (float(low), float(high)))

        if not isinstance(self.aggregate, str) or not self.aggregate.strip():
            raise SchemaError("aggregate must be a non-empty string")

        if sum(c.weight for c in self.criteria) <= 0:
            # Every weight zero means every criterion is worth nothing, so no
            # aggregate score is definable. Better to refuse than to silently
            # fall back to an unweighted mean the user did not ask for.
            raise SchemaError(
                "rubric weights sum to zero, so no criterion can contribute to "
                "a score; give at least one criterion a positive weight"
            )

    def weight_shares(self) -> dict[str, float]:
        """Each criterion's share of the total weight. Derived, not stored."""
        total = sum(c.weight for c in self.criteria)
        return {c.name: c.weight / total for c in self.criteria}

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"criteria": [c.as_dict() for c in self.criteria]}
        if self.scale is not None:
            out["scale"] = list(self.scale)
        if self.aggregate != "weighted_mean":
            out["aggregate"] = self.aggregate
        return out


@dataclass(frozen=True, slots=True)
class EvalCase:
    """One test case in an eval set.

    Only ``id`` and ``input`` are required. Everything else is optional because
    real eval sets in the wild are ragged, and a check that needs a field it
    does not have should say so rather than refuse to load the file.

    Attributes:
        id: Unique identifier within the set. Reports refer to cases by this.
        input: The prompt given to the model, as text. For a chat case this is
            derived from ``messages`` — see the module docstring.
        expected: Ground truth. Deliberately untyped: it can be a string, a
            bool, a number, a list, or (schema 2) a dict for a structured
            output. None of the checks interpret its type beyond looking for
            list-valued alternatives.
        label: The case's category / class. This is what the imbalance check
            counts. Non-string scalars are normalised to str.
        metadata: Any fields the source file had that are not canonical. Kept
            rather than dropped so nothing is silently lost on load.
        messages: The conversation, when the case is multi-turn. Empty for a
            single-prompt case.
        system: The system prompt. Held separately from ``input`` on purpose —
            see :func:`derive_input`.
        acceptable: Additional answers that should also count as correct.
            ``expected`` remains the canonical one; use
            :meth:`acceptable_answers` for the full set.
        expected_schema: A JSON Schema the output should satisfy. Stored, not
            enforced.
        rubric: Criteria for a grader. See :class:`Rubric`.
        tools: Tools the model was allowed to call.
        expected_tool_calls: Calls the model SHOULD have made.
        input_is_derived: True when ``input`` was flattened from ``messages``
            rather than written by the user.
    """

    id: str
    input: str
    expected: Any = None
    label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    messages: tuple[Message, ...] = ()
    system: str | None = None
    acceptable: tuple[Any, ...] = ()
    expected_schema: dict[str, Any] | None = None
    rubric: Rubric | None = None
    tools: tuple[ToolSpec, ...] = ()
    expected_tool_calls: tuple[ToolCall, ...] = ()
    input_is_derived: bool = False

    def __post_init__(self) -> None:
        _require_nonempty_str(self.id, "id")
        _require_nonempty_str(self.input, "input")

        if not isinstance(self.metadata, dict):
            raise SchemaError(
                f"metadata must be a dict, got {type(self.metadata).__name__}"
            )

        if self.label is not None:
            # A label is a category name, so it must be a scalar we can group
            # and print. We normalise 1 and "1" to the same class on purpose:
            # a CSV writes labels as text and a JSON file may write them as
            # numbers, and those should not count as two different classes.
            if isinstance(self.label, (str, int, float, bool)):
                normalised = str(self.label).strip()
                if not normalised:
                    raise SchemaError(
                        f"case '{self.id}': label must be non-empty when present"
                    )
                # frozen dataclasses block plain assignment, so write through
                # object.__setattr__ — the standard idiom for normalising a
                # field in __post_init__.
                object.__setattr__(self, "label", normalised)
            else:
                raise SchemaError(
                    f"case '{self.id}': label must be a string or number, "
                    f"got {type(self.label).__name__}"
                )

        object.__setattr__(self, "messages", tuple(self.messages))
        for position, message in enumerate(self.messages):
            if not isinstance(message, Message):
                raise SchemaError(
                    f"case '{self.id}': messages[{position}] is a "
                    f"{type(message).__name__}, expected Message"
                )

        if self.system is not None:
            if not isinstance(self.system, str):
                raise SchemaError(
                    f"case '{self.id}': system must be a string, got "
                    f"{type(self.system).__name__}"
                )
            if not self.system.strip():
                raise SchemaError(
                    f"case '{self.id}': system must be non-empty when present"
                )

        object.__setattr__(self, "acceptable", tuple(self.acceptable))

        if self.expected_schema is not None and not isinstance(
            self.expected_schema, dict
        ):
            raise SchemaError(
                f"case '{self.id}': expected_schema must be a JSON Schema "
                f"object, got {type(self.expected_schema).__name__}"
            )

        if self.rubric is not None and not isinstance(self.rubric, Rubric):
            raise SchemaError(
                f"case '{self.id}': rubric must be a Rubric, got "
                f"{type(self.rubric).__name__}"
            )

        object.__setattr__(self, "tools", tuple(self.tools))
        for position, spec in enumerate(self.tools):
            if not isinstance(spec, ToolSpec):
                raise SchemaError(
                    f"case '{self.id}': tools[{position}] is a "
                    f"{type(spec).__name__}, expected ToolSpec"
                )
        tool_names = [t.name for t in self.tools]
        repeated = sorted({n for n in tool_names if tool_names.count(n) > 1})
        if repeated:
            raise SchemaError(
                f"case '{self.id}': tool names must be unique, repeated: "
                + ", ".join(repeated)
            )

        object.__setattr__(self, "expected_tool_calls", tuple(self.expected_tool_calls))
        for position, call in enumerate(self.expected_tool_calls):
            if not isinstance(call, ToolCall):
                raise SchemaError(
                    f"case '{self.id}': expected_tool_calls[{position}] is a "
                    f"{type(call).__name__}, expected ToolCall"
                )

        # A call the model is required to make, to a tool it was never given,
        # is unsatisfiable: the case can only ever fail. Worth an error rather
        # than a finding, because no scoring run could produce a useful number.
        if self.tools and self.expected_tool_calls:
            available = set(tool_names)
            unavailable = sorted(
                {c.name for c in self.expected_tool_calls if c.name not in available}
            )
            if unavailable:
                raise SchemaError(
                    f"case '{self.id}': expected_tool_calls name tool(s) not in "
                    f"'tools': {', '.join(unavailable)}. Available: "
                    f"{', '.join(sorted(available))}"
                )

    def as_dict(self) -> dict[str, Any]:
        """Serialise back to the on-disk shape, omitting everything unset.

        ``input`` is written only when the user wrote it. A derived input is
        recomputed from ``messages`` on load, so writing it would store the
        same text twice and let the two drift apart -- and a stored copy is
        indistinguishable from something the user typed.
        """
        out: dict[str, Any] = {"id": self.id}
        if not self.input_is_derived:
            out["input"] = self.input
        if self.messages:
            out["messages"] = [m.as_dict() for m in self.messages]
        if self.system is not None:
            out["system"] = self.system
        if self.expected is not None:
            out["expected"] = self.expected
        if self.acceptable:
            out["acceptable"] = list(self.acceptable)
        if self.expected_schema is not None:
            out["expected_schema"] = dict(self.expected_schema)
        if self.rubric is not None:
            out["rubric"] = self.rubric.as_dict()
        if self.tools:
            out["tools"] = [t.as_dict() for t in self.tools]
        if self.expected_tool_calls:
            out["expected_tool_calls"] = [c.as_dict() for c in self.expected_tool_calls]
        if self.label is not None:
            out["label"] = self.label
        out.update(self.metadata)
        return out

    @property
    def is_chat(self) -> bool:
        """True when this case carries a conversation rather than one prompt."""
        return bool(self.messages)

    def acceptable_answers(self) -> tuple[Any, ...]:
        """Every answer that should count as correct, canonical one first.

        ``expected`` leads because it is the answer a report quotes. Duplicates
        are dropped where the values are hashable and kept otherwise — a dict
        or list in ``acceptable`` is unhashable, and dropping it to keep the
        de-duplication tidy would discard a valid answer.
        """
        answers: list[Any] = []
        seen: set[Any] = set()
        for value in (self.expected, *self.acceptable):
            if value is None:
                continue
            try:
                if value in seen:
                    continue
                seen.add(value)
            except TypeError:
                pass  # unhashable: keep it, de-duplication is the lesser goal
            answers.append(value)
        return tuple(answers)


@dataclass(frozen=True, slots=True)
class EvalSet:
    """A collection of eval cases, plus where it came from.

    ``cases`` is stored as a tuple: an eval set is a fixed snapshot being
    audited, and no check should ever mutate the thing it is measuring.

    ``schema_version`` is what the FILE declared, defaulting to 1. It is
    recorded rather than inferred from the fields present, so that a v1 file
    is never read under v2 rules on a guess.
    """

    cases: tuple[EvalCase, ...]
    source: str | None = None
    schema_version: int = 1
    #: Things the loader decided that the user needs to see but that are not
    #: errors -- above all a field demoted to metadata. Carried on the set
    #: because that is where `source` already lives: both describe the load
    #: rather than the data.
    load_notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.cases, (str, bytes)) or not isinstance(
            self.cases, Sequence
        ):
            raise SchemaError("cases must be a sequence of EvalCase objects")

        object.__setattr__(self, "cases", tuple(self.cases))

        if not self.cases:
            raise SchemaError("an eval set must contain at least one case")

        for position, case in enumerate(self.cases, start=1):
            if not isinstance(case, EvalCase):
                raise SchemaError(
                    f"case {position} is a {type(case).__name__}, expected EvalCase"
                )

        object.__setattr__(self, "load_notes", tuple(self.load_notes))

        if self.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
            raise SchemaError(
                f"unsupported schema version {self.schema_version!r}; this "
                f"build reads "
                + ", ".join(str(v) for v in SUPPORTED_SCHEMA_VERSIONS)
            )

        seen: set[str] = set()
        duplicates: list[str] = []
        for case in self.cases:
            if case.id in seen and case.id not in duplicates:
                duplicates.append(case.id)
            seen.add(case.id)
        if duplicates:
            # Duplicate ids are a load-time error, not a finding. The duplicate
            # CHECK is about semantically similar inputs; this is just two rows
            # claiming the same name, which would make any report ambiguous.
            raise SchemaError(
                "case ids must be unique, repeated: " + ", ".join(sorted(duplicates))
            )

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self) -> Iterator[EvalCase]:
        return iter(self.cases)

    @property
    def chat_cases(self) -> int:
        """How many cases carry a conversation. 0 for any v1 file."""
        return sum(1 for case in self.cases if case.is_chat)


def derive_input(messages: Sequence[Message]) -> str:
    """Flatten a conversation into the single text string ``input`` requires.

    THE RULES, in full, because every check downstream reads this string:

    1. A lone user turn with no tool calls flattens to its own content,
       with no role prefix. This is what makes the chat form of a simple case
       produce byte-identical text to the four-field form, so migrating a v1
       file cannot change a single finding.
    2. Otherwise each turn becomes ``role: content``, joined by newlines, in
       order. A turn with a ``name`` becomes ``role(name): content``.
    3. Tool calls become ``assistant: [tool_call] name(arg=value)`` on their own
       line, with argument keys sorted so key order cannot change the text.
    4. The system prompt is NOT included. See below.

    Why the system prompt is excluded, which is the least obvious rule here: a
    system prompt is usually identical across every case in a set. Prepending it
    to each ``input`` would give every case a long shared prefix, which inflates
    pairwise cosine similarity and token overlap. The redundancy check would
    then report near-duplicates throughout, the effective-size estimate would
    collapse toward one cluster, and both would be artefacts of the flattening
    rather than properties of the data. ``system`` therefore stays its own
    field, and a check that wants it must ask for it.
    """
    turns = list(messages)
    if not turns:
        return ""

    if (
        len(turns) == 1
        and turns[0].role is Role.USER
        and not turns[0].tool_calls
        and turns[0].name is None
    ):
        return turns[0].content

    lines: list[str] = []
    for message in turns:
        speaker = message.role.value
        if message.name:
            speaker = f"{speaker}({message.name})"
        if message.content:
            lines.append(f"{speaker}: {message.content}")
        for call in message.tool_calls:
            lines.append(f"{speaker}: [tool_call] {call.signature()}")
    return "\n".join(lines)


def _require_nonempty_str(value: Any, name: str) -> None:
    if not isinstance(value, str):
        raise SchemaError(f"{name} must be a string, got {type(value).__name__}")
    if not value.strip():
        raise SchemaError(f"{name} must be a non-empty string")
