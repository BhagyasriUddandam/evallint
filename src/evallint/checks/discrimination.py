"""Check 1 — discrimination failure.

The failure this catches: an eval reports 200 cases, but on 140 of them a
strong model and a weak model get exactly the same verdict. Those 140 cases
cannot tell the two apart, so they contribute no evidence about which model is
better. The eval is really 60 cases wearing a 200-case costume.

Run each case against two or more models of KNOWN differing capability and
compare the verdicts. "All the same verdict" splits into two situations with
different remedies, so they are reported separately:

  ceiling  every model passes  -> too easy; adds score, not signal
  floor    every model fails   -> too hard, OR the reference answer is wrong,
                                  OR the grader is broken. Check the case
                                  before blaming the models.

And one pattern that is not a discrimination failure at all but falls out of
the same data and is worth more than either:

  inverted  a weaker model passed where a stronger one failed. These cases do
            separate the models, but in the wrong direction. In practice that
            usually means a bad reference answer or a grader artefact.

The scorer is injected. That is the single most important design choice in the
tool: evallint never talks to a model provider, so it works with anyone's
stack and the tests need no API key.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ..schema import EvalCase, EvalSet
from .base import Check, CheckResult, Finding, Severity

__all__ = ["DiscriminationCheck", "Scorer", "ScorerError"]

Scorer = Callable[[EvalCase, Any], bool | float]

LIMITATIONS = (
    "This is the only check that runs models. It calls the scorer once per "
    "case per model, so on a real API it costs time and money proportional to "
    "n_cases x n_models.",
    "The verdict depends entirely on the model pair you chose. Two models that "
    "are closer in capability than you assumed will make a perfectly good eval "
    "look non-discriminating, and the check cannot tell the difference.",
    "A non-discriminating case is not automatically a bad case. Easy cases "
    "every model passes may be deliberate regression guards. This reports how "
    "much of your eval is doing discriminating work; it does not tell you what "
    "to delete.",
    "Continuous scores are reduced to pass/fail at the threshold, so a case "
    "where the strong model scores 0.95 and the weak one 0.55 is recorded as "
    "non-discriminating. Use a binary scorer, or move the threshold, if that "
    "distinction matters to you.",
    "Model outputs are usually sampled, so a case can flip between runs. Note "
    "that temperature=0 is NOT always available: current frontier models "
    "including Claude Opus 5 and Sonnet 5 reject the temperature parameter "
    "outright, so determinism may not be an option for your model pair.",
    "With max_workers above 1 the scorer is called from multiple threads. "
    "evallint cannot verify that your scorer is thread-safe, and an unsafe one "
    "corrupts results quietly rather than crashing. If a run at max_workers>1 "
    "disagrees with a serial run on the same data, suspect the scorer before "
    "the eval set.",
    "THIS IS A SINGLE-RUN MEASUREMENT AND MAY NOT REPRODUCE. Every number "
    "above comes from scoring each case exactly once. On the author's own "
    "20-case example set, repeating an identical run three times changed the "
    "verdict for 8 of 20 cases (40%), and only 1 of the inversions found in a "
    "single run survived all three repeats. Before reporting any figure here "
    "-- especially a specific inverted or floor case -- run the same "
    "configuration at least three times and keep only the cases whose verdict "
    "is identical every time. A case that flips is sampling noise, not a "
    "finding.",
)


class ScorerError(RuntimeError):
    """The user-supplied scorer failed or returned something unusable."""


class DiscriminationCheck(Check):
    """Reports how many cases actually separate models of differing capability."""

    name = "discrimination"
    description = "Cases that give the same verdict for strong and weak models."

    def __init__(
        self,
        scorer: Scorer,
        models: Sequence[Any],
        *,
        pass_threshold: float = 0.5,
        max_non_discriminating_share: float = 0.5,
        repeats: int = 1,
        max_workers: int = 1,
    ) -> None:
        """
        Args:
            scorer: ``scorer(case, model) -> bool | float``. Returns whether
                the model got this case right, or a score compared against
                ``pass_threshold``. evallint never calls a provider itself.
            models: Two or more model identifiers ordered WEAKEST FIRST. The
                order is what makes inversion detectable, so getting it
                backwards inverts that finding — see the limitations.
            pass_threshold: A float score >= this counts as a pass. Ignored
                for bool scorers.
            max_non_discriminating_share: Warn above this share of MEASURED
                cases (unstable cases are excluded from the denominator).
            repeats: How many times to score each (case, model). A case whose
                verdict is not unanimous across repeats is reported as
                UNSTABLE and excluded from the ceiling/floor/inverted counts,
                because a verdict that changes between identical runs is
                sampling noise rather than a property of the eval set.

                Default 1 preserves single-run behaviour and cost, but 1 also
                cannot detect instability at all. On the bundled 20-case
                example, 3 repeats found 40% of verdicts non-reproducible, so
                a single run there was measuring almost nothing reliably.
                Cost multiplies by ``repeats``.
            max_workers: How many scorer calls to run concurrently. Scoring is
                almost always I/O-bound (an API call per case per model), and
                running it serially is the difference between minutes and hours:
                1000 cases x 2 models x 3 repeats at 2s per call is 3.3 hours
                serial, roughly 15 minutes at 16 workers.

                Default 1 runs serially and takes a code path with no threads in
                it at all. That is deliberate: YOUR SCORER MUST BE THREAD-SAFE
                to raise this above 1. evallint cannot check that for you — the
                scorer is your code, and a client object, a shared session, a
                file handle or a non-locking cache inside it will corrupt
                quietly rather than crash. Opt in once you have checked.
        """
        if len(models) < 2:
            raise ValueError(
                "discrimination needs at least two models of differing "
                f"capability, got {len(models)}"
            )
        labels = [str(m) for m in models]
        if len(set(labels)) != len(labels):
            raise ValueError(f"model names must be unique, got {labels}")
        if not 0 < max_non_discriminating_share <= 1:
            raise ValueError("max_non_discriminating_share must be between 0 and 1")
        if repeats < 1:
            raise ValueError(f"repeats must be at least 1, got {repeats}")
        if max_workers < 1:
            raise ValueError(f"max_workers must be at least 1, got {max_workers}")
        self.scorer = scorer
        self.models = tuple(models)
        self.model_names = tuple(labels)
        self.pass_threshold = pass_threshold
        self.max_non_discriminating_share = max_non_discriminating_share
        self.repeats = repeats
        self.max_workers = max_workers

    def run(self, eval_set: EvalSet) -> CheckResult:
        cases = list(eval_set)
        # runs[case.id][model_index] -> tuple of `repeats` verdicts
        runs = self._score_all(cases)

        # A case is only measurable if every model gave the same answer on
        # every repeat. Anything else is a verdict that changes between
        # identical runs, which is noise — folding it into ceiling/floor/
        # inverted would report sampling variance as a property of the data.
        stable: dict[str, tuple[bool, ...]] = {}
        unstable: list[str] = []
        for case in cases:
            per_model = runs[case.id]
            if all(len(set(r)) == 1 for r in per_model):
                stable[case.id] = tuple(r[0] for r in per_model)
            else:
                unstable.append(case.id)

        ceiling = [c.id for c in cases if c.id in stable and all(stable[c.id])]
        floor = [c.id for c in cases if c.id in stable and not any(stable[c.id])]
        inverted = [
            c.id for c in cases if c.id in stable and _is_inverted(stable[c.id])
        ]
        non_discriminating = len(ceiling) + len(floor)
        n_cases = len(cases)
        n_measured = len(stable)
        # Denominator is MEASURED cases, not all cases: an unstable case is
        # unknown, not discriminating, and counting it as either would flatter
        # or slander the eval set. With repeats=1 nothing can be unstable, so
        # this is identical to the old behaviour.
        share = non_discriminating / n_measured if n_measured else 0.0

        stats = {
            "n_cases": n_cases,
            "models": list(self.model_names),
            "repeats": self.repeats,
            "max_workers": self.max_workers,
            "n_scorer_calls": n_cases * len(self.models) * self.repeats,
            "pass_threshold": self.pass_threshold,
            "n_measured": n_measured,
            "n_unstable": len(unstable),
            "n_discriminating": n_measured - non_discriminating,
            "n_non_discriminating": non_discriminating,
            "non_discriminating_share": share,
            "effective_n_cases": n_measured - non_discriminating,
            "n_ceiling": len(ceiling),
            "n_floor": len(floor),
            "n_inverted": len(inverted),
            # Mean pass rate across every call, so it stays meaningful when
            # verdicts disagree between repeats.
            "per_model_pass_rate": {
                name: sum(sum(runs[c.id][i]) for c in cases)
                / (n_cases * self.repeats)
                for i, name in enumerate(self.model_names)
            },
            "ceiling_case_ids": ceiling,
            "floor_case_ids": floor,
            "inverted_case_ids": inverted,
            "unstable_case_ids": unstable,
        }

        findings: list[Finding] = []
        if unstable:
            # First, and a WARNING: if verdicts are not reproducible then every
            # other number here is provisional, so this must be read before them.
            findings.append(
                Finding(
                    f"{len(unstable)} of {n_cases} cases gave a DIFFERENT verdict "
                    f"across {self.repeats} identical runs, so they cannot be "
                    f"classified and are excluded from the counts below. A verdict "
                    f"that changes between runs is sampling noise, not a property "
                    f"of the eval set — the remaining figures describe only the "
                    f"{n_measured} reproducible cases",
                    case_ids=tuple(unstable),
                )
            )
        if share > self.max_non_discriminating_share:
            # Say "measured cases" only when some were actually excluded. With
            # nothing excluded the qualifier is noise; with cases excluded,
            # plain "of cases" would imply a denominator that isn't the one used.
            scope = "measured cases" if unstable else "cases"
            findings.append(
                Finding(
                    f"{share:.0%} of {scope} ({non_discriminating}/{n_measured}) "
                    f"give every model the same verdict, so they carry no "
                    f"evidence about which model is better. This eval "
                    f"discriminates like a {n_measured - non_discriminating}-case "
                    f"set"
                )
            )
        if ceiling:
            # INFO, not WARNING. Every eval has easy cases, and some are
            # deliberate regression guards. Warning about them would flatly
            # contradict this check's own third limitation. Floor and inverted
            # cases DO get a warning, because they suggest something is broken
            # rather than merely easy.
            findings.append(
                Finding(
                    f"every model passes {_n_cases(len(ceiling))}, including "
                    f"'{self.model_names[0]}' — too easy to separate anything",
                    severity=Severity.INFO,
                    case_ids=tuple(ceiling),
                )
            )
        if floor:
            findings.append(
                Finding(
                    f"every model fails {_n_cases(len(floor))}, including "
                    f"'{self.model_names[-1]}'. Check the reference answer and "
                    f"the grader before assuming difficulty",
                    case_ids=tuple(floor),
                )
            )
        if inverted:
            findings.append(
                Finding(
                    f"in {_n_cases(len(inverted))} a weaker model passed where a "
                    f"stronger one failed. That ordering is suspicious: it "
                    f"usually means a wrong reference answer or a grader "
                    f"artefact rather than a real capability gap",
                    case_ids=tuple(inverted),
                )
            )
        findings.extend(self._sanity_findings(stats, non_discriminating, n_measured))

        summary = (
            f"{n_cases} cases x {len(self.models)} models"
            + (f" x {self.repeats} repeats" if self.repeats > 1 else "")
            + f": {n_measured - non_discriminating} discriminate, "
            f"{non_discriminating} do not ({share:.0%})"
            + (f", {len(unstable)} not reproducible" if unstable else "")
        )
        return CheckResult(
            check=self.name,
            summary=summary,
            findings=tuple(findings),
            stats=stats,
            limitations=LIMITATIONS,
        )

    def _sanity_findings(
        self, stats: dict, non_discriminating: int, n_cases: int
    ) -> list[Finding]:
        """Checks on the SETUP rather than the eval set.

        If these fire, the numbers above are probably not measuring what the
        user thinks they are, so it matters more that they are surfaced than
        that they fit the check's stated job.
        """
        findings = []
        rates = stats["per_model_pass_rate"]
        weakest, strongest = self.model_names[0], self.model_names[-1]
        if rates[weakest] > rates[strongest]:
            findings.append(
                Finding(
                    f"'{weakest}' is listed as the weakest model but passed "
                    f"{rates[weakest]:.0%} of cases against "
                    f"{rates[strongest]:.0%} for '{strongest}'. Either the "
                    f"models are in the wrong order or the scorer is inverted",
                    case_ids=(),
                )
            )
        if n_cases and non_discriminating == n_cases:
            # Same rule as the share finding: name the narrowed scope only when
            # the scope was actually narrowed.
            scope = "no measured case" if stats["n_unstable"] else "no case"
            findings.append(
                Finding(
                    f"{scope} separated these models at all. They may be closer "
                    "in capability than assumed, or the scorer may not be "
                    "sensitive enough to tell them apart",
                    severity=Severity.INFO,
                )
            )
        if not n_cases:
            findings.append(
                Finding(
                    "not one case produced a reproducible verdict, so this run "
                    "measured nothing. Either the scorer is highly "
                    "non-deterministic or the two models are indistinguishable "
                    "on this data",
                    severity=Severity.INFO,
                )
            )
        return findings

    def _score_all(
        self, cases: list[EvalCase]
    ) -> dict[str, tuple[tuple[bool, ...], ...]]:
        """Every (case, model, repeat) verdict, keyed by case id.

        The unit of work is one scorer call, not one case, because a case with
        two models and three repeats is six independent I/O-bound calls and
        there is no reason for them to wait on each other.

        ``max_workers == 1`` takes a plainly serial path rather than a pool of
        one: it keeps the default free of thread-pool semantics entirely, so a
        user whose scorer is not thread-safe is unaffected unless they opt in.
        """
        tasks = [
            (case, model_index, model, name, repeat)
            for case in cases
            for model_index, (model, name) in enumerate(
                zip(self.models, self.model_names, strict=True)
            )
            for repeat in range(self.repeats)
        ]

        if self.max_workers == 1:
            outcomes: list[bool | BaseException] = []
            for case, _, model, name, _ in tasks:
                # Serial path fails fast, exactly as it always has.
                outcomes.append(self._score_one(case, model, name))
        else:
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                futures = [
                    pool.submit(self._score_one, case, model, name)
                    for case, _, model, name, _ in tasks
                ]
                # Collect everything, including failures, before deciding what
                # to raise. Raising from whichever thread happened to finish
                # first would make the error message depend on scheduling.
                outcomes = []
                for future in futures:
                    try:
                        outcomes.append(future.result())
                    except BaseException as exc:  # noqa: BLE001 - re-raised below
                        outcomes.append(exc)

            for outcome in outcomes:
                if isinstance(outcome, BaseException):
                    # Task order is submission order, so the reported failure is
                    # the same one the serial path would have hit.
                    raise outcome

        runs: dict[str, list[list[bool]]] = {
            case.id: [[] for _ in self.models] for case in cases
        }
        for (case, model_index, _, _, _), outcome in zip(
            tasks, outcomes, strict=True
        ):
            runs[case.id][model_index].append(bool(outcome))

        return {
            case_id: tuple(tuple(per_model) for per_model in per_case)
            for case_id, per_case in runs.items()
        }

    def _verdicts_for(self, case: EvalCase) -> tuple[bool, ...]:
        return tuple(
            self._score_one(case, model, name)
            for model, name in zip(self.models, self.model_names, strict=True)
        )

    def _score_one(self, case: EvalCase, model: Any, name: str) -> bool:
        try:
            raw = self.scorer(case, model)
        except Exception as exc:
            # Scoring is the expensive part of this tool. If it breaks on case
            # 400 of 500, the user needs to know exactly where without digging
            # through a traceback from their own scorer.
            raise ScorerError(
                f"scorer raised on case '{case.id}' with model '{name}': {exc}"
            ) from exc

        # numpy / pandas scalars are the NORMAL return type for a scorer built
        # on array maths: `y_pred == y_true` gives np.bool_, not bool. Those are
        # not Python bool or int, so without this they were rejected — and with
        # a baffling message, since type(np.True_).__name__ is itself "bool".
        # Unwrapping via .item() keeps this module dependency-free: it never
        # imports numpy, which matters for the check that is meant to work with
        # anyone's stack. A real array fails .item() and falls through to the
        # error below, which is what should happen.
        if not isinstance(raw, (bool, int, float)) and hasattr(raw, "item"):
            try:
                raw = raw.item()
            except (ValueError, AttributeError):
                pass

        value = _unwrap_scalar(raw)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return float(value) >= self.pass_threshold
        raise ScorerError(
            f"scorer returned {type(raw).__name__} for case '{case.id}' with "
            f"model '{name}': expected bool or float"
        )


def _unwrap_scalar(value: Any) -> Any:
    """Return the Python equivalent of a 0-dimensional array scalar.

    numpy scalars are not Python builtins: np.bool_ is not a bool and np.int64
    is not an int. A scorer written with array maths — `y_pred == y_true` is
    the obvious way to write one — returns exactly those, so rejecting them
    would make "works with any provider" hollow. np.bool_ is the nastiest of
    them because its type name is literally "bool", which produced the
    baffling error "returned bool ... expected bool or float".

    Duck-typing on ndim instead of importing numpy means a torch or jax scalar
    works too, while a real array (ndim >= 1) still falls through and is
    rejected.
    """
    if getattr(value, "ndim", None) == 0 and hasattr(value, "item"):
        return value.item()
    return value


def _n_cases(n: int) -> str:
    return "1 case" if n == 1 else f"{n} cases"


def _is_inverted(verdicts: Sequence[bool]) -> bool:
    """True if a weaker model passed where a stronger one failed.

    Verdicts are ordered weakest to strongest, so a correct pattern never has
    a True before a False. Anything else contradicts the stated ordering.
    """
    return any(
        verdicts[i] and not verdicts[j]
        for i in range(len(verdicts))
        for j in range(i + 1, len(verdicts))
    )
