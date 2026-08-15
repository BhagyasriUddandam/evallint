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
tool: evalcheck never talks to a model provider, so it works with anyone's
stack and the tests need no API key.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
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
    "Model outputs are usually sampled, so with temperature above 0 a case can "
    "flip between runs. Score at temperature 0, or repeat the run, before "
    "acting on any single case listed here.",
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
    ) -> None:
        """
        Args:
            scorer: ``scorer(case, model) -> bool | float``. Returns whether
                the model got this case right, or a score compared against
                ``pass_threshold``. evalcheck never calls a provider itself.
            models: Two or more model identifiers ordered WEAKEST FIRST. The
                order is what makes inversion detectable, so getting it
                backwards inverts that finding — see the limitations.
            pass_threshold: A float score >= this counts as a pass. Ignored
                for bool scorers.
            max_non_discriminating_share: Warn above this share of cases.
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
        self.scorer = scorer
        self.models = tuple(models)
        self.model_names = tuple(labels)
        self.pass_threshold = pass_threshold
        self.max_non_discriminating_share = max_non_discriminating_share

    def run(self, eval_set: EvalSet) -> CheckResult:
        cases = list(eval_set)
        verdicts = {case.id: self._verdicts_for(case) for case in cases}

        ceiling = [c.id for c in cases if all(verdicts[c.id])]
        floor = [c.id for c in cases if not any(verdicts[c.id])]
        inverted = [c.id for c in cases if _is_inverted(verdicts[c.id])]
        non_discriminating = len(ceiling) + len(floor)
        n_cases = len(cases)
        share = non_discriminating / n_cases

        stats = {
            "n_cases": n_cases,
            "models": list(self.model_names),
            "n_scorer_calls": n_cases * len(self.models),
            "pass_threshold": self.pass_threshold,
            "n_discriminating": n_cases - non_discriminating,
            "n_non_discriminating": non_discriminating,
            "non_discriminating_share": share,
            "effective_n_cases": n_cases - non_discriminating,
            "n_ceiling": len(ceiling),
            "n_floor": len(floor),
            "n_inverted": len(inverted),
            "per_model_pass_rate": {
                name: sum(verdicts[c.id][i] for c in cases) / n_cases
                for i, name in enumerate(self.model_names)
            },
            "ceiling_case_ids": ceiling,
            "floor_case_ids": floor,
            "inverted_case_ids": inverted,
        }

        findings: list[Finding] = []
        if share > self.max_non_discriminating_share:
            findings.append(
                Finding(
                    f"{share:.0%} of cases ({non_discriminating}/{n_cases}) give "
                    f"every model the same verdict, so they carry no evidence "
                    f"about which model is better. This eval discriminates like "
                    f"a {n_cases - non_discriminating}-case set"
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
        findings.extend(self._sanity_findings(stats, non_discriminating, n_cases))

        summary = (
            f"{n_cases} cases x {len(self.models)} models: "
            f"{n_cases - non_discriminating} discriminate, "
            f"{non_discriminating} do not ({share:.0%})"
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
        if non_discriminating == n_cases:
            findings.append(
                Finding(
                    "no case separated these models at all. They may be closer "
                    "in capability than assumed, or the scorer may not be "
                    "sensitive enough to tell them apart",
                    severity=Severity.INFO,
                )
            )
        return findings

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
