"""Measuring a detector: confusion matrix, timing, memory.

## Why precision and recall both need negatives

A fixture containing only planted problems can measure recall and nothing else.
Every metric below is computed against an explicit *universe* of items -- all
cases, or all case pairs -- so a detector that fires on everything scores 1.0
recall and visibly terrible precision. Fixtures without adversarial negatives
would make this file lie.

## What F1 is and is not appropriate for

F1 is reported for per-case and per-pair CLASSIFICATION: exact duplicates,
semantic duplicates, leakage, ambiguity, and the discrimination case classes.

It is deliberately NOT reported for:

* **Class imbalance** -- five datasets is far too small a universe for a
  meaningful F1, so the result is reported as the confusion matrix itself.
* **Judge instability** -- the question is not "did it fire" but "is the measured
  agreement close to the constructed agreement", which is an error magnitude.
* **Statistical noise** -- the question is the false-positive rate under a true
  null, which is a rate against a theoretical target (alpha), not an F1.

Reporting an F1 for those would be a number with no interpretation, which is the
kind of thing this project exists to complain about.

## Memory measurement, and its limits

`tracemalloc` measures peak bytes allocated *through Python's allocator*. It
therefore misses:

  * numpy array buffers, which are allocated by numpy outside that allocator,
  * anything torch or sentence-transformers allocates,
  * memory the OS has mapped but Python has not touched.

So the figure is useful for comparing two versions of the same pure-Python
detector and useless as an absolute footprint. It is labelled
`python_peak_kib` rather than "memory" for exactly that reason, and the
embedding-based path reports it knowing it undercounts by orders of magnitude.

## Runtime, and its limits

Wall clock, best-of-N to reduce scheduler noise, on whatever machine ran it.
Comparable between two runs on one machine; not comparable across machines, and
not a benchmark of anything but this implementation. The recorded platform
string is there so a contributor can tell whether a comparison is meaningful.
"""

from __future__ import annotations

import platform
import statistics
import time
import tracemalloc
from collections.abc import Callable, Collection, Hashable
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "Confusion",
    "Measurement",
    "environment",
    "measure",
    "score_sets",
]


@dataclass(frozen=True, slots=True)
class Confusion:
    """A 2x2 table plus the rates derived from it.

    Every rate returns None where its denominator is zero rather than 0.0. A
    precision of 0.0 means "everything it flagged was wrong"; an undefined
    precision means "it flagged nothing", and those are different results.
    """

    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def precision(self) -> float | None:
        flagged = self.tp + self.fp
        return self.tp / flagged if flagged else None

    @property
    def recall(self) -> float | None:
        actual = self.tp + self.fn
        return self.tp / actual if actual else None

    @property
    def f1(self) -> float | None:
        p, r = self.precision, self.recall
        if p is None or r is None or p + r == 0:
            return None
        return 2 * p * r / (p + r)

    @property
    def false_positive_rate(self) -> float | None:
        """FP / (FP + TN) — of the things that were fine, how many were flagged."""
        negatives = self.fp + self.tn
        return self.fp / negatives if negatives else None

    @property
    def false_negative_rate(self) -> float | None:
        """FN / (FN + TP) — of the real problems, how many were missed."""
        positives = self.fn + self.tp
        return self.fn / positives if positives else None

    def as_dict(self, *, with_f1: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
            "n_items": self.n,
            "precision": _round(self.precision),
            "recall": _round(self.recall),
            "false_positive_rate": _round(self.false_positive_rate),
            "false_negative_rate": _round(self.false_negative_rate),
        }
        if with_f1:
            out["f1"] = _round(self.f1)
        else:
            out["f1"] = None
            out["f1_omitted_because"] = (
                "F1 has no useful interpretation for this category; see "
                "benchmarks/metrics.py"
            )
        return out


def score_sets(
    predicted: Collection[Hashable],
    actual: Collection[Hashable],
    universe: Collection[Hashable],
) -> Confusion:
    """Confusion matrix from a predicted set against a labelled set.

    Args:
        predicted: what the detector flagged.
        actual: what the fixture planted.
        universe: EVERY item that could have been flagged. Required, not
            derived: without it there is no true-negative count and therefore no
            false-positive rate, and a detector that flags everything would look
            perfect.

    Raises:
        ValueError: predicted or actual contains something outside the universe,
            which means the two are being compared at different granularities --
            cases against pairs, say -- and every number would be meaningless.
    """
    u = set(universe)
    p = set(predicted)
    a = set(actual)
    stray = (p | a) - u
    if stray:
        raise ValueError(
            f"{len(stray)} item(s) outside the universe, e.g. "
            f"{sorted(map(str, stray))[:3]}. Predicted and actual must be "
            "over the same item type."
        )
    return Confusion(
        tp=len(p & a),
        fp=len(p - a),
        fn=len(a - p),
        tn=len(u - p - a),
    )


@dataclass(frozen=True, slots=True)
class Measurement:
    """Timing and allocation for one call. See the module docstring for limits."""

    seconds_best: float
    seconds_median: float
    repeats: int
    python_peak_kib: float
    n_items: int
    result: Any = field(default=None, repr=False)

    @property
    def seconds_per_item(self) -> float | None:
        return self.seconds_best / self.n_items if self.n_items else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "seconds_best": round(self.seconds_best, 6),
            "seconds_median": round(self.seconds_median, 6),
            "repeats": self.repeats,
            "ms_per_item": (
                round(self.seconds_per_item * 1000, 4)
                if self.seconds_per_item is not None
                else None
            ),
            "python_peak_kib": round(self.python_peak_kib, 1),
            "python_peak_caveat": "tracemalloc only: excludes numpy and torch "
            "buffers, so this undercounts the embedding path by orders of "
            "magnitude. Comparable between runs, not an absolute footprint.",
        }


def measure(
    fn: Callable[[], Any], *, n_items: int, repeats: int = 3
) -> Measurement:
    """Run ``fn`` and record how long it took and what it allocated.

    Best-of-N for the headline time, because the minimum is the least
    contaminated by scheduler noise; the median is recorded too, so a large gap
    between them shows the machine was busy and the numbers should be distrusted.

    Memory is measured on a SEPARATE run with tracemalloc active. Tracing slows
    execution measurably, so timing it at the same time would inflate the
    duration by something like a third and quietly make every runtime figure
    wrong.
    """
    if repeats < 1:
        raise ValueError("repeats must be at least 1")

    result = fn()  # warm-up: first call pays import and cache costs

    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        timings.append(time.perf_counter() - started)

    tracemalloc.start()
    try:
        fn()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    return Measurement(
        seconds_best=min(timings),
        seconds_median=statistics.median(timings),
        repeats=repeats,
        python_peak_kib=peak / 1024,
        n_items=n_items,
        result=result,
    )


def environment() -> dict[str, Any]:
    """What ran the benchmark, so a contributor can tell whether two runs
    are comparable. Runtime figures are not portable across machines."""
    import evallint

    return {
        "evallint_version": evallint.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
    }


def _round(value: float | None, places: int = 4) -> float | None:
    return None if value is None else round(value, places)
