"""Write-through response cache, shared by the demo scorer backends.

NOT part of the shipped library. Lives here so the Anthropic backend and the
Ollama backend cannot drift into two different caching implementations — a
cache that quietly serves stale results is the failure mode this project
exists to catch, and the surest way to reintroduce it is to write it twice.

Key construction stays with each backend, because the two have genuinely
different request identities (one has an effort setting, the other a seed and
a think flag). What is shared is the part with no provider knowledge at all:
hashing a key, persisting on every miss, and tallying spend.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

__all__ = ["ResponseCache", "cache_key"]


def cache_key(key_parts: dict[str, Any]) -> str:
    """Hash a request identity into a cache key.

    One function, used by every write path and every read path. If callers
    built keys separately they could drift, and a reader would silently report
    "not in cache" for entries sitting right there.

    sort_keys so an identical request always hashes identically; without it,
    dict ordering could produce a cache miss and a charge.
    """
    return hashlib.sha256(
        json.dumps(key_parts, sort_keys=True).encode("utf-8")
    ).hexdigest()


class ResponseCache:
    """A write-through JSON cache so a re-run never re-pays for a response.

    Written to disk on every miss rather than once at the end: an exception on
    case 15 of 20 must not throw away the 14 answers already paid for.

    Spend is accumulated at the point of the miss rather than totalled
    afterwards from a list of records — cached responses cost nothing, and a
    tally that could not tell a hit from a miss would report a re-run as
    costing full price. ``pricing`` maps model name to (input, output) dollars
    per million tokens; omit it for a local backend, where everything is free
    and the counters still show what was computed versus reused.
    """

    def __init__(
        self, path: Path, pricing: dict[str, tuple[float, float]] | None = None
    ) -> None:
        self.path = path
        self.pricing = pricing or {}
        self.hits = 0
        self.misses = 0
        self.spend_usd = 0.0
        self._data: dict[str, Any] = {}
        if path.is_file():
            self._data = json.loads(path.read_text(encoding="utf-8"))

    def get_or_call(self, key_parts: dict[str, Any], call) -> dict[str, Any]:
        key = cache_key(key_parts)

        if key in self._data:
            self.hits += 1
            return self._data[key]

        self.misses += 1
        record = call()
        in_rate, out_rate = self.pricing.get(record["model"], (0.0, 0.0))
        self.spend_usd += record["usage"]["input"] / 1e6 * in_rate
        self.spend_usd += record["usage"]["output"] / 1e6 * out_rate

        self._data[key] = record
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
        return record

    def get(self, key_parts: dict[str, Any]) -> dict[str, Any] | None:
        """Read-only lookup, for reporting tools that must never call out."""
        return self._data.get(cache_key(key_parts))

    def __len__(self) -> int:
        return len(self._data)
