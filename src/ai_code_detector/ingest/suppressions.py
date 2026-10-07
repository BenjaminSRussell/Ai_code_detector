"""`.aicdignore` suppressions: path globs (with an optional rationale) that
are skipped by every scan, excluded from the exit-code gate, and counted in
reports so the omission stays visible.

File format (gitignore-like, one rule per line)::

    # comment line
    third_party/            # vendored deps, reviewed upstream
    **/*_pb2.py             # protoc output
    build/generated/*.rs
    !build/generated/keep.rs  # re-include (last matching rule wins)

* ``pattern  # rationale``: everything after whitespace + ``#`` is the rationale.
* A pattern containing ``/`` (other than a trailing one) is anchored at the
  repo root; otherwise it matches a file or directory name at any depth.
* A trailing ``/`` matches directories only (and everything under them).
* ``*`` and ``?`` stay within one path segment, ``**`` spans segments, and
  ``[...]`` is a character class. ``!pattern`` re-includes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional, Union

DEFAULT_FILENAME = ".aicdignore"
MAX_LISTED_PATHS = 200


def _glob_to_regex(glob: str) -> "re.Pattern[str]":
    out = []
    i, n = 0, len(glob)
    while i < n:
        c = glob[i]
        if glob.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif glob.startswith("/**", i) and i + 3 == n:
            out.append("/.*")
            i += 3
        elif glob.startswith("**", i):
            out.append(".*")
            i += 2
        elif c == "*":
            out.append("[^/]*")
            i += 1
        elif c == "?":
            out.append("[^/]")
            i += 1
        elif c == "[":
            j = glob.find("]", i + 1)
            if j == -1:
                out.append(re.escape(c))
                i += 1
            else:
                body = glob[i + 1:j]
                if body.startswith("!"):
                    body = "^" + body[1:]
                out.append("[" + body.replace("\\", "\\\\") + "]")
                i = j + 1
        else:
            out.append(re.escape(c))
            i += 1
    return re.compile("".join(out) + r"\Z")


@dataclass
class SuppressionRule:
    pattern: str
    rationale: str = ""
    negate: bool = False
    line: int = 0
    _regex: "re.Pattern[str]" = field(default=None, repr=False)
    _anchored: bool = field(default=False, repr=False)
    _dir_only: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        body = self.pattern
        self._dir_only = body.endswith("/")
        body = body.rstrip("/")
        self._anchored = "/" in body
        body = body.lstrip("/")
        self._regex = _glob_to_regex(body)

    def matches(self, rel_path: str) -> bool:
        parts = PurePosixPath(rel_path).parts
        candidates = []
        for k in range(1, len(parts) + 1):
            is_dir = k < len(parts)
            if self._dir_only and not is_dir:
                continue
            candidates.append("/".join(parts[:k]) if self._anchored else parts[k - 1])
        return any(self._regex.match(c) for c in candidates)


def parse_rules(text: str) -> List[SuppressionRule]:
    rules: List[SuppressionRule] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        rationale = ""
        m = re.search(r"\s+#\s?", line)
        if m:
            rationale = line[m.end():].strip()
            line = line[:m.start()].strip()
        negate = line.startswith("!")
        if negate:
            line = line[1:].strip()
        if line.startswith("\\"):  # escaped leading '#' or '!'
            line = line[1:]
        if not line:
            continue
        rules.append(SuppressionRule(pattern=line, rationale=rationale, negate=negate, line=lineno))
    return rules


@dataclass
class Suppressions:
    rules: List[SuppressionRule] = field(default_factory=list)
    source: Optional[str] = None

    @classmethod
    def load(cls, root: Union[str, Path], explicit: Optional[Union[str, Path]] = None) -> "Suppressions":
        """Load ``explicit`` if given (must exist), else ``<root>/.aicdignore`` if present."""
        if explicit is not None:
            path = Path(explicit)
            if not path.is_file():
                raise FileNotFoundError(f"suppressions file not found: {path}")
        else:
            path = Path(root) / DEFAULT_FILENAME
            if not path.is_file():
                return cls()
        return cls(rules=parse_rules(path.read_text(encoding="utf-8", errors="replace")), source=str(path))

    def match(self, rel_path: Union[str, Path]) -> Optional[SuppressionRule]:
        """Return the rule that suppresses ``rel_path``, or None (last match wins)."""
        rel = PurePosixPath(Path(rel_path).as_posix()).as_posix()
        hit: Optional[SuppressionRule] = None
        for rule in self.rules:
            if rule.matches(rel):
                hit = None if rule.negate else rule
        return hit


@dataclass
class SuppressionSummary:
    """What a scan skipped; serialised into every report."""
    source: Optional[str] = None
    paths: Dict[str, str] = field(default_factory=dict)  # rel path -> pattern
    rules: List[SuppressionRule] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.paths)

    def record(self, rel_path: str, rule: SuppressionRule) -> None:
        self.paths[rel_path] = rule.pattern

    def to_dict(self) -> Dict:
        per_rule: Dict[str, int] = {}
        for pattern in self.paths.values():
            per_rule[pattern] = per_rule.get(pattern, 0) + 1
        listed = sorted(self.paths)
        return {
            "count": self.count,
            "source": self.source,
            "rules": [
                {"pattern": r.pattern, "rationale": r.rationale, "count": per_rule.get(r.pattern, 0)}
                for r in self.rules if not r.negate
            ],
            "paths": listed[:MAX_LISTED_PATHS],
            "paths_truncated": len(listed) > MAX_LISTED_PATHS,
        }
