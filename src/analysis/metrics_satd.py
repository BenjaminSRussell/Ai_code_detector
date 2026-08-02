"""Self-admitted technical debt (SATD) marker detection."""

import re
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class SATDMarker:
    """A single self-admitted technical debt marker found in a file."""
    file_path: str
    line: int
    marker: str
    text: str


@dataclass
class SATDFeatures:
    """SATD markers found in a single file."""
    markers: List[SATDMarker]
    density: float


class SATDAnalyzer:
    """Scans source code for self-admitted technical debt markers."""

    MARKER_PATTERN = re.compile(r'\b(TODO|FIXME|HACK|XXX)\b', re.IGNORECASE)

    def __init__(self, config: Dict = None):
        self.config = config or {}

    def analyze_file(self, code: str, file_path: str) -> SATDFeatures:
        """Scan a file's source for SATD markers.

        Args:
            code: Source code content
            file_path: Relative path of the file (for reporting)

        Returns:
            SATDFeatures
        """
        lines = code.split('\n')
        markers = []

        for i, line in enumerate(lines, start=1):
            match = self.MARKER_PATTERN.search(line)
            if match:
                markers.append(SATDMarker(
                    file_path=file_path,
                    line=i,
                    marker=match.group(1).upper(),
                    text=line.strip()[:200],
                ))

        non_empty_lines = sum(1 for line in lines if line.strip())
        density = (len(markers) / non_empty_lines * 100) if non_empty_lines else 0.0

        return SATDFeatures(markers=markers, density=density)
