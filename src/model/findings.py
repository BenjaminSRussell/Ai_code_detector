"""Data model for agent-facing scan findings."""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class Finding:
    """A single actionable finding surfaced by the agent-prep scanner."""
    type: str
    file: str
    severity: str
    description: str
    line: Optional[int] = None
    function: Optional[str] = None
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScanFindings:
    """All findings produced by a single agent-prep scan run."""
    repo_path: str
    ai_probability: float
    findings: List[Finding] = field(default_factory=list)

    def by_type(self, finding_type: str) -> List[Finding]:
        return [f for f in self.findings if f.type == finding_type]
