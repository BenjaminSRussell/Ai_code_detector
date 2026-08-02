"""Report writers for agent-facing scan findings (Markdown + JSON)."""

import json
from pathlib import Path

from model.findings import ScanFindings

SEVERITY_ORDER = {"high": 0, "warning": 1, "info": 2}


class FindingsMarkdownWriter:
    """Writes ScanFindings as a Markdown file intended for a coding agent to read."""

    def generate(self, findings: ScanFindings, output_path: Path = None) -> str:
        lines = []
        lines.append("# AI Scan Findings")
        lines.append("")
        lines.append(f"**Repository:** `{findings.repo_path}`")
        lines.append(f"**AI Probability:** {findings.ai_probability * 100:.1f}%")
        lines.append("")

        if not findings.findings:
            lines.append("No findings.")
        else:
            sorted_findings = sorted(
                findings.findings,
                key=lambda f: SEVERITY_ORDER.get(f.severity, 99),
            )

            for finding in sorted_findings:
                location = finding.file
                if finding.line is not None:
                    location += f":{finding.line}"

                lines.append(f"## [{finding.severity.upper()}] {finding.type} — `{location}`")
                lines.append("")
                lines.append(finding.description)

                if finding.function:
                    lines.append("")
                    lines.append(f"Function: `{finding.function}`")

                lines.append("")

        report = "\n".join(lines)

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report)

        return report


class FindingsJSONWriter:
    """Writes ScanFindings as a machine-readable JSON task queue."""

    def generate(self, findings: ScanFindings, output_path: Path = None) -> str:
        payload = {
            "repo_path": findings.repo_path,
            "ai_probability": findings.ai_probability,
            "findings": [
                {
                    "type": f.type,
                    "file": f.file,
                    "line": f.line,
                    "function": f.function,
                    "severity": f.severity,
                    "description": f.description,
                    "evidence": f.evidence,
                }
                for f in findings.findings
            ],
        }

        report = json.dumps(payload, indent=2)

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report)

        return report
