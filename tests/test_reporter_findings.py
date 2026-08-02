"""Tests for the agent-facing findings report writers."""

import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model.findings import Finding, ScanFindings
from report.reporter_findings import FindingsMarkdownWriter, FindingsJSONWriter


def _sample_findings():
    return ScanFindings(
        repo_path="/tmp/example",
        ai_probability=0.72,
        findings=[
            Finding(
                type="performance_hotspot",
                file="src/slow.py",
                line=42,
                function="matrix_multiply",
                severity="warning",
                description="Nested loops 2 levels deep suggest O(n^2) behavior.",
            ),
            Finding(
                type="ai_attribution",
                file="src/app.py",
                severity="high",
                description="Commit message contains a Claude Code attribution trailer.",
                evidence={"commit_sha": "abc123"},
            ),
        ],
    )


def test_markdown_writer_includes_all_findings(tmp_path):
    writer = FindingsMarkdownWriter()
    output_path = tmp_path / "AI_SCAN_FINDINGS.md"

    report = writer.generate(_sample_findings(), output_path)

    assert "ai_attribution" in report
    assert "performance_hotspot" in report
    assert "src/slow.py:42" in report
    assert output_path.exists()


def test_markdown_writer_sorts_high_severity_first():
    writer = FindingsMarkdownWriter()

    report = writer.generate(_sample_findings())

    high_index = report.index("[HIGH]")
    warning_index = report.index("[WARNING]")
    assert high_index < warning_index


def test_markdown_writer_handles_no_findings():
    writer = FindingsMarkdownWriter()
    empty = ScanFindings(repo_path="/tmp/clean", ai_probability=0.1, findings=[])

    report = writer.generate(empty)

    assert "No findings." in report


def test_markdown_writer_includes_verdict_for_known_probability():
    writer = FindingsMarkdownWriter()

    report = writer.generate(_sample_findings())

    # 0.72 falls in the >=0.6 band per MarkdownReporter._get_verdict.
    assert "**Verdict:** Likely AI-generated" in report


def test_markdown_writer_includes_attribution_note_when_attribution_present():
    writer = FindingsMarkdownWriter()

    report = writer.generate(_sample_findings())

    assert "independent of and not included in the AI Probability score" in report


def test_markdown_writer_omits_attribution_note_when_no_attribution_findings():
    writer = FindingsMarkdownWriter()
    findings = ScanFindings(
        repo_path="/tmp/example",
        ai_probability=0.72,
        findings=[
            Finding(
                type="performance_hotspot",
                file="src/slow.py",
                line=42,
                function="matrix_multiply",
                severity="warning",
                description="Nested loops 2 levels deep suggest O(n^2) behavior.",
            ),
        ],
    )

    report = writer.generate(findings)

    assert "independent of and not included in the AI Probability score" not in report
    # Verdict should still be present regardless of attribution findings.
    assert "**Verdict:**" in report


def test_json_writer_produces_valid_json_with_all_fields(tmp_path):
    writer = FindingsJSONWriter()
    output_path = tmp_path / "ai_scan_findings.json"

    report = writer.generate(_sample_findings(), output_path)
    payload = json.loads(report)

    assert payload["repo_path"] == "/tmp/example"
    assert len(payload["findings"]) == 2
    assert payload["findings"][0]["line"] == 42
    assert payload["findings"][1]["type"] == "ai_attribution"
    assert output_path.exists()
