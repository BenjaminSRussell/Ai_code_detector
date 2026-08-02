"""Tests for AI-attribution commit-trailer analyzer."""

import sys
from pathlib import Path
from datetime import datetime
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from analysis.metrics_attribution import AttributionAnalyzer
from ingest.git_loader import RepoInfo, CommitInfo


def _make_repo_info(messages):
    commits = [
        CommitInfo(
            sha=f"sha{i}",
            author="Test Author",
            email="test@example.com",
            timestamp=datetime(2026, 1, 1),
            message=message,
            files_changed=[],
            lines_added=0,
            lines_deleted=0,
        )
        for i, message in enumerate(messages)
    ]
    return RepoInfo(
        path=Path("."),
        is_git=True,
        remote_url=None,
        commits=commits,
        authors=["Test Author"],
        total_commits=len(commits),
        first_commit_date=None,
        last_commit_date=None,
    )


def test_no_attribution_found():
    analyzer = AttributionAnalyzer()
    repo_info = _make_repo_info(["fix bug in parser", "add tests"])

    features = analyzer.analyze_repo(repo_info)

    assert features.has_ai_attribution is False
    assert features.match_ratio == 0.0
    assert features.matches == []


def test_detects_claude_code_co_author_trailer():
    analyzer = AttributionAnalyzer()
    repo_info = _make_repo_info([
        "Add login form\n\nCo-Authored-By: Claude <noreply@anthropic.com>",
        "Fix typo",
    ])

    features = analyzer.analyze_repo(repo_info)

    assert features.has_ai_attribution is True
    assert features.match_ratio == 0.5
    assert len(features.matches) == 1
    assert features.matches[0].commit_sha == "sha0"


def test_custom_pattern_from_config():
    analyzer = AttributionAnalyzer(config={'ai_attribution_patterns': [r'written by robots']})
    repo_info = _make_repo_info(["this file was written by robots"])

    features = analyzer.analyze_repo(repo_info)

    assert features.has_ai_attribution is True


def test_empty_commit_history_returns_no_attribution():
    analyzer = AttributionAnalyzer()
    repo_info = _make_repo_info([])

    features = analyzer.analyze_repo(repo_info)

    assert features.has_ai_attribution is False
    assert features.match_ratio == 0.0
