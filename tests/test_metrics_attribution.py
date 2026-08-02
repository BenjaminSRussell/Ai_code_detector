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


def test_does_not_match_human_named_claude():
    """Regression test: human co-authors named Claude should not trigger AI attribution."""
    analyzer = AttributionAnalyzer()
    repo_info = _make_repo_info([
        "Add feature\n\nCo-Authored-By: Claude Dupont <claude.dupont@example.com>",
    ])

    features = analyzer.analyze_repo(repo_info)

    assert features.has_ai_attribution is False
    assert features.match_ratio == 0.0
    assert features.matches == []


def test_does_not_match_claudel_name():
    """Regression test: human co-authors with similar names should not trigger AI attribution."""
    analyzer = AttributionAnalyzer()
    repo_info = _make_repo_info([
        "Fix bug\n\nCo-Authored-By: Claudel Martin <cmartin@example.com>",
    ])

    features = analyzer.analyze_repo(repo_info)

    assert features.has_ai_attribution is False


def test_does_not_match_claudette_name():
    """Regression test: human co-authors with similar names should not trigger AI attribution."""
    analyzer = AttributionAnalyzer()
    repo_info = _make_repo_info([
        "Add tests\n\nCo-Authored-By: Claudette Okoye <c.okoye@example.com>",
    ])

    features = analyzer.analyze_repo(repo_info)

    assert features.has_ai_attribution is False


def test_still_matches_real_claude_code_trailer():
    """Confirm that the real Claude Code trailer with anthropic.com email still matches."""
    analyzer = AttributionAnalyzer()
    repo_info = _make_repo_info([
        "Add login form\n\nCo-Authored-By: Claude <noreply@anthropic.com>",
    ])

    features = analyzer.analyze_repo(repo_info)

    assert features.has_ai_attribution is True
    assert len(features.matches) == 1


def test_matches_claude_with_model_name():
    """Regression test: Claude Code trailers with model names (e.g., Claude Sonnet 5) should match.

    This is the actual format used in the repo's own commit history.
    """
    analyzer = AttributionAnalyzer()
    repo_info = _make_repo_info([
        "Add feature\n\nCo-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>",
    ])

    features = analyzer.analyze_repo(repo_info)

    assert features.has_ai_attribution is True
    assert len(features.matches) == 1
    assert features.matches[0].commit_sha == "sha0"
