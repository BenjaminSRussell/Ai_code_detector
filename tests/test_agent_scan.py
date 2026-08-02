"""Integration test for the agent-prep scan orchestrator."""

import sys
import subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent_scan import AgentPrepScanner
from analysis.metrics_duplication import RepoDuplicationAnalyzer
import analysis.metrics_performance as metrics_performance


def _init_repo(repo_dir: Path):
    subprocess.run(["git", "init", "-q"], cwd=repo_dir, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_dir, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo_dir, check=True)


def _commit_all(repo_dir: Path, message: str):
    subprocess.run(["git", "add", "-A"], cwd=repo_dir, check=True)
    subprocess.run(["git", "commit", "-q", "-m", message], cwd=repo_dir, check=True)


def test_scan_surfaces_all_finding_types(tmp_path):
    repo_dir = tmp_path / "sample_repo"
    repo_dir.mkdir()
    _init_repo(repo_dir)

    slow_code = (
        "def matrix_multiply(a, b):\n"
        "    result = []\n"
        "    for i in a:\n"
        "        for j in b:\n"
        "            result.append(i * j)\n"
        "    return result\n"
        "\n"
        "# TODO: optimize this later\n"
    )

    (repo_dir / "module_a.py").write_text(slow_code)
    (repo_dir / "module_b.py").write_text(slow_code)

    _commit_all(repo_dir, "Add matrix helpers\n\nCo-Authored-By: Claude <noreply@anthropic.com>")

    scanner = AgentPrepScanner(enable_profiling=False)
    findings = scanner.scan(str(repo_dir), verbose=False)

    finding_types = {f.type for f in findings.findings}

    assert "ai_attribution" in finding_types
    assert "duplication" in finding_types
    assert "satd" in finding_types
    assert "performance_hotspot" in finding_types
    assert findings.repo_path == str(repo_dir.resolve())


# --- FIX 1: default config must exclude .venv -------------------------------

def test_default_config_excludes_venv_directory():
    """AgentPrepScanner() with no config_path must fall back to the packaged
    configs/default.yaml, whose excluded_dirs includes both venv and .venv."""
    scanner = AgentPrepScanner()

    assert ".venv" in scanner.file_filter.excluded_dirs


def test_scan_does_not_analyze_files_under_venv(tmp_path):
    repo_dir = tmp_path / "venv_repo"
    repo_dir.mkdir()
    _init_repo(repo_dir)

    (repo_dir / "app.py").write_text("def add(a, b):\n    return a + b\n")

    venv_pkg_dir = repo_dir / ".venv" / "lib" / "site-packages" / "fakepkg"
    venv_pkg_dir.mkdir(parents=True)
    (venv_pkg_dir / "installed.py").write_text(
        "# TODO: this is obviously-fake installed package content\n"
        "def totally_fake_marker_function():\n"
        "    pass\n"
    )

    _commit_all(repo_dir, "Add app and a fake .venv installed package")

    scanner = AgentPrepScanner()
    findings = scanner.scan(str(repo_dir), verbose=False)

    for finding in findings.findings:
        assert ".venv" not in finding.file
        assert ".venv" not in str(finding.evidence)


# --- FIX 2: overlapping n-gram windows collapse into one finding ------------

def test_duplication_findings_collapse_overlapping_windows():
    shared_block = (
        "def process(data):\n"
        "    result = []\n"
        "    for item in data:\n"
        "        cleaned = item.strip()\n"
        "        validated = cleaned.lower()\n"
        "        result.append(validated)\n"
        "    return result\n"
    )

    file_contents = {
        "a.py": shared_block,
        "b.py": shared_block,
    }

    duplication_analyzer = RepoDuplicationAnalyzer()
    features = duplication_analyzer.analyze_repo(file_contents)

    # Sanity check: overlapping 3-line windows over a 7-line shared block
    # produce multiple DuplicateBlock entries in the raw analyzer output.
    assert len(features.duplicate_blocks) > 1

    scanner = AgentPrepScanner()
    duplication_findings = scanner._build_duplication_findings(features)

    assert len(duplication_findings) == 1
    assert duplication_findings[0].type == "duplication"
    assert duplication_findings[0].file == "a.py"


# --- FIX 5: profiling no-op paths must not crash the scan --------------------

def test_scan_completes_with_profiling_enabled_and_no_tests_dir(tmp_path):
    repo_dir = tmp_path / "no_tests_repo"
    repo_dir.mkdir()
    _init_repo(repo_dir)
    (repo_dir / "app.py").write_text("def add(a, b):\n    return a + b\n")
    _commit_all(repo_dir, "Add app")

    scanner = AgentPrepScanner(enable_profiling=True)
    findings = scanner.scan(str(repo_dir), verbose=False)

    assert findings.repo_path == str(repo_dir.resolve())


def test_scan_prints_no_tests_dir_message_when_verbose(tmp_path, capsys):
    repo_dir = tmp_path / "no_tests_repo_verbose"
    repo_dir.mkdir()
    _init_repo(repo_dir)
    (repo_dir / "app.py").write_text("def add(a, b):\n    return a + b\n")
    _commit_all(repo_dir, "Add app")

    scanner = AgentPrepScanner(enable_profiling=True)
    scanner.scan(str(repo_dir), verbose=True)

    captured = capsys.readouterr()
    assert "No tests/ directory found" in captured.out


# --- FIX 7: enable_profiling gate must actually gate profiler invocation ----

def test_profiling_gate_is_not_invoked_when_disabled(tmp_path, monkeypatch):
    repo_dir = tmp_path / "profiled_repo_disabled"
    repo_dir.mkdir()
    _init_repo(repo_dir)
    (repo_dir / "app.py").write_text("def add(a, b):\n    return a + b\n")
    tests_dir = repo_dir / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_app.py").write_text("def test_add():\n    assert True\n")
    _commit_all(repo_dir, "Add app with tests")

    calls = []

    def fake_profile(self, repo_path, entry_point):
        calls.append((repo_path, entry_point))
        return {}

    monkeypatch.setattr(metrics_performance.PerformanceProfiler, "profile", fake_profile)

    scanner = AgentPrepScanner(enable_profiling=False)
    scanner.scan(str(repo_dir), verbose=False)

    assert calls == []


def test_profiling_gate_is_invoked_when_enabled_with_tests_dir(tmp_path, monkeypatch):
    repo_dir = tmp_path / "profiled_repo_enabled"
    repo_dir.mkdir()
    _init_repo(repo_dir)
    (repo_dir / "app.py").write_text("def add(a, b):\n    return a + b\n")
    tests_dir = repo_dir / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_app.py").write_text("def test_add():\n    assert True\n")
    _commit_all(repo_dir, "Add app with tests")

    calls = []

    def fake_profile(self, repo_path, entry_point):
        calls.append((repo_path, entry_point))
        return {}

    monkeypatch.setattr(metrics_performance.PerformanceProfiler, "profile", fake_profile)

    scanner = AgentPrepScanner(enable_profiling=True)
    scanner.scan(str(repo_dir), verbose=False)

    assert len(calls) == 1
