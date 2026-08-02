"""Tests for the agent-prep scan CLI."""

import sys
import subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from click.testing import CliRunner
from cli_agent_scan import main


def _init_repo(repo_dir: Path):
    subprocess.run(["git", "init", "-q"], cwd=repo_dir, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_dir, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo_dir, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo_dir, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "initial commit"], cwd=repo_dir, check=True)


def test_cli_writes_findings_files(tmp_path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "app.py").write_text("def add(a, b):\n    return a + b\n")
    _init_repo(repo_dir)

    runner = CliRunner()
    result = runner.invoke(main, [str(repo_dir), '--quiet'])

    assert result.exit_code == 0
    assert (repo_dir / 'AI_SCAN_FINDINGS.md').exists()
    assert (repo_dir / 'ai_scan_findings.json').exists()


def test_cli_respects_output_directory_override(tmp_path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "app.py").write_text("def add(a, b):\n    return a + b\n")
    _init_repo(repo_dir)

    output_dir = tmp_path / "scan-output"

    runner = CliRunner()
    result = runner.invoke(main, [str(repo_dir), '--output', str(output_dir), '--quiet'])

    assert result.exit_code == 0
    assert (output_dir / 'AI_SCAN_FINDINGS.md').exists()
    assert not (repo_dir / 'AI_SCAN_FINDINGS.md').exists()


def test_cli_handles_nonexistent_path_gracefully(tmp_path):
    missing_path = tmp_path / "does_not_exist"

    runner = CliRunner()
    result = runner.invoke(main, [str(missing_path), '--quiet'])

    assert result.exit_code != 0
    assert "Error:" in result.output
    # A raw, uncaught traceback would surface as some exception other than the
    # SystemExit(1) that our own `sys.exit(1)` raises after printing a clean
    # "Error:" message — anything else here means the exception escaped the
    # try/except and wasn't handled cleanly.
    assert isinstance(result.exception, SystemExit)
