"""Integration test for the agent-prep scan orchestrator."""

import sys
import subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent_scan import AgentPrepScanner


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
