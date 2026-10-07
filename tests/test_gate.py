"""`aicd gate` + PR Action + pre-commit hook (#12)."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from ai_code_detector import aicd
from ai_code_detector.gate import DEFAULT_THRESHOLD, run_gate

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
FIXTURE = HERE / "fixtures" / "gate_pr"
FIXTURE_PATHS = ["generated_helper.py", "hand_written.py", "third_party/vendored.py", "README.md"]


def test_run_gate_on_fixture_pr():
    r = run_gate(FIXTURE_PATHS + ["deleted.py"], FIXTURE, threshold=0.4)
    assert [f["path"] for f in r.flagged] == ["generated_helper.py"]
    assert {f["path"] for f in r.files} == {"generated_helper.py", "hand_written.py"}
    assert r.suppressed == [{"path": "third_party/vendored.py", "pattern": "third_party/"}]
    assert {s["path"] for s in r.skipped} == {"README.md", "deleted.py"}
    assert all(0.0 <= f["ai_probability"] <= 1.0 for f in r.files)


def _gate(args, env=None, input=None):
    return CliRunner().invoke(aicd.main, ["gate", *args], env=env or {}, input=input)


def test_cli_exit_codes_reports_and_summary(tmp_path):
    summary = tmp_path / "summary.md"
    summary.write_text("previous step\n")
    res = _gate(["--root", str(FIXTURE), *FIXTURE_PATHS, "-t", "0.4", "--json", str(tmp_path / "r.json"),
                 "--sarif", str(tmp_path / "r.sarif"), "--summary", str(summary)])
    assert res.exit_code == 1, res.output
    data = json.loads((tmp_path / "r.json").read_text())
    assert data["passed"] is False and data["flagged"] == ["generated_helper.py"] and data["threshold"] == 0.4
    text = summary.read_text()
    assert text.startswith("previous step\n")  # appended, not overwritten
    assert "🚩 | `generated_helper.py`" in text and "✅ | `hand_written.py`" in text
    sarif = json.loads((tmp_path / "r.sarif").read_text())
    assert sarif["version"] == "2.1.0"
    uris = {x["locations"][0]["physicalLocation"]["artifactLocation"]["uri"] for x in sarif["runs"][0]["results"]}
    assert "generated_helper.py" in uris and "third_party/vendored.py" not in uris

    ok = _gate(["--root", str(FIXTURE), *FIXTURE_PATHS, "-t", "0.9", "-q"])
    assert ok.exit_code == 0 and ok.output == ""  # quiet + passing = silent


def test_threshold_from_env_and_default():
    flagged = _gate(["--root", str(FIXTURE), "generated_helper.py", "-q"], env={"AICD_THRESHOLD": "0.1"})
    assert flagged.exit_code == 1 and "above 0.10" in flagged.output  # quiet still prints when failing
    default = _gate(["--root", str(FIXTURE), "generated_helper.py", "--json", "/dev/stdout", "-q"],
                    env={"AICD_THRESHOLD": ""})
    assert default.exit_code == 0 and DEFAULT_THRESHOLD == 0.7
    bad = _gate(["--root", str(FIXTURE), "generated_helper.py"], env={"AICD_THRESHOLD": "high"})
    assert bad.exit_code == 2


def test_paths_from_stdin_and_nothing_to_scan():
    res = _gate(["--root", str(FIXTURE), "--paths-from", "-", "-t", "0.4", "-q"],
                input="generated_helper.py\n\nhand_written.py\n")
    assert res.exit_code == 1
    empty = _gate(["--root", str(FIXTURE), "-q"])
    assert empty.exit_code == 0


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True,
                   env={"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
                        "GIT_COMMITTER_EMAIL": "t@t", "PATH": "/usr/bin:/bin"})


def test_simulated_pull_request_diff(tmp_path):
    """Mirror action.yml: diff base...HEAD for source pathspecs, then gate those files."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    shutil.copy(FIXTURE / "hand_written.py", repo / "untouched.py")  # exists on base, not in the PR
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "base")
    _git(repo, "checkout", "-qb", "pr")
    for rel in FIXTURE_PATHS + [".aicdignore"]:
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(FIXTURE / rel, repo / rel)
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "pr")
    diff = subprocess.run(["git", "diff", "--name-only", "--diff-filter=ACMR", "main...HEAD", "--",
                           "*.py", "*.ts", "*.tsx", "*.js", "*.jsx", "*.go", "*.rs"],
                          cwd=repo, check=True, capture_output=True, text=True).stdout.split()
    assert sorted(diff) == ["generated_helper.py", "hand_written.py", "third_party/vendored.py"]
    r = run_gate(diff, repo, threshold=0.4)
    assert [f["path"] for f in r.flagged] == ["generated_helper.py"]
    assert "untouched.py" not in {f["path"] for f in r.files}
    assert [s["path"] for s in r.suppressed] == ["third_party/vendored.py"]


def test_workflow_action_and_hook_are_wired():
    wf = yaml.safe_load((REPO / ".github" / "workflows" / "ai-scan.yml").read_text())
    trig = wf.get("on") or wf.get(True)
    paths = trig["pull_request"]["paths"]
    for ext in ("py", "ts", "js", "go", "rs"):
        assert f"**/*.{ext}" in paths
    step = next(s for s in wf["jobs"]["gate"]["steps"] if s.get("uses") == "./")
    assert step["env"]["AICD_THRESHOLD"] == "${{ vars.AICD_THRESHOLD }}"

    action = yaml.safe_load((REPO / "action.yml").read_text())
    assert action["runs"]["using"] == "composite"
    run_text = "\n".join(s.get("run", "") for s in action["runs"]["steps"])
    assert "aicd gate --paths-from" in run_text and "set -f" in run_text
    assert any(s.get("uses", "").startswith("actions/upload-artifact") for s in action["runs"]["steps"])

    hooks = yaml.safe_load((REPO / ".pre-commit-hooks.yaml").read_text())
    hook = next(h for h in hooks if h["id"] == "aicd-gate")
    assert hook["entry"].startswith("aicd gate") and hook["pass_filenames"] is True
