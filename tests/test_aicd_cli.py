"""Unified `aicd` CLI (#9): help, scan parity with cli_enhanced, agent-scan,
explain, train, and the deprecated entrypoints."""
import json
import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

from ai_code_detector import aicd, cli_enhanced

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


@pytest.fixture
def repo(tmp_path):
    dst = tmp_path / "repo"
    dst.mkdir()
    for name in ("sample_ai_code.py", "sample_human_code.py"):
        shutil.copy(EXAMPLES / name, dst / name)
    return dst


def test_help_lists_subcommands():
    r = CliRunner().invoke(aicd.main, ["--help"])
    assert r.exit_code == 0
    for sub in aicd.SUBCOMMANDS:
        assert sub in r.output


def _strip_volatile(report):
    report = json.loads(json.dumps(report))
    for key in ("timestamp", "generated_at", "analysis_time", "duration_seconds"):
        report.pop(key, None)
        report.get("metadata", {}).pop(key, None)
    return report


@pytest.mark.parametrize("mode", ["basic", "enhanced"])
def test_scan_matches_cli_enhanced(repo, tmp_path, mode):
    runner = CliRunner()
    a, b = tmp_path / "aicd", tmp_path / "legacy"
    ra = runner.invoke(aicd.main, ["scan", str(repo), "--mode", mode, "-f", "json", "-q", "-o", str(a)])
    rb = runner.invoke(cli_enhanced.main, [str(repo), "--mode", mode, "-f", "json", "-q", "-o", str(b)])
    assert ra.exit_code == rb.exit_code and ra.exit_code in (0, 1, 2), ra.output
    ja = json.loads((a / "ai_detection_report_enhanced.json").read_text())
    jb = json.loads((b / "ai_detection_report_enhanced.json").read_text())
    assert _strip_volatile(ja) == _strip_volatile(jb)


def test_agent_scan(repo, tmp_path):
    out = tmp_path / "findings"
    r = CliRunner().invoke(aicd.main, ["agent-scan", str(repo), "-o", str(out), "-q"])
    assert r.exit_code == 0, r.output
    assert (out / "ai_scan_findings.json").exists()


def test_explain_json(repo):
    r = CliRunner().invoke(aicd.main, ["explain", str(repo / "sample_ai_code.py"), "--json", "--no-ml"])
    assert r.exit_code == 0, r.output
    data = json.loads(r.output)
    assert 0.0 <= data["ai_probability"] <= 1.0
    assert data["explanation"]


def test_train_then_explain_with_model(repo, tmp_path):
    ds = tmp_path / "labels.jsonl"
    ds.write_text(
        json.dumps({"path": str(repo / "sample_ai_code.py"), "label": 1}) + "\n"
        + json.dumps({"path": str(repo / "sample_human_code.py"), "label": 0}) + "\n"
    )
    model = tmp_path / "model.json"
    r = CliRunner().invoke(aicd.main, ["train", str(ds), "-o", str(model), "--epochs", "5"])
    assert r.exit_code == 0, r.output
    assert json.loads(model.read_text())["feature_dim"] == 23
    r = CliRunner().invoke(aicd.main, ["explain", str(repo / "sample_ai_code.py"), "--model", str(model), "--json"])
    assert r.exit_code == 0, r.output


def test_train_rejects_bad_dataset(tmp_path):
    ds = tmp_path / "bad.jsonl"
    ds.write_text('{"path": "nope.py", "label": 1}\n')
    r = CliRunner().invoke(aicd.main, ["train", str(ds), "-o", str(tmp_path / "m.json")])
    assert r.exit_code != 0 and "no such file" in r.output


def test_legacy_entrypoints_warn_and_still_work(repo, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("sys.argv", ["ai-code-detector-agent-scan", str(repo), "-o", str(tmp_path / "o"), "-q"])
    with pytest.raises(SystemExit) as exc:
        aicd.legacy_agent_scan()
    assert exc.value.code == 0
    assert "deprecated; use `aicd agent-scan`" in capsys.readouterr().err
    # Old modules stay importable for one release.
    from ai_code_detector import cli, cli_agent_scan  # noqa: F401


def test_module_main_dispatches_subcommands_and_legacy_form(repo, tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        aicd.module_main(["--help"])
    assert exc.value.code == 0 and "agent-scan" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        aicd.module_main([str(repo), "-q", "-f", "json", "-o", str(tmp_path / "old")])
    assert "deprecated" in capsys.readouterr().err
    assert (tmp_path / "old" / "ai_detection_report.json").exists()
