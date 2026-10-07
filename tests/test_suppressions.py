"""`.aicdignore` suppressions (#14): parsing, glob semantics, scan wiring,
exit-code gate, and the suppressed counts in JSON reports."""
import json
import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

from ai_code_detector import cli, cli_agent_scan, cli_enhanced
from ai_code_detector.ingest.file_filter import FileFilter
from ai_code_detector.ingest.suppressions import Suppressions, parse_rules

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _sup(text):
    return Suppressions(rules=parse_rules(text))


def test_parse_rationale_comments_and_negation():
    rules = parse_rules(
        "# header\n\nthird_party/   # vendored, reviewed upstream\n"
        "**/*_pb2.py #protoc\n!keep/me.py\n\\#literal.py\n"
    )
    assert [(r.pattern, r.rationale, r.negate) for r in rules] == [
        ("third_party/", "vendored, reviewed upstream", False),
        ("**/*_pb2.py", "protoc", False),
        ("keep/me.py", "", True),
        ("#literal.py", "", False),
    ]
    assert rules[0].line == 3


@pytest.mark.parametrize("pattern,path,expected", [
    ("third_party/", "third_party/lib/x.py", True),
    ("third_party/", "src/third_party/x.py", True),       # unanchored dir name
    ("third_party/", "third_party.py", False),            # dir-only
    ("/third_party/", "src/third_party/x.py", False),     # anchored
    ("vendor", "a/vendor/b.py", True),
    ("*.gen.py", "deep/er/m.gen.py", True),
    ("*.gen.py", "m.py", False),
    ("build/generated/*.rs", "build/generated/a.rs", True),
    ("build/generated/*.rs", "build/generated/sub/a.rs", False),  # * stays in segment
    ("build/**/*.rs", "build/generated/sub/a.rs", True),
    ("**/*_pb2.py", "x_pb2.py", True),
    ("**/*_pb2.py", "proto/x_pb2.py", True),
    ("src/**", "src/a/b.py", True),
    ("file?.py", "file1.py", True),
    ("file[!0-9].py", "file1.py", False),
    ("file[!0-9].py", "filex.py", True),
])
def test_glob_semantics(pattern, path, expected):
    assert (_sup(pattern).match(path) is not None) is expected


def test_last_match_wins_negation():
    s = _sup("gen/\n!gen/keep.py\n")
    assert s.match("gen/a.py").pattern == "gen/"
    assert s.match("gen/keep.py") is None


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    (root / "third_party" / "lib").mkdir(parents=True)
    shutil.copy(EXAMPLES / "sample_human_code.py", root / "main.py")
    shutil.copy(EXAMPLES / "sample_ai_code.py", root / "third_party" / "lib" / "vendored.py")
    shutil.copy(EXAMPLES / "sample_ai_code.py", root / "api_pb2.py")
    (root / ".aicdignore").write_text(
        "third_party/   # vendored deps\n*_pb2.py       # protoc output\n"
    )
    return root


def test_file_filter_skips_and_records(repo):
    ff = FileFilter([".py"], [".git"])
    files = ff.scan_directory(repo)
    assert [f.relative_path.as_posix() for f in files] == ["main.py"]
    d = ff.last_suppressed.to_dict()
    assert d["count"] == 2
    assert d["paths"] == ["api_pb2.py", "third_party/lib/vendored.py"]
    assert {r["pattern"]: (r["count"], r["rationale"]) for r in d["rules"]} == {
        "third_party/": (1, "vendored deps"),
        "*_pb2.py": (1, "protoc output"),
    }
    assert d["source"].endswith(".aicdignore")

    ff.use_suppressions = False
    assert len(ff.scan_directory(repo)) == 3
    assert ff.last_suppressed.to_dict()["count"] == 0


def _scan_json(entry, target, out, *extra):
    args = [str(target), "-f", "json", "-q", "-o", str(out), *extra]
    if entry is cli_enhanced.main:
        args += ["--mode", "basic"]
    r = CliRunner().invoke(entry, args)
    assert r.exit_code in (0, 1, 2), r.output
    report = next(out.glob("*.json"))
    return r.exit_code, json.loads(report.read_text())


@pytest.mark.parametrize("entry", [cli.main, cli_enhanced.main], ids=["basic-cli", "scan"])
def test_gate_equals_scan_without_suppressed_files(entry, repo, tmp_path):
    # Reference: the same repo with the suppressed files physically removed.
    ref = tmp_path / "ref"
    ref.mkdir()
    shutil.copy(repo / "main.py", ref / "main.py")
    code_ref, rep_ref = _scan_json(entry, ref, tmp_path / "o_ref")
    code, rep = _scan_json(entry, repo, tmp_path / "o")

    assert code == code_ref
    assert rep["summary"]["ai_probability"] == rep_ref["summary"]["ai_probability"]
    assert rep["statistics"]["total_files"] == 1
    assert rep["suppressed"]["count"] == 2
    assert rep_ref["suppressed"]["count"] == 0
    listed = {d["path"] for d in rep.get("file_details", [])} | set(map(str, rep.get("top_suspicious_files", [])))
    assert not any("third_party" in p or "_pb2" in p for p in listed)

    # --no-suppressions brings them back into the gate and the counts.
    _, rep_all = _scan_json(entry, repo, tmp_path / "o_all", "--no-suppressions")
    assert rep_all["statistics"]["total_files"] == 3
    assert rep_all["suppressed"]["count"] == 0


def test_explicit_suppressions_file_and_all_suppressed(repo, tmp_path):
    rules = tmp_path / "ci.aicdignore"
    rules.write_text("*.py  # everything\n")
    code, rep = _scan_json(cli_enhanced.main, repo, tmp_path / "o", "--suppressions", str(rules))
    assert code == 0
    assert rep["suppressed"]["count"] == 3
    assert rep["statistics"]["total_files"] == 0


def test_agent_scan_reports_suppressed(repo, tmp_path):
    out = tmp_path / "findings"
    r = CliRunner().invoke(cli_agent_scan.main, [str(repo), "-o", str(out), "-q"])
    assert r.exit_code == 0, r.output
    payload = json.loads((out / "ai_scan_findings.json").read_text())
    assert payload["suppressed"]["count"] == 2
    assert not any("third_party" in (f["file"] or "") for f in payload["findings"])
    assert "Suppressed:** 2 files" in (out / "AI_SCAN_FINDINGS.md").read_text()


def test_missing_explicit_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        Suppressions.load(tmp_path, tmp_path / "nope")
