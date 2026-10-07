"""SARIF 2.1.0 export (#13): schema-valid output for a fixture repo, verdict →
level mapping, stable fingerprints, and agent-scan findings."""
import json
import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

from ai_code_detector import aicd
from ai_code_detector.model.findings import Finding, ScanFindings
from ai_code_detector.report.reporter_sarif import (
    FILE_RULE_ID, SARIF_VERSION, FindingsSarifWriter, probability_level,
)

jsonschema = pytest.importorskip("jsonschema")

HERE = Path(__file__).resolve().parent
EXAMPLES = HERE.parent / "examples"
SCHEMA = json.loads((HERE / "fixtures" / "sarif-schema-2.1.0.json").read_text())


def _validate(doc):
    jsonschema.Draft7Validator(SCHEMA).validate(doc)
    assert doc["version"] == SARIF_VERSION == "2.1.0"
    assert doc["$schema"].endswith("sarif-2.1.0.json")


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "fixture_repo"
    (root / "pkg").mkdir(parents=True)
    shutil.copy(EXAMPLES / "sample_ai_code.py", root / "pkg" / "generated.py")
    shutil.copy(EXAMPLES / "sample_human_code.py", root / "human.py")
    (root / "debt.py").write_text("def f(x):\n    # TODO: handle negatives\n    return x\n")
    return root


def _scan_sarif(repo, out, *extra):
    r = CliRunner().invoke(aicd.main, ["scan", str(repo), "-f", "sarif", "-q", "-o", str(out), *extra])
    assert r.exit_code in (0, 1, 2), r.output
    return json.loads((out / "ai_detection_report.sarif").read_text())


@pytest.mark.parametrize("mode", ["basic", "enhanced"])
def test_scan_sarif_is_schema_valid(repo, tmp_path, mode):
    doc = _scan_sarif(repo, tmp_path / "o", "--mode", mode, "--sarif-min-probability", "0")
    _validate(doc)
    run = doc["runs"][0]
    assert run["tool"]["driver"]["name"] == "aicd"
    assert [r["id"] for r in run["tool"]["driver"]["rules"]] == [FILE_RULE_ID]
    uris = {res["locations"][0]["physicalLocation"]["artifactLocation"]["uri"] for res in run["results"]}
    assert uris == {"pkg/generated.py", "human.py", "debt.py"}  # relative, posix, every file at min=0
    for res in run["results"]:
        p = res["properties"]["ai_probability"]
        assert res["level"] == probability_level(p, 0.0)
        assert res["locations"][0]["physicalLocation"]["artifactLocation"]["uriBaseId"] == "%SRCROOT%"
        assert "AI probability" in res["message"]["text"]
    assert run["properties"]["files_analyzed"] == 3


def test_min_probability_filters_and_fingerprints_are_stable(repo, tmp_path):
    a = _scan_sarif(repo, tmp_path / "a", "--mode", "basic", "--sarif-min-probability", "0")
    b = _scan_sarif(repo, tmp_path / "b", "--mode", "basic", "--sarif-min-probability", "0")
    fp = lambda d: sorted(r["partialFingerprints"]["aicdFile/v1"] for r in d["runs"][0]["results"])
    assert fp(a) == fp(b) and len(set(fp(a))) == 3
    none = _scan_sarif(repo, tmp_path / "c", "--mode", "basic", "--sarif-min-probability", "1.0")
    assert none["runs"][0]["results"] == [] or all(
        r["properties"]["ai_probability"] >= 1.0 for r in none["runs"][0]["results"])
    _validate(none)


@pytest.mark.parametrize("p,level", [
    (0.95, "error"), (0.8, "error"), (0.79, "warning"), (0.6, "warning"),
    (0.59, "note"), (0.4, "note"), (0.39, None),
])
def test_probability_level_mapping(p, level):
    assert probability_level(p) == level


def test_format_all_writes_every_report(repo, tmp_path):
    out = tmp_path / "all"
    r = CliRunner().invoke(aicd.main, ["scan", str(repo), "--mode", "basic", "-f", "all", "-q", "-o", str(out)])
    assert r.exit_code in (0, 1, 2), r.output
    assert {p.name for p in out.iterdir()} >= {
        "ai_detection_report_enhanced.json", "ai_detection_report_enhanced.md", "ai_detection_report.sarif"}


def test_agent_scan_sarif(repo, tmp_path):
    out = tmp_path / "agent"
    r = CliRunner().invoke(aicd.main, ["agent-scan", str(repo), "-o", str(out), "-q", "--sarif"])
    assert r.exit_code == 0, r.output
    doc = json.loads((out / "ai_scan_findings.sarif").read_text())
    _validate(doc)
    run = doc["runs"][0]
    satd = [x for x in run["results"] if x["ruleId"] == "aicd/agent/satd"]
    assert satd, run["results"]
    region = satd[0]["locations"][0]["physicalLocation"]["region"]
    assert satd[0]["locations"][0]["physicalLocation"]["artifactLocation"]["uri"] == "debt.py"
    assert region["startLine"] == 2 and satd[0]["level"] == "note"
    rule_ids = [rl["id"] for rl in run["tool"]["driver"]["rules"]]
    for x in run["results"]:
        assert rule_ids[x["ruleIndex"]] == x["ruleId"]
    # Without --sarif nothing extra is written.
    out2 = tmp_path / "agent2"
    CliRunner().invoke(aicd.main, ["agent-scan", str(repo), "-o", str(out2), "-q"])
    assert not (out2 / "ai_scan_findings.sarif").exists()


def test_repo_level_findings_go_to_run_properties():
    findings = ScanFindings(repo_path="/r", ai_probability=0.7, findings=[
        Finding(type="ai_attribution", file="(commit history)", severity="high", description="Co-authored-by bot"),
        Finding(type="performance_hotspot", file="a/b.py", line=10, function="slow",
                severity="warning", description="nested loops", evidence={"risk": 0.8}),
    ])
    doc = FindingsSarifWriter().build(findings)
    _validate(doc)
    run = doc["runs"][0]
    assert [r["ruleId"] for r in run["results"]] == ["aicd/agent/performance_hotspot"]
    assert run["results"][0]["level"] == "warning"
    assert run["results"][0]["properties"]["evidence"] == {"risk": 0.8}
    assert run["properties"]["repositoryFindings"][0]["type"] == "ai_attribution"
