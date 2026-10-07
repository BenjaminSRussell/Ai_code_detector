"""Incremental scan cache keyed by content hash + config fingerprint (#8)."""
import json
import shutil
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from ai_code_detector import aicd
from ai_code_detector.cache import analysis_key, score_from_dict, score_to_dict
from ai_code_detector.model.aggregator import FileScore

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "src" / "ai_code_detector" / "configs" / "default.yaml"


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    for f in (REPO / "examples").glob("*.py"):
        shutil.copy(f, r / f.name)
    return r


def _runner():
    try:
        return CliRunner(mix_stderr=False)  # click < 8.2
    except TypeError:
        return CliRunner()  # click >= 8.2 always captures stderr separately


def scan(tmp_path, repo, *extra):
    out = tmp_path / "out"
    res = _runner().invoke(
        aicd.main, ["scan", str(repo), "--mode", "basic", "-q", "-f", "json", "-o", str(out),
                    "--incremental", "--store", str(tmp_path / "cache.db"), *extra])
    assert res.exit_code == 0, res.output + res.stderr
    report = json.loads((out / "ai_detection_report_enhanced.json").read_text())
    return report, res.stderr


def scores(report):
    return {f["path"]: f["ai_probability"] for f in report["file_details"]}


def test_unchanged_rescan_recomputes_nothing_and_matches(tmp_path, repo):
    first, log1 = scan(tmp_path, repo)
    assert first["cache"]["recomputed"] == 3 and first["cache"]["cached"] == 0
    second, log2 = scan(tmp_path, repo)
    assert second["cache"]["recomputed"] == 0 and second["cache"]["cached"] == 3
    assert "Incremental: 0 recomputed, 3 from cache" in log2  # logged
    assert scores(second) == scores(first)
    assert second["summary"]["ai_probability"] == first["summary"]["ai_probability"]
    assert second["file_details"] == first["file_details"]


def test_touching_one_file_recomputes_only_that_path(tmp_path, repo):
    scan(tmp_path, repo)
    with open(repo / "demo.py", "a") as fh:
        fh.write("\n# edited\n")
    report, _ = scan(tmp_path, repo)
    assert report["cache"]["recomputed_paths"] == ["demo.py"] and report["cache"]["cached"] == 2


def test_rename_is_a_cache_hit(tmp_path, repo):
    scan(tmp_path, repo)
    (repo / "demo.py").rename(repo / "renamed_demo.py")
    report, _ = scan(tmp_path, repo)
    assert report["cache"]["recomputed"] == 0 and "renamed_demo.py" in scores(report)


def test_config_weight_change_invalidates_cache(tmp_path, repo):
    scan(tmp_path, repo)
    cfg = yaml.safe_load(DEFAULT_CONFIG.read_text())
    cfg.setdefault("scoring", {}).setdefault("weights", {})["stylometry"] = 0.9
    changed = tmp_path / "changed.yaml"
    changed.write_text(yaml.safe_dump(cfg))
    report, _ = scan(tmp_path, repo, "--config", str(changed))
    assert report["cache"]["recomputed"] == 3 and report["cache"]["cached"] == 0
    again, _ = scan(tmp_path, repo, "--config", str(changed))
    assert again["cache"]["recomputed"] == 0  # the new config has its own cache entries
    back, _ = scan(tmp_path, repo)
    assert back["cache"]["recomputed"] == 0  # and the original config's entries are still valid


def test_force_full_recomputes_and_requires_incremental(tmp_path, repo):
    scan(tmp_path, repo)
    report, log = scan(tmp_path, repo, "--force-full")
    assert report["cache"]["recomputed"] == 3 and "(--force-full)" in log
    bad = CliRunner().invoke(aicd.main, ["scan", str(repo), "-q", "--force-full", "-o", str(tmp_path / "o")])
    assert bad.exit_code == 2 and "--incremental" in bad.output


def test_non_incremental_scan_has_cache_disabled(tmp_path, repo):
    out = tmp_path / "plain"
    res = CliRunner().invoke(aicd.main, ["scan", str(repo), "--mode", "basic", "-q", "-f", "json", "-o", str(out)])
    assert res.exit_code == 0
    assert json.loads((out / "ai_detection_report_enhanced.json").read_text())["cache"] == {"enabled": False}


def test_analysis_key_covers_detector_config_phases_and_model(tmp_path):
    base = dict(detector_version="0.3.0", config_fingerprint="abc", use_ml=True, use_explanations=False,
                embedder="hash", explainer="template", feature_dim=23)
    k = analysis_key(**base)
    assert k == analysis_key(**base)
    for change in ({"detector_version": "0.4.0"}, {"config_fingerprint": "abd"}, {"use_ml": False},
                   {"use_explanations": True}, {"embedder": "mlx"}, {"feature_dim": 24}):
        assert analysis_key(**{**base, **change}) != k, change
    assert analysis_key(**{**base, "explainer": "qwen"}) == k
    assert analysis_key(**base, parsers={"backend": "auto", "go": True}) != \
        analysis_key(**base, parsers={"backend": "auto", "go": False})  # grammar install changes results  # explainer irrelevant when explanations are off
    m = tmp_path / "m.json"
    m.write_text("{}")
    k1 = analysis_key(**base, ml_model_path=m)
    m.write_text('{"retrained": 1}')
    assert analysis_key(**base, ml_model_path=m) != k1


def test_score_round_trip_keeps_fields_and_extras():
    fs = FileScore(file_path="a.py", ai_probability=0.5, stylometry_score=0.4, structural_score=0.3,
                   feature_explanations={"generic_naming": 0.6}, suspicious_snippets=[], parse_failed=False)
    fs.explanation, fs.language, fs.features, fs.ml_used = "why", "python", {"stylometry": {}}, False
    back = score_from_dict(json.loads(json.dumps(score_to_dict(fs))), "b/a.py", "sha")
    assert back.file_path == "b/a.py" and back.content_sha256 == "sha"
    assert (back.ai_probability, back.feature_explanations, back.explanation, back.language) == \
        (0.5, {"generic_naming": 0.6}, "why", "python")
