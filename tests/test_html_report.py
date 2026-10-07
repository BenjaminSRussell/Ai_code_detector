"""Self-contained HTML findings viewer + additive feature contributions (#10)."""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner

from ai_code_detector import aicd
from ai_code_detector.model.aggregator import FileScore, HeuristicAggregator, RepoScore
from ai_code_detector.report.reporter_html import HTMLReporter

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "examples"
DATA_RE = re.compile(r'<script type="application/json" id="aicd-data">(.*?)</script>', re.S)


def _data(page: str):
    return json.loads(DATA_RE.search(page).group(1))


def _scan(tmp_path, *extra, source=EXAMPLES):
    out = tmp_path / "out" / "report.html"
    r = CliRunner().invoke(aicd.main, ["scan", str(source), "-q", "-f", "json", "-o", str(tmp_path / "out"),
                                       "--html", str(out), *extra])
    assert r.exit_code == 0, r.output
    return out.read_text(encoding="utf-8")


def test_scan_examples_writes_openable_self_contained_report(tmp_path):
    page = _scan(tmp_path, "--mode", "basic")
    assert page.startswith("<!doctype html>")
    assert not re.search(r'<(script|link|img)[^>]+(src|href)=', page)  # nothing loaded from anywhere
    assert "http://" not in page and "https://" not in page
    d = _data(page)
    assert d["mode"] == "basic" and d["summary"]["total_files"] == 3 and d["summary"]["total_lines"] > 200
    probs = [f["ai_probability"] for f in d["files"]]
    assert probs == sorted(probs, reverse=True) and d["files"][0]["path"] == "sample_ai_code.py"
    report = json.loads((tmp_path / "out" / "ai_detection_report_enhanced.json").read_text())
    by_path = {f["path"]: f for f in report["file_details"]}
    for f in d["files"]:
        # No fabricated scores: viewer numbers are the scan's numbers.
        assert f["ai_probability"] == pytest.approx(by_path[f["path"]]["ai_probability"], abs=5e-4)
        assert sum(c["share"] for c in f["contributions"]) == pytest.approx(f["ai_probability"], abs=1e-3)
    top = d["files"][0]["contributions"][:5]
    assert len(top) == 5 and [c["share"] for c in top] == sorted((c["share"] for c in top), reverse=True)
    assert [t["feature"] for t in by_path["sample_ai_code.py"]["top_features"]] == [c["feature"] for c in top]


def test_enhanced_mode_keeps_contributions_additive(tmp_path):
    d = _data(_scan(tmp_path, "--mode", "enhanced"))
    for f in d["files"]:
        feats = {c["feature"] for c in f["contributions"]}
        assert "ml_classifier" in feats
        assert sum(c["share"] for c in f["contributions"]) == pytest.approx(f["ai_probability"], abs=1e-3)


def test_empty_scan_renders_empty_state(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    page = _scan(tmp_path, "--mode", "basic", source=empty)
    d = _data(page)
    assert d["files"] == [] and d["summary"]["verdict"] == "No files scanned"
    assert "No files were scanned" in page


def _repo(paths_and_probs, explanation=None):
    files = []
    for p, prob in paths_and_probs:
        fs = FileScore(file_path=p, ai_probability=prob, stylometry_score=prob, structural_score=prob,
                       feature_explanations={}, suspicious_snippets=[],
                       feature_contributions={"generic_naming": prob})
        fs.explanation = explanation
        files.append(fs)
    return RepoScore(repo_path="r", ai_probability=0.5, confidence=0.5, stylometry_score=0.5,
                     structural_score=0.5, history_score=0.0, file_scores=files,
                     top_suspicious_files=[], total_files_analyzed=len(files), total_lines_analyzed=10,
                     language_distribution={"python": len(files)})


def test_hostile_paths_cannot_break_out_of_the_data_block():
    evil = '</script><img src=x onerror=alert(1)>.py'
    page = HTMLReporter(threshold=0.25).render(_repo([(evil, 0.9)], explanation="<b>x</b> & \u2028"))
    assert "</script><img" not in page
    d = _data(page)
    assert d["files"][0]["path"] == evil and d["files"][0]["explanation"].startswith("<b>")
    assert d["default_threshold"] == 0.25


def test_aggregator_contributions_sum_and_cap():
    from ai_code_detector.analysis.metrics_stylometry import StylometricFeatures
    from ai_code_detector.analysis.metrics_structural import StructuralFeatures
    import dataclasses

    def maxed(cls, **over):
        vals = {}
        for f in dataclasses.fields(cls):
            vals[f.name] = 2.0  # out-of-range scores push the raw sum past the 1.0 cap
        vals.update(over)
        return cls(**vals)

    agg = HeuristicAggregator()
    sty = maxed(StylometricFeatures, identifier_entropy=0.0, trailing_whitespace_ratio=0.0)
    st = maxed(StructuralFeatures, avg_cyclomatic_complexity=1.0, complexity_to_docstring_ratio=100.0)
    assert sum(agg._structural_terms(st).values()) > 1.0  # structural raw terms exceed the cap
    fs = agg.aggregate_file_features(sty, st)
    assert fs.ai_probability == pytest.approx(1.0)
    assert sum(fs.feature_contributions.values()) == pytest.approx(fs.ai_probability)
    assert agg._score_structural(st) == 1.0


def _script(page):
    return re.findall(r"<script>(.*?)</script>", page, re.S)[0]


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_viewer_javascript_parses(tmp_path):
    js = tmp_path / "viewer.js"
    js.write_text(_script(HTMLReporter().render(_repo([("a.py", 0.9)]))))
    subprocess.run(["node", "--check", str(js)], check=True)


CHROME = next((c for c in ("google-chrome", "chromium", "chromium-browser") if shutil.which(c)), None)


@pytest.mark.skipif(CHROME is None, reason="no headless Chrome")
def test_threshold_query_param_filters_table_in_browser(tmp_path):
    out = tmp_path / "r.html"
    HTMLReporter().generate(_repo([("high.py", 0.9), ("mid.py", 0.5), ("low.py", 0.1)]), out)

    def dom(query):
        try:
            res = subprocess.run([CHROME, "--headless=new", "--no-sandbox", "--disable-gpu", "--dump-dom",
                                  out.as_uri() + query], capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            pytest.skip("headless Chrome timed out")
        if res.returncode != 0 or "<tbody" not in res.stdout and "empty" not in res.stdout:
            pytest.skip(f"headless Chrome unusable: {res.stderr[-200:]}")
        return res.stdout

    all_rows = dom("")
    assert all(f'data-path="{p}"' in all_rows for p in ("high.py", "mid.py", "low.py"))
    filtered = dom("?threshold=0.6")
    assert 'data-path="high.py"' in filtered and 'data-path="mid.py"' not in filtered
    assert "1 of 3 files at or above 60.0%" in filtered
    detail = dom("#file=mid.py")
    assert "Top contributing features" in detail and "Generic identifier names" in detail
