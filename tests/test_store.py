"""SQLite scan history (#6)."""
import json
import sqlite3
from pathlib import Path

import pytest
from click.testing import CliRunner

from ai_code_detector import aicd, cli_enhanced
from ai_code_detector.model.aggregator import FileScore, RepoScore
from ai_code_detector.store import DEFAULT_STORE, SCHEMA_VERSION, ScanStore

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _repo(repo_path, probs):
    files = []
    for path, p in probs.items():
        fs = FileScore(file_path=path, ai_probability=p, stylometry_score=p, structural_score=p,
                       feature_explanations={"generic_naming": 0.5}, suspicious_snippets=[])
        fs.language, fs.content_sha256, fs.ml_used = "python", f"sha-{path}", False
        fs.features = {"stylometry": {"generic_name_ratio": 0.5}, "structural": {}}
        fs.explanation = "looks templated" if p > 0.7 else None
        files.append(fs)
    return RepoScore(repo_path=repo_path, ai_probability=0.5, confidence=0.8, stylometry_score=0.4,
                     structural_score=0.3, history_score=0.1, file_scores=files, top_suspicious_files=[],
                     total_files_analyzed=len(files), total_lines_analyzed=42, language_distribution={"python": 2})


def test_insert_and_list_by_repo(tmp_path):
    db_path = tmp_path / "scans.db"
    with ScanStore(db_path) as db:
        assert db.schema_version == SCHEMA_VERSION
        a1 = db.record(_repo("/r/a", {"x.py": 0.9, "y.py": 0.2}), source="/r/a", mode="basic",
                       use_ml=False, use_explanations=False, config={"w": 1})
        db.record(_repo("/r/b", {"z.py": 0.3}), source="https://github.com/o/b", mode="enhanced",
                  use_ml=True, use_explanations=True, config={"w": 1})
        a2 = db.record(_repo("/r/a", {"x.py": 0.4, "y.py": 0.2}), source="/r/a", mode="basic",
                       use_ml=False, use_explanations=False, config={"w": 2})
        scans_a = db.list_scans("/r/a")
        assert [s["id"] for s in scans_a] == [a2, a1]  # newest first, history kept
        assert scans_a[0]["config_fingerprint"] != scans_a[1]["config_fingerprint"]
        assert [s["repo_path"] for s in db.list_scans("https://github.com/o/b")] == ["/r/b"]
        files = db.files(a1)
        assert [(f["path"], f["ai_probability"], f["phase_heuristic"], f["phase_ml"], f["phase_explanation"])
                for f in files] == [("x.py", 0.9, 1, 0, 1), ("y.py", 0.2, 1, 0, 0)]
        assert files[0]["features"]["stylometry"]["generic_name_ratio"] == 0.5
        assert files[0]["content_sha256"] == "sha-x.py"
        v = db.verdicts(a1)
        assert [(x["level"], x["path"]) for x in v] == [("repo", None), ("file", "x.py")]
        assert v[1]["explanation"] == "looks templated" and v[1]["threshold"] == 0.6
        assert [r["ai_probability"] for r in db.file_history("/r/a", "x.py")] == [0.9, 0.4]


def test_reopen_is_idempotent_and_newer_schema_is_refused(tmp_path):
    db_path = tmp_path / "scans.db"
    ScanStore(db_path).close()
    ScanStore(db_path).close()  # CREATE IF NOT EXISTS: no error, no duplicate objects
    con = sqlite3.connect(db_path)
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"scans", "files", "features", "verdicts"} <= tables
    con.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    con.commit()
    con.close()
    with pytest.raises(RuntimeError, match="Upgrade aicd"):
        ScanStore(db_path)


def test_cli_scan_store_appends_runs_without_overwriting(tmp_path):
    db_path = tmp_path / "scans.db"
    run = lambda *a: CliRunner().invoke(aicd.main, ["scan", str(EXAMPLES), "-q", "-f", "json", "-o",
                                                    str(tmp_path / "out"), "--store", str(db_path), *a])
    r1 = run("--mode", "basic")
    assert r1.exit_code == 0, r1.output
    con = sqlite3.connect(db_path)
    first = con.execute("SELECT * FROM files ORDER BY id").fetchall()
    assert sorted(r[2] for r in first) == ["demo.py", "sample_ai_code.py", "sample_human_code.py"]
    r2 = run("--mode", "enhanced")
    assert r2.exit_code == 0, r2.output
    assert con.execute("SELECT COUNT(*) FROM scans").fetchone()[0] == 2
    assert con.execute("SELECT * FROM files WHERE scan_id = 1 ORDER BY id").fetchall() == first
    phases = con.execute("SELECT DISTINCT scan_id, phase_ml FROM files ORDER BY scan_id").fetchall()
    assert phases == [(1, 0), (2, 1)]
    assert all(len(r[0]) == 64 for r in con.execute("SELECT content_sha256 FROM files"))
    src = con.execute("SELECT DISTINCT source FROM scans").fetchall()
    assert src == [(str(EXAMPLES.resolve()),)]

    h = CliRunner().invoke(aicd.main, ["history", "--store", str(db_path), "--repo", str(EXAMPLES), "--json"])
    assert h.exit_code == 0 and [s["mode"] for s in json.loads(h.output)] == ["enhanced", "basic"]
    one = CliRunner().invoke(aicd.main, ["history", "--store", str(db_path), "--scan", "1"])
    assert one.exit_code == 0 and "sample_ai_code.py" in one.output and "[H  ]" in one.output
    missing = CliRunner().invoke(aicd.main, ["history", "--store", str(tmp_path / "nope.db")])
    assert missing.exit_code == 1 and "no scan store" in missing.output


def test_bare_store_flag_defaults_to_cache_path():
    opt = next(p for p in cli_enhanced.main.params if p.name == "store")
    assert opt.flag_value == str(DEFAULT_STORE) and opt.default is None
    assert DEFAULT_STORE.name == "scans.db" and DEFAULT_STORE.parent.name == "ai_code_detector"
