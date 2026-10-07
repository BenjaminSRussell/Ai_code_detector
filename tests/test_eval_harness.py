"""Labeled eval harness (#11)."""
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import pytest
from click.testing import CliRunner

from ai_code_detector import aicd
from ai_code_detector.evaluation import (ManifestError, calibration, confusion, load_manifest, roc_auc,
                                         roc_curve)

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "datasets" / "manifest.yaml"


def test_roc_auc_matches_sklearn_and_edge_cases():
    sk = pytest.importorskip("sklearn.metrics")
    rng = random.Random(7)
    y = [rng.randint(0, 1) for _ in range(200)]
    p = [round(rng.random() * 0.5 + 0.4 * t, 2) for t in y]  # rounded -> plenty of ties
    assert roc_auc(y, p) == pytest.approx(sk.roc_auc_score(y, p))
    assert roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert roc_auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert roc_auc([0, 1], [0.5, 0.5]) == 0.5
    assert roc_auc([1, 1], [0.2, 0.3]) is None


def test_roc_curve_runs_from_origin_to_one():
    pts = roc_curve([0, 1, 0, 1], [0.1, 0.4, 0.35, 0.8])
    assert (pts[0]["fpr"], pts[0]["tpr"], pts[0]["threshold"]) == (0.0, 0.0, None)
    assert (pts[-1]["fpr"], pts[-1]["tpr"]) == (1.0, 1.0)
    assert [p["threshold"] for p in pts[1:]] == [0.8, 0.4, 0.35, 0.1]
    assert pts[3] == {"threshold": 0.35, "fpr": 0.5, "tpr": 1.0}


def test_confusion_and_calibration():
    y = [0, 0, 1, 1]
    p = [0.1, 0.6, 0.7, 1.0]
    c = confusion(y, p, 0.5)
    assert (c["tp"], c["fp"], c["tn"], c["fn"]) == (2, 1, 1, 0)
    assert c["precision"] == pytest.approx(2 / 3) and c["recall"] == 1.0 and c["accuracy"] == 0.75
    cal = calibration(y, p, bins=2)
    lo, hi = cal["bins"]
    assert lo["count"] == 1 and lo["fraction_ai"] == 0.0
    assert hi["count"] == 3 and hi["mean_predicted"] == pytest.approx(0.7667, abs=1e-3)  # 1.0 lands in the top bin
    assert cal["ece"] == pytest.approx(0.25 * 0.1 + 0.75 * abs(2.3 / 3 - 2 / 3))
    assert cal["brier"] == pytest.approx((0.01 + 0.36 + 0.09 + 0.0) / 4)


def _manifest(tmp_path, body):
    (tmp_path / "h").mkdir(exist_ok=True)
    (tmp_path / "a").mkdir(exist_ok=True)
    for i in range(3):
        (tmp_path / "h" / f"h{i}.py").write_text(f"x{i} = {i}\n")
        (tmp_path / "a" / f"a{i}.py").write_text(f"y{i} = {i}\n")
    m = tmp_path / "m.yaml"
    m.write_text(body)
    return m


def test_manifest_globs_auto_split_and_errors(tmp_path):
    m = _manifest(tmp_path, "test_fraction: 0.5\nentries:\n"
                            "  - {glob: 'h/*.py', label: human}\n  - {glob: 'a/*.py', label: 1, split: test}\n")
    entries = load_manifest(m)
    assert len(entries) == 6 and sum(e.label for e in entries) == 3
    assert all(e.split == "test" for e in entries if e.label == 1)
    assert [e.split for e in load_manifest(m)] == [e.split for e in entries]  # deterministic
    assert {e.rel for e in entries} >= {"h/h0.py", "a/a2.py"}
    for body, msg in [("entries:\n  - {path: h/nope.py, label: ai}\n", "no such file"),
                      ("entries:\n  - {path: h/h0.py, label: robot}\n", "label must be"),
                      ("entries:\n  - {glob: 'zzz/*.py', label: ai}\n", "matched nothing"),
                      ("entries:\n  - {label: ai}\n", "needs `path` or `glob`"),
                      ("entries:\n  - {path: h/h0.py, label: ai, split: dev}\n", "split must be"),
                      ("name: x\n", "entries")]:
        with pytest.raises(ManifestError, match=msg):
            load_manifest(_manifest(tmp_path, body))


def test_script_on_fixture_manifest_writes_metrics(tmp_path):
    out = tmp_path / "eval"
    t0 = time.monotonic()
    proc = subprocess.run([sys.executable, str(REPO / "scripts" / "eval_harness.py"), "--manifest", str(MANIFEST),
                           "--out", str(out), "--train", "-q"], capture_output=True, text=True, cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert time.monotonic() - t0 < 30
    data = json.loads((out / "metrics.json").read_text(), parse_constant=lambda c: pytest.fail(f"non-JSON {c}"))
    assert data["dataset"] == {"n": 12, "n_ai": 6, "n_human": 6, "train": 8, "test": 4}
    basic, trained = data["scorers"]["basic"], data["scorers"]["trained"]
    assert basic["evaluated_on"] == "all" and basic["n"] == 12
    assert trained["evaluated_on"] == "test" and trained["n"] == 4 and trained["n_train"] == 8
    for m in (basic, trained):
        assert 0.0 <= m["roc_auc"] <= 1.0 and 0.0 <= m["brier"] <= 1.0 and 0.0 <= m["ece"] <= 1.0
        assert set(m["confusion"]) >= {"tp", "fp", "tn", "fn", "precision", "recall", "f1"}
    assert data["caveats"]
    for name in ("predictions.csv", "roc_basic.csv", "calibration_basic.csv", "roc_trained.csv",
                 "summary.md", "model.json"):
        assert (out / name).is_file(), name
    rows = (out / "predictions.csv").read_text().splitlines()
    assert rows[0] == "path,label,split,source,basic,trained" and len(rows) == 13
    assert "not proof" in (out / "summary.md").read_text()


def test_cli_eval_with_saved_model_and_min_auc(tmp_path):
    out = tmp_path / "eval"
    r = CliRunner().invoke(aicd.main, ["eval", "-m", str(MANIFEST), "-o", str(out), "--train", "-q"])
    assert r.exit_code == 0, r.output
    r2 = CliRunner().invoke(aicd.main, ["eval", "-m", str(MANIFEST), "-o", str(tmp_path / "e2"), "-q",
                                        "--model", str(out / "model.json"), "--min-auc", "0"])
    assert r2.exit_code == 0, r2.output
    assert set(json.loads((tmp_path / "e2" / "metrics.json").read_text())["scorers"]) == {"basic", "model"}

    # Same files, labels swapped: AUC collapses and --min-auc fails the run.
    flipped = MANIFEST.read_text().replace("label: human", "label: TMP").replace("label: ai", "label: human") \
        .replace("label: TMP", "label: ai")
    fm = MANIFEST.parent / "_flipped_test_manifest.yaml"
    try:
        fm.write_text(flipped)
        r3 = CliRunner().invoke(aicd.main, ["eval", "-m", str(fm), "-o", str(tmp_path / "e3"), "-q",
                                            "--min-auc", "0.5"])
    finally:
        fm.unlink()
    assert r3.exit_code == 1
    bad = CliRunner().invoke(aicd.main, ["eval", "-m", str(_manifest(tmp_path, "entries: []\n")), "-o",
                                         str(tmp_path / "e4"), "-q"])
    assert bad.exit_code == 1 and "no files" in bad.output
