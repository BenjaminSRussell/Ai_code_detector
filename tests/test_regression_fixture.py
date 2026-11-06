"""Regression fixture for the v1 goal (issue #31).

The eval harness already runs in CI, but its numbers only live in the
gitignored ``reports/eval/`` output — nothing in the repo fails if the
detector silently starts scoring the known AI samples below the known
human ones. This test pins the ``basic`` score of every file in
``datasets/manifest.yaml`` to the values recorded in
``datasets/expected_scores.json`` (captured from a real harness run) and
asserts the ordering the goal depends on: every AI file above every
human file (ROC AUC 1.0 on the fixture).

To refresh after an intentional scoring change, re-run
``python scripts/eval_harness.py --manifest datasets/manifest.yaml --train``
and copy the ``basic`` column of ``predictions.csv`` into
``datasets/expected_scores.json``. Never edit the expected values by hand
to make a failing test pass.
"""
import json
from pathlib import Path

import pytest

from ai_code_detector.evaluation import load_manifest, roc_auc, score_files

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "datasets" / "manifest.yaml"
EXPECTED = REPO / "datasets" / "expected_scores.json"

LABELS = {0: "human", 1: "ai"}


def _load_expected():
    data = json.loads(EXPECTED.read_text(encoding="utf-8"))
    assert data["manifest"] == "datasets/manifest.yaml"
    assert data["manifest_version"] == 1 and data["scorer"] == "basic"
    return data


def test_fixture_scores_match_recorded_baseline():
    data = _load_expected()
    entries = load_manifest(MANIFEST)
    expected = {f["path"]: f for f in data["files"]}
    # The fixture set itself must not drift: every manifest entry recorded,
    # no stale entries left behind.
    assert set(expected) == {e.rel for e in entries}, (
        "datasets/expected_scores.json is out of sync with datasets/manifest.yaml; "
        "re-run the eval harness and refresh it")
    for e in entries:
        assert expected[e.rel]["label"] == LABELS[e.label]

    scores = score_files([e.path for e in entries], use_ml=False)
    tolerance = float(data["tolerance"])
    for e, actual in zip(entries, scores):
        want = expected[e.rel]["expected"]
        assert actual == pytest.approx(want, abs=tolerance), (
            f"{e.rel}: basic score {actual:.4f} drifted from recorded {want:.4f} "
            f"(tolerance {tolerance}); if the scoring change was intentional, "
            "re-run the eval harness and refresh datasets/expected_scores.json")

    ai = [s for e, s in zip(entries, scores) if e.label == 1]
    human = [s for e, s in zip(entries, scores) if e.label == 0]
    assert min(ai) > max(human), (
        f"AI/human ordering broke: lowest AI score {min(ai):.4f} is not above "
        f"highest human score {max(human):.4f}")
    assert roc_auc([e.label for e in entries], scores) == 1.0
