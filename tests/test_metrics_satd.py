"""Tests for SATD (self-admitted technical debt) marker analyzer."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from analysis.metrics_satd import SATDAnalyzer


def test_no_markers_in_clean_code():
    analyzer = SATDAnalyzer()
    code = "def add(a, b):\n    return a + b\n"

    features = analyzer.analyze_file(code, "clean.py")

    assert features.markers == []
    assert features.density == 0.0


def test_detects_todo_and_fixme_markers():
    analyzer = SATDAnalyzer()
    code = (
        "def process(data):\n"
        "    # TODO: handle empty input\n"
        "    result = data\n"
        "    # FIXME this is broken for negative numbers\n"
        "    return result\n"
    )

    features = analyzer.analyze_file(code, "messy.py")

    assert len(features.markers) == 2
    assert features.markers[0].marker == "TODO"
    assert features.markers[0].line == 2
    assert features.markers[1].marker == "FIXME"
    assert features.density > 0.0


def test_marker_matching_is_word_bounded():
    analyzer = SATDAnalyzer()
    code = "TODOLIST = []\n"

    features = analyzer.analyze_file(code, "vars.py")

    assert features.markers == []
