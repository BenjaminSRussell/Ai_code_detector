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


def test_multiple_markers_on_single_line():
    """Regression test: a line with multiple markers should detect all of them."""
    analyzer = SATDAnalyzer()
    code = "    # TODO: fix this HACK\n"

    features = analyzer.analyze_file(code, "multi.py")

    assert len(features.markers) == 2
    markers_found = {m.marker for m in features.markers}
    assert "TODO" in markers_found
    assert "HACK" in markers_found


def test_satd_marker_fields():
    """Verify SATDMarker fields (file_path and text) are correctly populated."""
    analyzer = SATDAnalyzer()
    code = "def process():\n    # TODO: handle edge case\n    pass\n"

    features = analyzer.analyze_file(code, "test_file.py")

    assert len(features.markers) == 1
    marker = features.markers[0]
    assert marker.file_path == "test_file.py"
    assert "TODO" in marker.text
    assert "handle edge case" in marker.text


def test_hack_and_xxx_markers_and_case_insensitive():
    """Verify HACK, XXX markers and case-insensitive matching (e.g., lowercase 'todo')."""
    analyzer = SATDAnalyzer()
    code = (
        "# HACK: temporary workaround\n"
        "# XXX: refactor this\n"
        "# todo: case insensitive\n"
        "# fixme: also lowercase\n"
    )

    features = analyzer.analyze_file(code, "all_markers.py")

    assert len(features.markers) == 4
    markers_found = [m.marker for m in features.markers]
    assert "HACK" in markers_found
    assert "XXX" in markers_found
    assert "TODO" in markers_found
    assert "FIXME" in markers_found
