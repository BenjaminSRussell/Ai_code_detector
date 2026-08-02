"""Tests for repository-wide duplication analyzer."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from analysis.metrics_duplication import RepoDuplicationAnalyzer


def test_no_duplication_across_distinct_files():
    analyzer = RepoDuplicationAnalyzer()

    file_contents = {
        "a.py": "def add(a, b):\n    return a + b\n",
        "b.py": "def subtract(a, b):\n    return a - b\n",
    }

    features = analyzer.analyze_repo(file_contents)

    assert features.duplication_ratio == 0.0
    assert features.duplicate_blocks == []


def test_detects_block_duplicated_across_files():
    analyzer = RepoDuplicationAnalyzer()

    shared_block = (
        "def process(data):\n"
        "    result = []\n"
        "    for item in data:\n"
        "        result.append(item)\n"
        "    return result\n"
    )

    file_contents = {
        "a.py": shared_block,
        "b.py": shared_block,
    }

    features = analyzer.analyze_repo(file_contents)

    assert features.duplication_ratio > 0.0
    assert len(features.duplicate_blocks) > 0

    block = features.duplicate_blocks[0]
    files_hit = {loc[0] for loc in block.locations}
    assert files_hit == {"a.py", "b.py"}


def test_intra_file_repetition_is_not_counted_as_cross_file_duplication():
    analyzer = RepoDuplicationAnalyzer()

    repeated = "x = 1\ny = 2\nz = 3\n"
    file_contents = {
        "a.py": repeated + repeated,
    }

    features = analyzer.analyze_repo(file_contents)

    assert features.duplication_ratio == 0.0
    assert features.duplicate_blocks == []


def test_line_numbers_account_for_blank_lines():
    """Regression test: ensure reported line numbers are actual source line numbers, not filtered indices."""
    analyzer = RepoDuplicationAnalyzer()

    # File with leading blank/import lines before the duplicated block
    # The duplicated block starts at line 4, not line 1
    file_a = (
        "\n"  # line 1: blank
        "import sys\n"  # line 2
        "\n"  # line 3: blank
        "def process(data):\n"  # line 4: start of duplicated block
        "    result = []\n"  # line 5
        "    for item in data:\n"  # line 6
        "        result.append(item)\n"  # line 7
        "    return result\n"  # line 8
    )

    file_b = (
        "def process(data):\n"  # line 1: start of duplicated block
        "    result = []\n"  # line 2
        "    for item in data:\n"  # line 3
        "        result.append(item)\n"  # line 4
        "    return result\n"  # line 5
    )

    file_contents = {
        "a.py": file_a,
        "b.py": file_b,
    }

    features = analyzer.analyze_repo(file_contents)

    assert len(features.duplicate_blocks) > 0

    # Find the duplicate block
    block = features.duplicate_blocks[0]

    # Extract locations by file
    locations_by_file = {loc[0]: loc[1] for loc in block.locations}

    # For file_a, the reported line should be 4 (where "def process" actually is)
    # not 1 (what the position would be after filtering blank lines)
    assert locations_by_file["a.py"] == 4, f"Expected line 4 for a.py, got {locations_by_file['a.py']}"

    # For file_b, it should be 1
    assert locations_by_file["b.py"] == 1, f"Expected line 1 for b.py, got {locations_by_file['b.py']}"
