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
