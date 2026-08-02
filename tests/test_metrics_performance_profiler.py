"""Tests for the opt-in dynamic profiling pass."""

import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from analysis.metrics_performance import HotspotFunction, PerformanceProfiler, merge_profiling_results


def test_detect_entry_point_finds_pytest_tests_dir(tmp_path):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_example.py").write_text("def test_ok():\n    assert True\n")

    profiler = PerformanceProfiler()
    entry_point = profiler.detect_entry_point(tmp_path)

    assert entry_point is not None
    assert "pytest" in entry_point


def test_detect_entry_point_returns_none_without_tests():
    profiler = PerformanceProfiler()

    with tempfile.TemporaryDirectory() as tmpdir:
        entry_point = profiler.detect_entry_point(Path(tmpdir))

    assert entry_point is None


def test_profile_returns_timings_for_executed_functions(tmp_path):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()

    (tmp_path / "slow_module.py").write_text(
        "def slow_function():\n"
        "    total = 0\n"
        "    for i in range(200):\n"
        "        for j in range(200):\n"
        "            total += i * j\n"
        "    return total\n"
    )

    (tests_dir / "test_slow.py").write_text(
        "import sys\n"
        "sys.path.insert(0, '.')\n"
        "from slow_module import slow_function\n\n"
        "def test_runs_slow_function():\n"
        "    assert slow_function() >= 0\n"
    )

    profiler = PerformanceProfiler(timeout_seconds=30)
    entry_point = profiler.detect_entry_point(tmp_path)
    assert entry_point is not None

    timings = profiler.profile(tmp_path, entry_point)

    assert any(key.endswith(":slow_function") for key in timings)


def test_profile_returns_empty_dict_on_timeout(tmp_path):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_hang.py").write_text(
        "import time\n\n"
        "def test_hangs():\n"
        "    time.sleep(5)\n"
    )

    profiler = PerformanceProfiler(timeout_seconds=1)
    entry_point = profiler.detect_entry_point(tmp_path)

    timings = profiler.profile(tmp_path, entry_point)

    assert timings == {}


def test_merge_profiling_results_attaches_measured_time():
    hotspot = HotspotFunction(
        file_path="slow_module.py",
        function_name="slow_function",
        start_line=1,
        risk_score=0.5,
        reasons=["nested loops 2 levels deep"],
    )

    timings = {"/abs/path/slow_module.py:slow_function": 0.042}

    result = merge_profiling_results([hotspot], timings)

    assert result[0].measured_time_seconds == 0.042


def test_merge_profiling_results_leaves_unmatched_hotspots_untouched():
    hotspot = HotspotFunction(
        file_path="a.py",
        function_name="untouched",
        start_line=1,
        risk_score=0.5,
        reasons=["nested loops"],
    )

    result = merge_profiling_results([hotspot], {})

    assert result[0].measured_time_seconds is None
