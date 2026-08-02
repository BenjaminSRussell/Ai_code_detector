"""Tests for performance hotspot analyzer (static pass)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from analysis.ast_parser import PythonASTParser
from analysis.metrics_performance import PerformanceAnalyzer


def _parse(code):
    parser = PythonASTParser()
    return parser.parse_file(Path("test.py"), code)


def test_simple_function_has_no_hotspots():
    code = "def add(a, b):\n    return a + b\n"
    file_ast = _parse(code)

    analyzer = PerformanceAnalyzer()
    hotspots = analyzer.analyze_file(file_ast)

    assert hotspots == []


def test_flags_nested_loops():
    code = (
        "def matrix_multiply(a, b):\n"
        "    result = []\n"
        "    for i in a:\n"
        "        for j in b:\n"
        "            result.append(i * j)\n"
        "    return result\n"
    )
    file_ast = _parse(code)

    analyzer = PerformanceAnalyzer()
    hotspots = analyzer.analyze_file(file_ast)

    assert len(hotspots) == 1
    assert hotspots[0].function_name == "matrix_multiply"
    assert hotspots[0].risk_score > 0
    assert any("nested loops" in reason for reason in hotspots[0].reasons)


def test_flags_nested_loops_inside_a_class_method():
    code = (
        "class Matrix:\n"
        "    def multiply(self, a, b):\n"
        "        result = []\n"
        "        for i in a:\n"
        "            for j in b:\n"
        "                result.append(i * j)\n"
        "        return result\n"
    )
    file_ast = _parse(code)

    analyzer = PerformanceAnalyzer()
    hotspots = analyzer.analyze_file(file_ast)

    assert len(hotspots) == 1
    assert hotspots[0].function_name == "multiply"


def test_flags_unmemoized_recursion():
    code = (
        "def fib(n):\n"
        "    if n <= 1:\n"
        "        return n\n"
        "    return fib(n - 1) + fib(n - 2)\n"
    )
    file_ast = _parse(code)

    analyzer = PerformanceAnalyzer()
    hotspots = analyzer.analyze_file(file_ast)

    assert len(hotspots) == 1
    assert any("recursive" in reason for reason in hotspots[0].reasons)


def test_memoized_recursion_is_not_flagged():
    code = (
        "from functools import lru_cache\n\n"
        "@lru_cache\n"
        "def fib(n):\n"
        "    if n <= 1:\n"
        "        return n\n"
        "    return fib(n - 1) + fib(n - 2)\n"
    )
    file_ast = _parse(code)

    analyzer = PerformanceAnalyzer()
    hotspots = analyzer.analyze_file(file_ast)

    assert hotspots == []


def test_flags_self_recursion_in_class_method():
    """Regression test for Finding 1: self.method() recursion detection."""
    code = (
        "class Fibonacci:\n"
        "    def fib(self, n):\n"
        "        if n <= 1:\n"
        "            return n\n"
        "        return self.fib(n - 1) + self.fib(n - 2)\n"
    )
    file_ast = _parse(code)

    analyzer = PerformanceAnalyzer()
    hotspots = analyzer.analyze_file(file_ast)

    assert len(hotspots) == 1
    assert hotspots[0].function_name == "fib"
    assert any("recursive" in reason for reason in hotspots[0].reasons)


def test_qualified_memoization_decorator_recognized():
    """Regression test for Finding 2: @functools.lru_cache decorator handling."""
    code = (
        "import functools\n\n"
        "@functools.lru_cache\n"
        "def fib(n):\n"
        "    if n <= 1:\n"
        "        return n\n"
        "    return fib(n - 1) + fib(n - 2)\n"
    )
    file_ast = _parse(code)

    analyzer = PerformanceAnalyzer()
    hotspots = analyzer.analyze_file(file_ast)

    assert hotspots == []


def test_nested_function_deduplication():
    """Regression test for Finding 3: nested function duplicate hotspot exclusion."""
    code = (
        "def outer(a, b):\n"
        "    def inner():\n"
        "        result = []\n"
        "        for i in a:\n"
        "            for j in b:\n"
        "                result.append(i * j)\n"
        "        return result\n"
        "    return inner()\n"
    )
    file_ast = _parse(code)

    analyzer = PerformanceAnalyzer()
    hotspots = analyzer.analyze_file(file_ast)

    # Should return exactly ONE hotspot (not two for outer+inner)
    assert len(hotspots) == 1
    # The hotspot should be for 'outer' (the outermost function)
    assert hotspots[0].function_name == "outer"
