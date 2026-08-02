"""Performance hotspot detection: static heuristics and opt-in dynamic profiling."""

import ast
import textwrap
import subprocess
import sys
import tempfile
import pstats
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from analysis.ast_parser import FileAST, FunctionInfo


@dataclass
class HotspotFunction:
    """A function flagged as a potential performance hotspot."""
    file_path: str
    function_name: str
    start_line: int
    risk_score: float
    reasons: List[str] = field(default_factory=list)
    measured_time_seconds: Optional[float] = None


class PerformanceAnalyzer:
    """Static AST-based heuristics for likely-slow functions."""

    NESTED_LOOP_THRESHOLD = 2
    LOOP_CALL_THRESHOLD = 5
    MEMO_DECORATORS = {'lru_cache', 'cache', 'cached_property'}

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.top_n = self.config.get('top_n_hotspots', 20)

    def analyze_file(self, file_ast: FileAST) -> List[HotspotFunction]:
        """Flag functions in a file that show risky performance patterns.

        Args:
            file_ast: Parsed AST of the file

        Returns:
            List of HotspotFunction, one per flagged function
        """
        functions = list(file_ast.functions)
        for cls in file_ast.classes:
            functions.extend(cls.methods)

        functions = self._exclude_nested_functions(functions)

        hotspots = []
        for func in functions:
            result = self._analyze_function(func, str(file_ast.file_path))
            if result is not None:
                hotspots.append(result)

        hotspots.sort(key=lambda h: h.risk_score, reverse=True)
        return hotspots[:self.top_n]

    def _analyze_function(self, func: FunctionInfo, file_path: str) -> Optional[HotspotFunction]:
        try:
            tree = ast.parse(textwrap.dedent(func.code))
        except (SyntaxError, IndentationError):
            return None

        func_node = self._find_function_node(tree)
        if func_node is None:
            return None

        nesting = self._max_loop_nesting_depth(func_node)
        recursive = self._is_self_recursive(func_node, func.name) and not self._is_memoized(func)
        loop_calls = self._count_calls_in_loops(func_node)

        reasons = []
        risk_score = 0.0

        if nesting >= self.NESTED_LOOP_THRESHOLD:
            risk_score += 0.5
            reasons.append(f"nested loops {nesting} levels deep (possible O(n^{nesting}) behavior)")

        if recursive:
            risk_score += 0.3
            reasons.append("self-recursive without caching/memoization")

        if loop_calls > self.LOOP_CALL_THRESHOLD:
            risk_score += 0.2
            reasons.append(f"{loop_calls} function calls inside loop bodies")

        if risk_score == 0.0:
            return None

        return HotspotFunction(
            file_path=file_path,
            function_name=func.name,
            start_line=func.start_line,
            risk_score=min(1.0, risk_score),
            reasons=reasons,
        )

    def _find_function_node(self, tree: ast.AST):
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return node
        return None

    def _max_loop_nesting_depth(self, node: ast.AST) -> int:
        max_depth = 0

        def walk(n, depth):
            nonlocal max_depth
            for child in ast.iter_child_nodes(n):
                if isinstance(child, (ast.For, ast.While)):
                    new_depth = depth + 1
                    max_depth = max(max_depth, new_depth)
                    walk(child, new_depth)
                else:
                    walk(child, depth)

        walk(node, 0)
        return max_depth

    def _is_self_recursive(self, node: ast.AST, func_name: str) -> bool:
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name) and child.func.id == func_name:
                    return True
                if isinstance(child.func, ast.Attribute) and child.func.attr == func_name:
                    if isinstance(child.func.value, ast.Name) and child.func.value.id == 'self':
                        return True
        return False

    def _is_memoized(self, func: FunctionInfo) -> bool:
        for d in func.decorators:
            name = d.split('(')[0].split('.')[-1]
            if name in self.MEMO_DECORATORS:
                return True
        return False

    def _count_calls_in_loops(self, node: ast.AST) -> int:
        count = 0

        def walk(n, inside_loop):
            nonlocal count
            for child in ast.iter_child_nodes(n):
                is_loop = isinstance(child, (ast.For, ast.While))
                if isinstance(child, ast.Call) and inside_loop:
                    count += 1
                walk(child, inside_loop or is_loop)

        walk(node, False)
        return count

    def _exclude_nested_functions(self, functions: List[FunctionInfo]) -> List[FunctionInfo]:
        """Drop functions whose line range is fully contained within another
        function's range (nested/closure defs) so a hotspot isn't double-reported
        under both the outer and inner function names."""
        result = []
        for func in functions:
            is_nested = any(
                other is not func
                and other.start_line <= func.start_line
                and func.end_line <= other.end_line
                for other in functions
            )
            if not is_nested:
                result.append(func)
        return result


class PerformanceProfiler:
    """Runs a repo's test suite under cProfile to get real per-function timings.

    This executes code from the scanned repository as a subprocess. Callers
    MUST treat this as an explicit opt-in action (e.g. a CLI flag), never a
    default/automatic step, since the scanned repo's code is untrusted.
    """

    def __init__(self, timeout_seconds: int = 30):
        self.timeout_seconds = timeout_seconds

    def detect_entry_point(self, repo_path: Path) -> Optional[List[str]]:
        """Find a safe, discoverable way to exercise the repo's code.

        Returns a pytest invocation (as argv, minus the python executable)
        if a tests directory is found, otherwise None.
        """
        tests_dir = repo_path / "tests"
        if tests_dir.is_dir() and any(tests_dir.glob("test_*.py")):
            return ["-m", "pytest", str(tests_dir), "-q"]
        return None

    def profile(self, repo_path: Path, entry_point: List[str]) -> Dict[str, float]:
        """Run entry_point under cProfile and return per-function cumulative time.

        Args:
            repo_path: Directory to run the command in
            entry_point: Argv to pass to the interpreter (from detect_entry_point)

        Returns:
            Mapping of "file:function" to cumulative seconds. Empty dict if
            profiling failed or timed out.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            stats_path = Path(tmpdir) / "profile.stats"
            command = [sys.executable, "-m", "cProfile", "-o", str(stats_path)] + entry_point

            try:
                subprocess.run(
                    command,
                    cwd=repo_path,
                    timeout=self.timeout_seconds,
                    capture_output=True,
                )
            except (subprocess.TimeoutExpired, OSError):
                return {}

            if not stats_path.exists():
                return {}

            try:
                return self._load_stats(stats_path)
            except Exception:
                return {}

    def _load_stats(self, stats_path: Path) -> Dict[str, float]:
        stats = pstats.Stats(str(stats_path))

        timings = {}
        for func_key, raw_stats in stats.stats.items():
            file_path, _line_number, func_name = func_key
            cumulative_time = raw_stats[3]  # (cc, nc, tt, ct)
            timings[f"{file_path}:{func_name}"] = cumulative_time

        return timings


def merge_profiling_results(
    hotspots: List[HotspotFunction], timings: Dict[str, float]
) -> List[HotspotFunction]:
    """Attach real profiled timings to statically-detected hotspots by function name."""
    for hotspot in hotspots:
        matches = [
            seconds for key, seconds in timings.items()
            if key.endswith(f":{hotspot.function_name}")
        ]
        if matches:
            hotspot.measured_time_seconds = max(matches)

    return hotspots
