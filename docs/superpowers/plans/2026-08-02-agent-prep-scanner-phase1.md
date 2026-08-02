# Agent-Prep Scanner Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `Ai_code_detector` with four new detection signals (repo-wide duplication, AI-attribution commit trailers, SATD markers, performance hotspots) and a new agent-readable findings report (Markdown + JSON), all driven by a new CLI, so the tool can be run as a pre-agent scan before handing a codebase to a coding agent.

**Architecture:** New analyzer modules under `src/analysis/` follow the existing `Analyzer` class + dataclass pattern (see `metrics_stylometry.py`, `metrics_structural.py`, `metrics_history.py`). They are NOT wired into the existing weighted `ai_probability` scoring in `model/aggregator.py` — the spec is explicit that these are findings, not probabilistic score inputs. A new orchestrator (`src/agent_scan.py`) runs the existing `AICodeDetector` for the AI-probability verdict, then separately runs the four new analyzers and merges everything into a `ScanFindings` object, which two new report writers (`src/report/reporter_findings.py`) render as `AI_SCAN_FINDINGS.md` and `ai_scan_findings.json`. A new CLI (`src/cli_agent_scan.py`) exposes it, mirroring the existing `cli.py`/`cli_enhanced.py` pattern.

**Tech Stack:** Python 3.9+, stdlib only for new code (`ast`, `re`, `subprocess`, `pstats`, `tempfile`, `textwrap`, `json`, `collections`) plus the project's existing dependencies (`click`, `GitPython` via `ingest.git_loader`, `pytest` for tests). No new third-party dependencies.

## Global Constraints

- Local codebases must work exactly like remote GitHub URLs (already true via `GitLoader._is_url`) — no new work needed there, but every new analyzer must be exercised against local-path scans in its tests.
- The dynamic profiling pass executes code from the scanned repository. It must NEVER run automatically — it is only triggered via an explicit `--profile` CLI flag (`enable_profiling=True` passed explicitly), never a default.
- Findings files (`AI_SCAN_FINDINGS.md`, `ai_scan_findings.json`) are written to the scanned repo's root by default, with an `--output` override.
- New signals (duplication, attribution, SATD, performance) are surfaced as `Finding` entries in the new findings report — they do NOT feed into `model/aggregator.py`'s weighted `ai_probability` score. Do not modify `aggregator.py`, `classifier.py`, or the existing `FileScore`/`RepoScore` dataclasses.
- Follow existing code style exactly: `dataclass` result types, an `Analyzer` class taking `config: Dict = None` in `__init__`, `analyze_file`/`analyze_repo` method names, flat (non-relative) imports like `from analysis.tokenizer import CodeTokenizer` (this works because `setup.py` uses `package_dir={"": "src"}` and `pytest.ini` sets `pythonpath = . src`).
- Tests go in `tests/`, follow the existing pattern of `sys.path.insert(0, str(Path(__file__).parent.parent / "src"))` at the top followed by flat imports, and use plain `pytest` functions (no test classes), matching `tests/test_basic.py` and `tests/test_bug.py`.
- No new pip dependencies. `pyproject`/`requirements.txt`/`setup.py` are not touched by this plan.

---

### Task 1: Repo-wide (cross-file) duplication analyzer

**Files:**
- Create: `src/analysis/metrics_duplication.py`
- Test: `tests/test_metrics_duplication.py`

**Interfaces:**
- Consumes: nothing new (plain `Dict[str, str]` of file contents, caller-supplied).
- Produces:
  - `RepoDuplicationAnalyzer(config: Dict = None)` with `.analyze_repo(file_contents: Dict[str, str]) -> RepoDuplicationFeatures`
  - `RepoDuplicationFeatures(duplication_ratio: float, duplicate_blocks: List[DuplicateBlock])`
  - `DuplicateBlock(lines: Tuple[str, ...], locations: List[Tuple[str, int]])` — `locations` entries are `(file_path, start_line)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_metrics_duplication.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_metrics_duplication.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'analysis.metrics_duplication'`

- [ ] **Step 3: Write the implementation**

Create `src/analysis/metrics_duplication.py`:

```python
"""Repository-wide (cross-file) code duplication detection."""

from typing import Dict, List, Tuple
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class DuplicateBlock:
    """A block of code duplicated across two or more files."""
    lines: Tuple[str, ...]
    locations: List[Tuple[str, int]]


@dataclass
class RepoDuplicationFeatures:
    """Cross-file duplication features for a repository."""
    duplication_ratio: float
    duplicate_blocks: List[DuplicateBlock]


class RepoDuplicationAnalyzer:
    """Detects code blocks duplicated across multiple files in a repository."""

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.ngram_size = self.config.get('ngram_size', 3)
        self.top_n_blocks = self.config.get('top_n_blocks', 20)

    def analyze_repo(self, file_contents: Dict[str, str]) -> RepoDuplicationFeatures:
        """Find duplicate code blocks across files.

        Args:
            file_contents: Mapping of relative file path to source code

        Returns:
            RepoDuplicationFeatures
        """
        ngram_locations: Dict[Tuple[str, ...], List[Tuple[str, int]]] = defaultdict(list)
        total_ngrams = 0
        n = self.ngram_size

        for file_path, code in file_contents.items():
            lines = [line.strip() for line in code.split('\n') if line.strip()]

            for i in range(len(lines) - n + 1):
                ngram = tuple(lines[i:i + n])
                ngram_locations[ngram].append((file_path, i + 1))
                total_ngrams += 1

        duplicate_blocks = []
        cross_file_duplicate_count = 0

        for ngram, locations in ngram_locations.items():
            distinct_files = {loc[0] for loc in locations}
            if len(distinct_files) >= 2:
                cross_file_duplicate_count += len(locations)
                duplicate_blocks.append(DuplicateBlock(lines=ngram, locations=locations))

        duplicate_blocks.sort(key=lambda b: len(b.locations), reverse=True)

        duplication_ratio = (
            min(1.0, cross_file_duplicate_count / total_ngrams) if total_ngrams else 0.0
        )

        return RepoDuplicationFeatures(
            duplication_ratio=duplication_ratio,
            duplicate_blocks=duplicate_blocks[:self.top_n_blocks],
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_metrics_duplication.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/analysis/metrics_duplication.py tests/test_metrics_duplication.py
git commit -m "feat: add repo-wide cross-file duplication analyzer"
```

---

### Task 2: AI-attribution commit-trailer analyzer

**Files:**
- Create: `src/analysis/metrics_attribution.py`
- Test: `tests/test_metrics_attribution.py`

**Interfaces:**
- Consumes: `RepoInfo`, `CommitInfo` from `src/ingest/git_loader.py` (existing — `RepoInfo.commits: List[CommitInfo]`, `CommitInfo.sha: str`, `CommitInfo.message: str`).
- Produces:
  - `AttributionAnalyzer(config: Dict = None)` with `.analyze_repo(repo_info: RepoInfo) -> AttributionFeatures`
  - `AttributionFeatures(has_ai_attribution: bool, match_ratio: float, matches: List[AttributionMatch])`
  - `AttributionMatch(commit_sha: str, pattern_matched: str, message_snippet: str)`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_metrics_attribution.py`:

```python
"""Tests for AI-attribution commit-trailer analyzer."""

import sys
from pathlib import Path
from datetime import datetime
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from analysis.metrics_attribution import AttributionAnalyzer
from ingest.git_loader import RepoInfo, CommitInfo


def _make_repo_info(messages):
    commits = [
        CommitInfo(
            sha=f"sha{i}",
            author="Test Author",
            email="test@example.com",
            timestamp=datetime(2026, 1, 1),
            message=message,
            files_changed=[],
            lines_added=0,
            lines_deleted=0,
        )
        for i, message in enumerate(messages)
    ]
    return RepoInfo(
        path=Path("."),
        is_git=True,
        remote_url=None,
        commits=commits,
        authors=["Test Author"],
        total_commits=len(commits),
        first_commit_date=None,
        last_commit_date=None,
    )


def test_no_attribution_found():
    analyzer = AttributionAnalyzer()
    repo_info = _make_repo_info(["fix bug in parser", "add tests"])

    features = analyzer.analyze_repo(repo_info)

    assert features.has_ai_attribution is False
    assert features.match_ratio == 0.0
    assert features.matches == []


def test_detects_claude_code_co_author_trailer():
    analyzer = AttributionAnalyzer()
    repo_info = _make_repo_info([
        "Add login form\n\nCo-Authored-By: Claude <noreply@anthropic.com>",
        "Fix typo",
    ])

    features = analyzer.analyze_repo(repo_info)

    assert features.has_ai_attribution is True
    assert features.match_ratio == 0.5
    assert len(features.matches) == 1
    assert features.matches[0].commit_sha == "sha0"


def test_custom_pattern_from_config():
    analyzer = AttributionAnalyzer(config={'ai_attribution_patterns': [r'written by robots']})
    repo_info = _make_repo_info(["this file was written by robots"])

    features = analyzer.analyze_repo(repo_info)

    assert features.has_ai_attribution is True


def test_empty_commit_history_returns_no_attribution():
    analyzer = AttributionAnalyzer()
    repo_info = _make_repo_info([])

    features = analyzer.analyze_repo(repo_info)

    assert features.has_ai_attribution is False
    assert features.match_ratio == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_metrics_attribution.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'analysis.metrics_attribution'`

- [ ] **Step 3: Write the implementation**

Create `src/analysis/metrics_attribution.py`:

```python
"""Detection of explicit AI-tool attribution markers in commit messages."""

import re
from typing import Dict, List
from dataclasses import dataclass

from ingest.git_loader import RepoInfo


@dataclass
class AttributionMatch:
    """A commit whose message matched a known AI-attribution pattern."""
    commit_sha: str
    pattern_matched: str
    message_snippet: str


@dataclass
class AttributionFeatures:
    """AI attribution signal extracted from commit history."""
    has_ai_attribution: bool
    match_ratio: float
    matches: List[AttributionMatch]


class AttributionAnalyzer:
    """Scans commit messages for known AI coding-tool attribution markers."""

    DEFAULT_PATTERNS = [
        r"Co-Authored-By:\s*Claude",
        r"Co-Authored-By:\s*.*[Cc]opilot",
        r"Co-Authored-By:\s*.*[Cc]ursor",
        r"Co-Authored-By:\s*.*[Cc]odex",
        r"Generated with Claude Code",
        r"Generated by GitHub Copilot",
        r"🤖 Generated with",
    ]

    def __init__(self, config: Dict = None):
        self.config = config or {}
        custom_patterns = self.config.get('ai_attribution_patterns', [])
        self.patterns = [
            re.compile(p, re.IGNORECASE) for p in self.DEFAULT_PATTERNS + custom_patterns
        ]

    def analyze_repo(self, repo_info: RepoInfo) -> AttributionFeatures:
        """Scan repository commit history for AI-attribution markers.

        Args:
            repo_info: Repository information with commit history

        Returns:
            AttributionFeatures
        """
        if not repo_info.commits:
            return AttributionFeatures(has_ai_attribution=False, match_ratio=0.0, matches=[])

        matches = []

        for commit in repo_info.commits:
            for pattern in self.patterns:
                if pattern.search(commit.message):
                    matches.append(AttributionMatch(
                        commit_sha=commit.sha,
                        pattern_matched=pattern.pattern,
                        message_snippet=commit.message.strip()[:200],
                    ))
                    break

        match_ratio = len(matches) / len(repo_info.commits)

        return AttributionFeatures(
            has_ai_attribution=len(matches) > 0,
            match_ratio=match_ratio,
            matches=matches,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_metrics_attribution.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/analysis/metrics_attribution.py tests/test_metrics_attribution.py
git commit -m "feat: add AI-attribution commit-trailer analyzer"
```

---

### Task 3: SATD marker analyzer

**Files:**
- Create: `src/analysis/metrics_satd.py`
- Test: `tests/test_metrics_satd.py`

**Interfaces:**
- Consumes: nothing new (plain source-code string).
- Produces:
  - `SATDAnalyzer(config: Dict = None)` with `.analyze_file(code: str, file_path: str) -> SATDFeatures`
  - `SATDFeatures(markers: List[SATDMarker], density: float)`
  - `SATDMarker(file_path: str, line: int, marker: str, text: str)`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_metrics_satd.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_metrics_satd.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'analysis.metrics_satd'`

- [ ] **Step 3: Write the implementation**

Create `src/analysis/metrics_satd.py`:

```python
"""Self-admitted technical debt (SATD) marker detection."""

import re
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class SATDMarker:
    """A single self-admitted technical debt marker found in a file."""
    file_path: str
    line: int
    marker: str
    text: str


@dataclass
class SATDFeatures:
    """SATD markers found in a single file."""
    markers: List[SATDMarker]
    density: float


class SATDAnalyzer:
    """Scans source code for self-admitted technical debt markers."""

    MARKER_PATTERN = re.compile(r'\b(TODO|FIXME|HACK|XXX)\b', re.IGNORECASE)

    def __init__(self, config: Dict = None):
        self.config = config or {}

    def analyze_file(self, code: str, file_path: str) -> SATDFeatures:
        """Scan a file's source for SATD markers.

        Args:
            code: Source code content
            file_path: Relative path of the file (for reporting)

        Returns:
            SATDFeatures
        """
        lines = code.split('\n')
        markers = []

        for i, line in enumerate(lines, start=1):
            match = self.MARKER_PATTERN.search(line)
            if match:
                markers.append(SATDMarker(
                    file_path=file_path,
                    line=i,
                    marker=match.group(1).upper(),
                    text=line.strip()[:200],
                ))

        non_empty_lines = sum(1 for line in lines if line.strip())
        density = (len(markers) / non_empty_lines * 100) if non_empty_lines else 0.0

        return SATDFeatures(markers=markers, density=density)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_metrics_satd.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/analysis/metrics_satd.py tests/test_metrics_satd.py
git commit -m "feat: add SATD marker analyzer"
```

---

### Task 4: Performance hotspot analyzer — static pass

**Files:**
- Create: `src/analysis/metrics_performance.py`
- Test: `tests/test_metrics_performance.py`

**Interfaces:**
- Consumes: `FileAST`, `FunctionInfo` from `src/analysis/ast_parser.py` (existing — `FileAST.functions: List[FunctionInfo]`, `FileAST.classes: List[ClassInfo]`, `ClassInfo.methods: List[FunctionInfo]`, `FunctionInfo.name/start_line/code/decorators`).
- Produces:
  - `PerformanceAnalyzer(config: Dict = None)` with `.analyze_file(file_ast: FileAST) -> List[HotspotFunction]`
  - `HotspotFunction(file_path: str, function_name: str, start_line: int, risk_score: float, reasons: List[str], measured_time_seconds: Optional[float] = None)`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_metrics_performance.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_metrics_performance.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'analysis.metrics_performance'`

- [ ] **Step 3: Write the implementation**

Create `src/analysis/metrics_performance.py`:

```python
"""Performance hotspot detection: static heuristics and opt-in dynamic profiling."""

import ast
import textwrap
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
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                if child.func.id == func_name:
                    return True
        return False

    def _is_memoized(self, func: FunctionInfo) -> bool:
        return any(d.split('(')[0] in self.MEMO_DECORATORS for d in func.decorators)

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_metrics_performance.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/analysis/metrics_performance.py tests/test_metrics_performance.py
git commit -m "feat: add static performance hotspot analyzer"
```

---

### Task 5: Performance hotspot analyzer — opt-in dynamic profiling pass

**Files:**
- Modify: `src/analysis/metrics_performance.py` (add `PerformanceProfiler` and `merge_profiling_results`)
- Test: `tests/test_metrics_performance_profiler.py`

**Interfaces:**
- Consumes: `HotspotFunction` from Task 4 (same module).
- Produces:
  - `PerformanceProfiler(timeout_seconds: int = 30)` with:
    - `.detect_entry_point(repo_path: Path) -> Optional[List[str]]`
    - `.profile(repo_path: Path, entry_point: List[str]) -> Dict[str, float]`
  - Module-level `merge_profiling_results(hotspots: List[HotspotFunction], timings: Dict[str, float]) -> List[HotspotFunction]`

**Safety requirement (Global Constraints):** `PerformanceProfiler.profile()` runs a subprocess that executes code from the scanned repo. Nothing in this task calls it automatically — that gate lives in Task 7/8, which must only invoke it when a caller explicitly opts in.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_metrics_performance_profiler.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_metrics_performance_profiler.py -v`
Expected: FAIL with `ImportError: cannot import name 'PerformanceProfiler' from 'analysis.metrics_performance'`

- [ ] **Step 3: Add the profiler to the implementation**

Add to the top of `src/analysis/metrics_performance.py` (new imports, alongside the existing `ast`/`textwrap` imports):

```python
import subprocess
import sys
import tempfile
import pstats
from pathlib import Path
```

Append to `src/analysis/metrics_performance.py` (after the `PerformanceAnalyzer` class):

```python
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

            return self._load_stats(stats_path)

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_metrics_performance_profiler.py -v`
Expected: PASS (6 passed). Note: this test run takes a few seconds longer than other tasks' tests because it spawns real subprocesses and includes a 1-second timeout test.

- [ ] **Step 5: Commit**

```bash
git add src/analysis/metrics_performance.py tests/test_metrics_performance_profiler.py
git commit -m "feat: add opt-in dynamic profiling pass for performance hotspots"
```

---

### Task 6: Findings data model + Markdown/JSON report writers

**Files:**
- Create: `src/model/findings.py`
- Create: `src/report/reporter_findings.py`
- Test: `tests/test_reporter_findings.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `Finding(type: str, file: str, severity: str, description: str, line: Optional[int] = None, function: Optional[str] = None, evidence: Dict[str, Any] = {})`
  - `ScanFindings(repo_path: str, ai_probability: float, findings: List[Finding] = [])` with `.by_type(finding_type: str) -> List[Finding]`
  - `FindingsMarkdownWriter().generate(findings: ScanFindings, output_path: Path = None) -> str`
  - `FindingsJSONWriter().generate(findings: ScanFindings, output_path: Path = None) -> str`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_reporter_findings.py`:

```python
"""Tests for the agent-facing findings report writers."""

import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model.findings import Finding, ScanFindings
from report.reporter_findings import FindingsMarkdownWriter, FindingsJSONWriter


def _sample_findings():
    return ScanFindings(
        repo_path="/tmp/example",
        ai_probability=0.72,
        findings=[
            Finding(
                type="ai_attribution",
                file="src/app.py",
                severity="high",
                description="Commit message contains a Claude Code attribution trailer.",
                evidence={"commit_sha": "abc123"},
            ),
            Finding(
                type="performance_hotspot",
                file="src/slow.py",
                line=42,
                function="matrix_multiply",
                severity="warning",
                description="Nested loops 2 levels deep suggest O(n^2) behavior.",
            ),
        ],
    )


def test_markdown_writer_includes_all_findings(tmp_path):
    writer = FindingsMarkdownWriter()
    output_path = tmp_path / "AI_SCAN_FINDINGS.md"

    report = writer.generate(_sample_findings(), output_path)

    assert "ai_attribution" in report
    assert "performance_hotspot" in report
    assert "src/slow.py:42" in report
    assert output_path.exists()


def test_markdown_writer_sorts_high_severity_first():
    writer = FindingsMarkdownWriter()

    report = writer.generate(_sample_findings())

    high_index = report.index("[HIGH]")
    warning_index = report.index("[WARNING]")
    assert high_index < warning_index


def test_markdown_writer_handles_no_findings():
    writer = FindingsMarkdownWriter()
    empty = ScanFindings(repo_path="/tmp/clean", ai_probability=0.1, findings=[])

    report = writer.generate(empty)

    assert "No findings." in report


def test_json_writer_produces_valid_json_with_all_fields(tmp_path):
    writer = FindingsJSONWriter()
    output_path = tmp_path / "ai_scan_findings.json"

    report = writer.generate(_sample_findings(), output_path)
    payload = json.loads(report)

    assert payload["repo_path"] == "/tmp/example"
    assert len(payload["findings"]) == 2
    assert payload["findings"][0]["type"] == "ai_attribution"
    assert payload["findings"][1]["line"] == 42
    assert output_path.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_reporter_findings.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'model.findings'`

- [ ] **Step 3: Write the implementation**

Create `src/model/findings.py`:

```python
"""Data model for agent-facing scan findings."""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class Finding:
    """A single actionable finding surfaced by the agent-prep scanner."""
    type: str
    file: str
    severity: str
    description: str
    line: Optional[int] = None
    function: Optional[str] = None
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScanFindings:
    """All findings produced by a single agent-prep scan run."""
    repo_path: str
    ai_probability: float
    findings: List[Finding] = field(default_factory=list)

    def by_type(self, finding_type: str) -> List[Finding]:
        return [f for f in self.findings if f.type == finding_type]
```

Create `src/report/reporter_findings.py`:

```python
"""Report writers for agent-facing scan findings (Markdown + JSON)."""

import json
from pathlib import Path

from model.findings import ScanFindings

SEVERITY_ORDER = {"high": 0, "warning": 1, "info": 2}


class FindingsMarkdownWriter:
    """Writes ScanFindings as a Markdown file intended for a coding agent to read."""

    def generate(self, findings: ScanFindings, output_path: Path = None) -> str:
        lines = []
        lines.append("# AI Scan Findings")
        lines.append("")
        lines.append(f"**Repository:** `{findings.repo_path}`")
        lines.append(f"**AI Probability:** {findings.ai_probability * 100:.1f}%")
        lines.append("")

        if not findings.findings:
            lines.append("No findings.")
        else:
            sorted_findings = sorted(
                findings.findings,
                key=lambda f: SEVERITY_ORDER.get(f.severity, 99),
            )

            for finding in sorted_findings:
                location = finding.file
                if finding.line is not None:
                    location += f":{finding.line}"

                lines.append(f"## [{finding.severity.upper()}] {finding.type} — `{location}`")
                lines.append("")
                lines.append(finding.description)

                if finding.function:
                    lines.append("")
                    lines.append(f"Function: `{finding.function}`")

                lines.append("")

        report = "\n".join(lines)

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report)

        return report


class FindingsJSONWriter:
    """Writes ScanFindings as a machine-readable JSON task queue."""

    def generate(self, findings: ScanFindings, output_path: Path = None) -> str:
        payload = {
            "repo_path": findings.repo_path,
            "ai_probability": findings.ai_probability,
            "findings": [
                {
                    "type": f.type,
                    "file": f.file,
                    "line": f.line,
                    "function": f.function,
                    "severity": f.severity,
                    "description": f.description,
                    "evidence": f.evidence,
                }
                for f in findings.findings
            ],
        }

        report = json.dumps(payload, indent=2)

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report)

        return report
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_reporter_findings.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/model/findings.py src/report/reporter_findings.py tests/test_reporter_findings.py
git commit -m "feat: add findings data model and Markdown/JSON report writers"
```

---

### Task 7: Orchestrator — `AgentPrepScanner`

**Files:**
- Create: `src/agent_scan.py`
- Test: `tests/test_agent_scan.py`

**Interfaces:**
- Consumes:
  - `AICodeDetector(config_path).analyze_repo(source, verbose) -> RepoScore` (existing, `src/detector.py`)
  - `GitLoader().load(source) -> RepoInfo` (existing, `src/ingest/git_loader.py`)
  - `FileFilter(supported_extensions, excluded_dirs, max_file_size_mb).scan_directory(path) -> List[FileInfo]` (existing, `src/ingest/file_filter.py`)
  - `ASTParserFactory.get_parser(language)` → parser with `.parse_file(path, code) -> FileAST` (existing, `src/analysis/ast_parser.py`)
  - `RepoDuplicationAnalyzer`, `AttributionAnalyzer`, `SATDAnalyzer`, `PerformanceAnalyzer`, `PerformanceProfiler`, `merge_profiling_results` (Tasks 1–5)
  - `Finding`, `ScanFindings` (Task 6)
- Produces:
  - `AgentPrepScanner(config_path: Optional[Path] = None, enable_profiling: bool = False)` with `.scan(source: str, verbose: bool = True) -> ScanFindings`

- [ ] **Step 1: Write the failing test**

Create `tests/test_agent_scan.py`:

```python
"""Integration test for the agent-prep scan orchestrator."""

import sys
import subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent_scan import AgentPrepScanner


def _init_repo(repo_dir: Path):
    subprocess.run(["git", "init", "-q"], cwd=repo_dir, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_dir, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo_dir, check=True)


def _commit_all(repo_dir: Path, message: str):
    subprocess.run(["git", "add", "-A"], cwd=repo_dir, check=True)
    subprocess.run(["git", "commit", "-q", "-m", message], cwd=repo_dir, check=True)


def test_scan_surfaces_all_finding_types(tmp_path):
    repo_dir = tmp_path / "sample_repo"
    repo_dir.mkdir()
    _init_repo(repo_dir)

    slow_code = (
        "def matrix_multiply(a, b):\n"
        "    result = []\n"
        "    for i in a:\n"
        "        for j in b:\n"
        "            result.append(i * j)\n"
        "    return result\n"
        "\n"
        "# TODO: optimize this later\n"
    )

    (repo_dir / "module_a.py").write_text(slow_code)
    (repo_dir / "module_b.py").write_text(slow_code)

    _commit_all(repo_dir, "Add matrix helpers\n\nCo-Authored-By: Claude <noreply@anthropic.com>")

    scanner = AgentPrepScanner(enable_profiling=False)
    findings = scanner.scan(str(repo_dir), verbose=False)

    finding_types = {f.type for f in findings.findings}

    assert "ai_attribution" in finding_types
    assert "duplication" in finding_types
    assert "satd" in finding_types
    assert "performance_hotspot" in finding_types
    assert findings.repo_path == str(repo_dir.resolve())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_agent_scan.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agent_scan'`

- [ ] **Step 3: Write the implementation**

Create `src/agent_scan.py`:

```python
"""Orchestrates the agent-prep scan pipeline.

Runs the existing AI-likelihood detector plus the repo-wide duplication,
AI-attribution, SATD, and performance-hotspot analyzers, and produces a
ScanFindings object for the report writers to render.
"""

from pathlib import Path
from typing import List, Optional

from detector import AICodeDetector
from ingest.git_loader import GitLoader
from ingest.file_filter import FileFilter
from analysis.ast_parser import ASTParserFactory
from analysis.metrics_duplication import RepoDuplicationAnalyzer, RepoDuplicationFeatures
from analysis.metrics_attribution import AttributionAnalyzer, AttributionFeatures
from analysis.metrics_satd import SATDAnalyzer, SATDMarker
from analysis.metrics_performance import (
    PerformanceAnalyzer,
    PerformanceProfiler,
    HotspotFunction,
    merge_profiling_results,
)
from model.findings import Finding, ScanFindings


class AgentPrepScanner:
    """Runs the full agent-prep scan pipeline against a repository."""

    def __init__(self, config_path: Optional[Path] = None, enable_profiling: bool = False):
        self.enable_profiling = enable_profiling

        self.detector = AICodeDetector(config_path=config_path)
        self.git_loader = GitLoader()
        self.file_filter = FileFilter(
            supported_extensions=self.detector.config['ingestion']['supported_extensions'],
            excluded_dirs=self.detector.config['ingestion']['excluded_dirs'],
            max_file_size_mb=self.detector.config['ingestion']['max_file_size_mb'],
        )

        feature_config = self.detector.config.get('features', {})
        self.duplication_analyzer = RepoDuplicationAnalyzer(config=feature_config.get('duplication', {}))
        self.attribution_analyzer = AttributionAnalyzer(config=feature_config.get('attribution', {}))
        self.satd_analyzer = SATDAnalyzer()
        self.performance_analyzer = PerformanceAnalyzer(config=feature_config.get('performance', {}))

    def scan(self, source: str, verbose: bool = True) -> ScanFindings:
        """Run the full agent-prep scan against a repository or local path."""
        repo_score = self.detector.analyze_repo(source, verbose=verbose)

        repo_info = self.git_loader.load(source)
        files = self.file_filter.scan_directory(repo_info.path)

        file_contents = {}
        performance_hotspots: List[HotspotFunction] = []
        satd_markers: List[SATDMarker] = []

        for file_info in files:
            try:
                with open(file_info.path, 'r', encoding='utf-8', errors='ignore') as f:
                    code = f.read()
            except Exception:
                continue

            relative_path = str(file_info.relative_path)
            file_contents[relative_path] = code

            satd_result = self.satd_analyzer.analyze_file(code, relative_path)
            satd_markers.extend(satd_result.markers)

            parser = ASTParserFactory.get_parser(file_info.language)
            if parser:
                try:
                    file_ast = parser.parse_file(Path(relative_path), code)
                    performance_hotspots.extend(self.performance_analyzer.analyze_file(file_ast))
                except Exception:
                    continue

        duplication_features = self.duplication_analyzer.analyze_repo(file_contents)
        attribution_features = self.attribution_analyzer.analyze_repo(repo_info)

        if self.enable_profiling:
            profiler = PerformanceProfiler()
            entry_point = profiler.detect_entry_point(repo_info.path)
            if entry_point:
                timings = profiler.profile(repo_info.path, entry_point)
                performance_hotspots = merge_profiling_results(performance_hotspots, timings)

        findings = []
        findings.extend(self._build_attribution_findings(attribution_features))
        findings.extend(self._build_duplication_findings(duplication_features))
        findings.extend(self._build_satd_findings(satd_markers))
        findings.extend(self._build_performance_findings(performance_hotspots))

        return ScanFindings(
            repo_path=str(repo_info.path),
            ai_probability=repo_score.ai_probability,
            findings=findings,
        )

    def _build_attribution_findings(self, features: AttributionFeatures) -> List[Finding]:
        findings = []
        for match in features.matches:
            findings.append(Finding(
                type="ai_attribution",
                file="(commit history)",
                severity="high",
                description=(
                    f"Commit {match.commit_sha[:8]} matched AI-attribution pattern "
                    f"'{match.pattern_matched}': {match.message_snippet}"
                ),
                evidence={"commit_sha": match.commit_sha, "pattern": match.pattern_matched},
            ))
        return findings

    def _build_duplication_findings(self, features: RepoDuplicationFeatures) -> List[Finding]:
        findings = []
        for block in features.duplicate_blocks:
            files_hit = sorted({loc[0] for loc in block.locations})
            findings.append(Finding(
                type="duplication",
                file=files_hit[0],
                severity="warning",
                description=(
                    f"Code block duplicated across {len(files_hit)} files: "
                    f"{', '.join(files_hit)}"
                ),
                evidence={"locations": [list(loc) for loc in block.locations]},
            ))
        return findings

    def _build_satd_findings(self, markers: List[SATDMarker]) -> List[Finding]:
        findings = []
        for marker in markers:
            findings.append(Finding(
                type="satd",
                file=marker.file_path,
                line=marker.line,
                severity="info",
                description=f"{marker.marker} marker: {marker.text}",
            ))
        return findings

    def _build_performance_findings(self, hotspots: List[HotspotFunction]) -> List[Finding]:
        findings = []
        for hotspot in hotspots:
            description = f"Risk score {hotspot.risk_score:.2f}: {'; '.join(hotspot.reasons)}"
            if hotspot.measured_time_seconds is not None:
                description += f" (measured {hotspot.measured_time_seconds:.3f}s cumulative)"

            findings.append(Finding(
                type="performance_hotspot",
                file=hotspot.file_path,
                line=hotspot.start_line,
                function=hotspot.function_name,
                severity="warning" if hotspot.risk_score >= 0.5 else "info",
                description=description,
                evidence={"risk_score": hotspot.risk_score},
            ))
        return findings
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_agent_scan.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add src/agent_scan.py tests/test_agent_scan.py
git commit -m "feat: add AgentPrepScanner orchestrator"
```

---

### Task 8: CLI entrypoint + `.gitignore` + dogfood run

**Files:**
- Create: `src/cli_agent_scan.py`
- Modify: `.gitignore`
- Test: `tests/test_cli_agent_scan.py`

**Interfaces:**
- Consumes: `AgentPrepScanner` (Task 7), `FindingsMarkdownWriter`/`FindingsJSONWriter` (Task 6).
- Produces: a runnable CLI, `python -m src.cli_agent_scan <source>`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cli_agent_scan.py`:

```python
"""Tests for the agent-prep scan CLI."""

import sys
import subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from click.testing import CliRunner
from cli_agent_scan import main


def _init_repo(repo_dir: Path):
    subprocess.run(["git", "init", "-q"], cwd=repo_dir, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_dir, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo_dir, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo_dir, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "initial commit"], cwd=repo_dir, check=True)


def test_cli_writes_findings_files(tmp_path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "app.py").write_text("def add(a, b):\n    return a + b\n")
    _init_repo(repo_dir)

    runner = CliRunner()
    result = runner.invoke(main, [str(repo_dir), '--quiet'])

    assert result.exit_code == 0
    assert (repo_dir / 'AI_SCAN_FINDINGS.md').exists()
    assert (repo_dir / 'ai_scan_findings.json').exists()


def test_cli_respects_output_directory_override(tmp_path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "app.py").write_text("def add(a, b):\n    return a + b\n")
    _init_repo(repo_dir)

    output_dir = tmp_path / "scan-output"

    runner = CliRunner()
    result = runner.invoke(main, [str(repo_dir), '--output', str(output_dir), '--quiet'])

    assert result.exit_code == 0
    assert (output_dir / 'AI_SCAN_FINDINGS.md').exists()
    assert not (repo_dir / 'AI_SCAN_FINDINGS.md').exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cli_agent_scan.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cli_agent_scan'`

- [ ] **Step 3: Write the implementation**

Create `src/cli_agent_scan.py`:

```python
"""Command-line interface for the agent-prep scanner."""

from pathlib import Path
import click

from agent_scan import AgentPrepScanner
from report.reporter_findings import FindingsMarkdownWriter, FindingsJSONWriter


@click.command()
@click.argument('source', type=str)
@click.option(
    '--config', '-c',
    type=click.Path(exists=True, path_type=Path),
    help='Path to config YAML file'
)
@click.option(
    '--output', '-o',
    type=click.Path(path_type=Path),
    help='Output directory for findings (defaults to the scanned repo root)'
)
@click.option(
    '--profile',
    is_flag=True,
    help='Also run an opt-in dynamic profiling pass (executes code from the scanned repo)'
)
@click.option(
    '--quiet', '-q',
    is_flag=True,
    help='Suppress progress output'
)
def main(source: str, config: Path, output: Path, profile: bool, quiet: bool):
    """Scan a repository and produce agent-readable findings.

    SOURCE can be a GitHub URL or a local path.
    """
    scanner = AgentPrepScanner(config_path=config, enable_profiling=profile)
    findings = scanner.scan(source, verbose=not quiet)

    output_dir = output if output else Path(findings.repo_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    md_path = output_dir / 'AI_SCAN_FINDINGS.md'
    json_path = output_dir / 'ai_scan_findings.json'

    FindingsMarkdownWriter().generate(findings, md_path)
    FindingsJSONWriter().generate(findings, json_path)

    if not quiet:
        click.echo("\nFindings written to:")
        click.echo(f"  {md_path}")
        click.echo(f"  {json_path}")
        click.echo(
            f"\n{len(findings.findings)} findings across "
            f"{len({f.type for f in findings.findings})} categories."
        )


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cli_agent_scan.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Add findings output to `.gitignore`**

In `.gitignore`, under the existing `# Reports` section (which already ignores `*.json`, so `ai_scan_findings.json` is already covered — only the Markdown file needs a new entry since `reports/*.md` only covers a `reports/` subdirectory, not repo-root files), add:

```
AI_SCAN_FINDINGS.md
```

- [ ] **Step 6: Run the full test suite**

Run: `pytest -v`
Expected: All tests pass, including the pre-existing `tests/test_basic.py` and `tests/test_bug.py` (this task must not break them).

- [ ] **Step 7: Dogfood — run the scanner against this repo**

Run: `python -m src.cli_agent_scan .`

Confirm `AI_SCAN_FINDINGS.md` and `ai_scan_findings.json` were written to the repo root, and read `AI_SCAN_FINDINGS.md` to sanity-check the findings look reasonable (e.g. no crash, plausible severities, real file paths). Do not commit the generated files — Step 5 already gitignores them.

- [ ] **Step 8: Commit**

```bash
git add src/cli_agent_scan.py tests/test_cli_agent_scan.py .gitignore
git commit -m "feat: add agent-prep scan CLI entrypoint"
```
