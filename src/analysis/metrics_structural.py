"""Structural feature extraction for AI detection."""

import ast
import re
from typing import Dict, List, Set
from dataclasses import dataclass
from collections import Counter

from .ast_parser import FileAST, FunctionInfo, ClassInfo


@dataclass
class StructuralFeatures:
    """Structural features for AI detection."""
    # Function complexity metrics
    avg_cyclomatic_complexity: float
    complexity_to_docstring_ratio: float
    over_explained_simple_functions: float

    # Error handling patterns
    generic_exception_ratio: float
    print_error_pattern_score: float
    try_except_ratio: float

    # Dead code metrics
    unused_function_ratio: float
    unused_import_ratio: float
    unreachable_code_score: float

    # API usage patterns
    rare_api_combination_score: float
    missing_cleanup_score: float


class StructuralAnalyzer:
    """Analyzes structural code features."""

    # Generic exception patterns
    GENERIC_EXCEPTION_PATTERNS = [
        r"except\s+Exception",
        r"except\s*:",
        r"except\s+\w+\s+as\s+e\s*:",
    ]

    # Print error patterns
    PRINT_ERROR_PATTERNS = [
        r'print\s*\(\s*f?["\'].*error.*["\']',
        r'print\s*\(\s*f?["\'].*Error.*["\']',
        r'print\s*\(\s*.*\{e\}',
        r'console\.log\s*\(\s*.*error',
    ]

    # Cleanup pairs (resource allocation that should have cleanup)
    CLEANUP_PAIRS = {
        'open': ['close', 'with'],
        'connect': ['disconnect', 'close'],
        'lock': ['unlock', 'release'],
        'allocate': ['free', 'deallocate'],
    }

    def __init__(self, config: Dict = None):
        """Initialize analyzer.

        Args:
            config: Configuration dict
        """
        self.config = config or {}

    def analyze_file(self, code: str, language: str, file_ast: FileAST = None) -> StructuralFeatures:
        """Analyze structural features of a file.

        Args:
            code: Source code content
            language: Programming language
            file_ast: Parsed AST (required for full analysis)

        Returns:
            StructuralFeatures
        """
        if not file_ast or language != "python":
            # Return default features for unsupported languages
            return self._default_features(code)

        # Complexity metrics
        avg_complexity = self._calculate_avg_complexity(file_ast)
        complexity_docstring_ratio = self._calculate_complexity_docstring_ratio(file_ast)
        over_explained = self._calculate_over_explained_score(file_ast)

        # Error handling
        generic_exception_ratio = self._calculate_generic_exception_ratio(code)
        print_error_score = self._calculate_print_error_score(code)
        try_except_ratio = self._calculate_try_except_ratio(file_ast)

        # Dead code
        unused_func_ratio = self._calculate_unused_function_ratio(file_ast)
        unused_import_ratio = self._calculate_unused_import_ratio(code, file_ast)
        unreachable_score = self._calculate_unreachable_code_score(file_ast)

        # API patterns
        rare_api_score = self._calculate_rare_api_score(file_ast)
        cleanup_score = self._calculate_cleanup_score(code)

        return StructuralFeatures(
            avg_cyclomatic_complexity=avg_complexity,
            complexity_to_docstring_ratio=complexity_docstring_ratio,
            over_explained_simple_functions=over_explained,
            generic_exception_ratio=generic_exception_ratio,
            print_error_pattern_score=print_error_score,
            try_except_ratio=try_except_ratio,
            unused_function_ratio=unused_func_ratio,
            unused_import_ratio=unused_import_ratio,
            unreachable_code_score=unreachable_score,
            rare_api_combination_score=rare_api_score,
            missing_cleanup_score=cleanup_score,
        )

    def _default_features(self, code: str) -> StructuralFeatures:
        """Return default features when AST not available."""
        # Use simple regex-based analysis
        generic_exception_ratio = self._calculate_generic_exception_ratio(code)
        print_error_score = self._calculate_print_error_score(code)
        cleanup_score = self._calculate_cleanup_score(code)

        return StructuralFeatures(
            avg_cyclomatic_complexity=0.0,
            complexity_to_docstring_ratio=0.0,
            over_explained_simple_functions=0.0,
            generic_exception_ratio=generic_exception_ratio,
            print_error_pattern_score=print_error_score,
            try_except_ratio=0.0,
            unused_function_ratio=0.0,
            unused_import_ratio=0.0,
            unreachable_code_score=0.0,
            rare_api_combination_score=0.0,
            missing_cleanup_score=cleanup_score,
        )

    def _calculate_avg_complexity(self, file_ast: FileAST) -> float:
        """Calculate average cyclomatic complexity."""
        functions = file_ast.functions
        for cls in file_ast.classes:
            functions.extend(cls.methods)

        if not functions:
            return 0.0

        return sum(f.cyclomatic_complexity for f in functions) / len(functions)

    def _calculate_complexity_docstring_ratio(self, file_ast: FileAST) -> float:
        """Calculate ratio of complexity to docstring length.

        High value suggests simple code with verbose docs (AI pattern).
        """
        functions = file_ast.functions
        for cls in file_ast.classes:
            functions.extend(cls.methods)

        if not functions:
            return 0.0

        ratios = []
        for func in functions:
            docstring_len = len(func.docstring) if func.docstring else 0
            if docstring_len > 0:
                ratio = docstring_len / max(func.cyclomatic_complexity, 1)
                ratios.append(ratio)

        return sum(ratios) / len(ratios) if ratios else 0.0

    def _calculate_over_explained_score(self, file_ast: FileAST) -> float:
        """Score for simple functions with long docstrings.

        Returns 0-1, higher means more over-explanation.
        """
        functions = file_ast.functions
        for cls in file_ast.classes:
            functions.extend(cls.methods)

        if not functions:
            return 0.0

        over_explained_count = 0

        for func in functions:
            # Simple function: complexity <= 3
            if func.cyclomatic_complexity <= 3:
                docstring_len = len(func.docstring) if func.docstring else 0
                # Long docstring: > 200 chars
                if docstring_len > 200:
                    over_explained_count += 1

        return over_explained_count / len(functions)

    def _calculate_generic_exception_ratio(self, code: str) -> float:
        """Calculate ratio of generic exception handling."""
        all_excepts = len(re.findall(r'\bexcept\b', code))
        if all_excepts == 0:
            return 0.0

        generic_count = 0
        for pattern in self.GENERIC_EXCEPTION_PATTERNS:
            generic_count += len(re.findall(pattern, code))

        return min(1.0, generic_count / all_excepts)

    def _calculate_print_error_score(self, code: str) -> float:
        """Score for print-based error handling patterns."""
        matches = 0
        for pattern in self.PRINT_ERROR_PATTERNS:
            matches += len(re.findall(pattern, code, re.IGNORECASE))

        # Normalize by file size (per 100 lines)
        lines = code.count('\n') + 1
        return min(1.0, (matches * 100) / max(lines, 1))

    def _calculate_try_except_ratio(self, file_ast: FileAST) -> float:
        """Calculate ratio of functions using try/except."""
        functions = file_ast.functions
        for cls in file_ast.classes:
            functions.extend(cls.methods)

        if not functions:
            return 0.0

        try_count = sum(1 for f in functions if 'try' in f.code.lower())
        return try_count / len(functions)

    def _calculate_unused_function_ratio(self, file_ast: FileAST) -> float:
        """Estimate ratio of unused functions.

        Simple heuristic: function defined but name never appears elsewhere.
        """
        functions = file_ast.functions
        if not functions:
            return 0.0

        # Collect all function names
        func_names = {f.name for f in functions}

        # Collect all code outside function definitions
        # This is a simplified check - doesn't parse call sites perfectly
        all_code = ' '.join(f.code for f in functions)

        unused_count = 0
        for func in functions:
            # Check if function name appears in other code
            # Subtract this function's own definition
            occurrences = all_code.count(func.name)
            # Function name appears at least once in its own definition
            if occurrences <= 2:  # Appears only in def and maybe once more
                unused_count += 1

        return unused_count / len(functions)

    def _calculate_unused_import_ratio(self, code: str, file_ast: FileAST) -> float:
        """Estimate ratio of unused imports."""
        if not file_ast.imports:
            return 0.0

        unused_count = 0

        for imp in file_ast.imports:
            # Get the actual imported name
            parts = imp.split('.')
            name = parts[-1]

            # Check if name appears in code (simple heuristic)
            if name not in code or code.count(name) <= 1:
                unused_count += 1

        return unused_count / len(file_ast.imports)

    def _calculate_unreachable_code_score(self, file_ast: FileAST) -> float:
        """Detect unreachable code patterns.

        Simple patterns: code after return/break/continue in same block.
        """
        functions = file_ast.functions
        for cls in file_ast.classes:
            functions.extend(cls.methods)

        if not functions:
            return 0.0

        unreachable_count = 0

        for func in functions:
            lines = func.code.split('\n')
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith('return ') and i < len(lines) - 1:
                    # Check if there's non-empty code after return
                    remaining = lines[i+1:]
                    has_code_after = any(l.strip() and not l.strip().startswith('#') for l in remaining)
                    if has_code_after:
                        unreachable_count += 1
                        break

        return unreachable_count / len(functions) if functions else 0.0

    def _calculate_rare_api_score(self, file_ast: FileAST) -> float:
        """Score for rare/unusual API usage patterns.

        This is a placeholder - would need library-specific knowledge.
        For now, returns 0.
        """
        # TODO: Implement with common library patterns
        return 0.0

    def _calculate_cleanup_score(self, code: str) -> float:
        """Score for missing cleanup patterns.

        Checks for resource allocation without proper cleanup.
        Returns 0-1, higher means more missing cleanup.
        """
        missing_count = 0
        total_allocations = 0

        for allocate, cleanups in self.CLEANUP_PAIRS.items():
            # Find allocation calls
            allocations = re.findall(rf'\b{allocate}\b\s*\(', code)
            total_allocations += len(allocations)

            if allocations:
                # Check if any cleanup is present
                has_cleanup = any(cleanup in code for cleanup in cleanups)
                if not has_cleanup:
                    missing_count += len(allocations)

        if total_allocations == 0:
            return 0.0

        return min(1.0, missing_count / total_allocations)
