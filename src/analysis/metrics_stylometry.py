"""Stylometric feature extraction for AI detection."""

import re
from typing import Dict, List, Set
from dataclasses import dataclass
from collections import Counter

from .tokenizer import CodeTokenizer
from .ast_parser import FileAST, FunctionInfo


@dataclass
class StylometricFeatures:
    """Stylometric features for AI detection."""
    # Comment features
    comment_to_code_ratio: float
    avg_comment_length: float
    boilerplate_comment_score: float
    tutorial_comment_score: float

    # Naming features
    avg_identifier_length: float
    generic_name_ratio: float
    identifier_entropy: float

    # Formatting features
    indentation_consistency: float
    avg_line_length: float
    trailing_whitespace_ratio: float

    # Duplication features
    code_duplication_score: float
    intra_file_similarity: float


class StylometryAnalyzer:
    """Analyzes stylometric features of code."""

    # Patterns that suggest AI-generated boilerplate
    BOILERPLATE_PATTERNS = [
        r"This function",
        r"This method",
        r"This class",
        r"In this function",
        r"Returns:",
        r"Args:",
        r"Parameters:",
        r"Example:",
        r"Note:",
    ]

    # Tutorial-style phrases
    TUTORIAL_PATTERNS = [
        r"First,? we",
        r"Then,? we",
        r"Next,? we",
        r"Finally,? we",
        r"Let's",
        r"Now we",
        r"Here we",
    ]

    # Generic variable names common in AI code
    GENERIC_NAMES = {
        "result", "data", "temp", "tmp", "value", "values",
        "item", "items", "obj", "object", "element", "elements",
        "helper", "utils", "util", "process", "handle", "handler",
        "manager", "service", "controller", "model", "view",
        "foo", "bar", "baz", "test", "example",
    }

    def __init__(self, config: Dict = None):
        """Initialize analyzer.

        Args:
            config: Configuration dict with patterns and thresholds
        """
        self.tokenizer = CodeTokenizer()
        self.config = config or {}

        # Load custom patterns if provided
        custom_boilerplate = self.config.get("comment_boilerplate_patterns", [])
        custom_generic = self.config.get("generic_names", [])

        self.boilerplate_patterns = self.BOILERPLATE_PATTERNS + custom_boilerplate
        self.generic_names = self.GENERIC_NAMES.union(set(custom_generic))

    def analyze_file(self, code: str, language: str, file_ast: FileAST = None) -> StylometricFeatures:
        """Analyze stylometric features of a file.

        Args:
            code: Source code content
            language: Programming language
            file_ast: Optional parsed AST

        Returns:
            StylometricFeatures
        """
        lines = code.split('\n')

        # Tokenize
        token_stats = self.tokenizer.tokenize(code, language)

        # Comment analysis
        comment_ratio = self._calculate_comment_ratio(code, token_stats.comment_tokens)
        avg_comment_len = self._calculate_avg_comment_length(token_stats.comment_tokens)
        boilerplate_score = self._calculate_boilerplate_score(token_stats.comment_tokens)
        tutorial_score = self._calculate_tutorial_score(token_stats.comment_tokens)

        # Naming analysis
        avg_id_len = sum(len(id) for id in token_stats.identifier_tokens) / len(token_stats.identifier_tokens) if token_stats.identifier_tokens else 0
        generic_ratio = self._calculate_generic_name_ratio(token_stats.identifier_tokens)
        id_entropy = token_stats.token_entropy

        # Formatting analysis
        indent_consistency = self._calculate_indentation_consistency(lines)
        avg_line_len = sum(len(line) for line in lines) / len(lines) if lines else 0
        trailing_ws_ratio = self._calculate_trailing_whitespace_ratio(lines)

        # Duplication analysis
        duplication_score = self._calculate_duplication_score(code, language)
        similarity = self._calculate_intra_file_similarity(file_ast) if file_ast else 0.0

        return StylometricFeatures(
            comment_to_code_ratio=comment_ratio,
            avg_comment_length=avg_comment_len,
            boilerplate_comment_score=boilerplate_score,
            tutorial_comment_score=tutorial_score,
            avg_identifier_length=avg_id_len,
            generic_name_ratio=generic_ratio,
            identifier_entropy=id_entropy,
            indentation_consistency=indent_consistency,
            avg_line_length=avg_line_len,
            trailing_whitespace_ratio=trailing_ws_ratio,
            code_duplication_score=duplication_score,
            intra_file_similarity=similarity,
        )

    def _calculate_comment_ratio(self, code: str, comments: List[str]) -> float:
        """Calculate ratio of comment lines to code lines."""
        code_lines = [line for line in code.split('\n') if line.strip()]
        if not code_lines:
            return 0.0

        comment_line_count = sum(comment.count('\n') + 1 for comment in comments)
        return comment_line_count / len(code_lines)

    def _calculate_avg_comment_length(self, comments: List[str]) -> float:
        """Calculate average comment length."""
        if not comments:
            return 0.0
        return sum(len(c) for c in comments) / len(comments)

    def _calculate_boilerplate_score(self, comments: List[str]) -> float:
        """Calculate score for boilerplate comment patterns.

        Returns value 0-1, higher means more boilerplate.
        """
        if not comments:
            return 0.0

        combined_comments = ' '.join(comments)
        matches = 0

        for pattern in self.boilerplate_patterns:
            matches += len(re.findall(pattern, combined_comments, re.IGNORECASE))

        # Normalize by comment count
        return min(1.0, matches / len(comments))

    def _calculate_tutorial_score(self, comments: List[str]) -> float:
        """Calculate score for tutorial-style comments."""
        if not comments:
            return 0.0

        combined_comments = ' '.join(comments)
        matches = 0

        for pattern in self.TUTORIAL_PATTERNS:
            matches += len(re.findall(pattern, combined_comments, re.IGNORECASE))

        return min(1.0, matches / len(comments))

    def _calculate_generic_name_ratio(self, identifiers: List[str]) -> float:
        """Calculate ratio of generic to total identifiers."""
        if not identifiers:
            return 0.0

        generic_count = sum(1 for id in identifiers if id.lower() in self.generic_names)
        return generic_count / len(identifiers)

    def _calculate_indentation_consistency(self, lines: List[str]) -> float:
        """Calculate consistency of indentation.

        Returns 1.0 for perfect consistency, lower for inconsistency.
        """
        if not lines:
            return 1.0

        # Detect indentation style
        indent_counts = Counter()

        for line in lines:
            if not line.strip():
                continue

            # Count leading spaces/tabs
            leading = len(line) - len(line.lstrip())
            if leading > 0:
                # Detect if spaces or tabs
                if line[0] == '\t':
                    indent_counts['tab'] += 1
                elif line[0] == ' ':
                    indent_counts['space'] += 1

        if not indent_counts:
            return 1.0

        # Consistency is ratio of most common to total
        total = sum(indent_counts.values())
        most_common_count = max(indent_counts.values())

        return most_common_count / total

    def _calculate_trailing_whitespace_ratio(self, lines: List[str]) -> float:
        """Calculate ratio of lines with trailing whitespace."""
        if not lines:
            return 0.0

        trailing_count = sum(1 for line in lines if line and line[-1] in ' \t')
        return trailing_count / len(lines)

    def _calculate_duplication_score(self, code: str, language: str) -> float:
        """Calculate code duplication score using n-grams.

        Returns 0-1, higher means more duplication.
        """
        lines = [line.strip() for line in code.split('\n') if line.strip()]

        if len(lines) < 4:
            return 0.0

        # Use 3-line n-grams
        n = 3
        ngrams = []

        for i in range(len(lines) - n + 1):
            ngram = tuple(lines[i:i+n])
            ngrams.append(ngram)

        if not ngrams:
            return 0.0

        # Count duplicates
        ngram_counts = Counter(ngrams)
        duplicates = sum(count - 1 for count in ngram_counts.values() if count > 1)

        return min(1.0, duplicates / len(ngrams))

    def _calculate_intra_file_similarity(self, file_ast: FileAST) -> float:
        """Calculate similarity between functions in same file.

        Returns 0-1, higher means more similar functions.
        """
        if not file_ast or len(file_ast.functions) < 2:
            return 0.0

        # Compare function signatures and structures
        similarities = []

        for i, func1 in enumerate(file_ast.functions):
            for func2 in file_ast.functions[i+1:]:
                sim = self._compare_functions(func1, func2)
                similarities.append(sim)

        return sum(similarities) / len(similarities) if similarities else 0.0

    def _compare_functions(self, func1: FunctionInfo, func2: FunctionInfo) -> float:
        """Compare two functions for similarity.

        Returns 0-1 similarity score.
        """
        score = 0.0
        comparisons = 0

        # Compare parameter count
        if abs(len(func1.params) - len(func2.params)) <= 1:
            score += 0.3
        comparisons += 1

        # Compare complexity
        if abs(func1.cyclomatic_complexity - func2.cyclomatic_complexity) <= 2:
            score += 0.3
        comparisons += 1

        # Compare code length
        len1 = len(func1.code.split('\n'))
        len2 = len(func2.code.split('\n'))
        if abs(len1 - len2) / max(len1, len2, 1) < 0.3:
            score += 0.4
        comparisons += 1

        return score / comparisons if comparisons > 0 else 0.0
