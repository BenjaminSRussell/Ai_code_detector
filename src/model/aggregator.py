"""Feature aggregation and scoring for AI detection."""

from typing import Dict, List, Tuple
from dataclasses import dataclass
import numpy as np

from ..analysis.metrics_stylometry import StylometricFeatures
from ..analysis.metrics_structural import StructuralFeatures
from ..analysis.metrics_history import HistoryFeatures


@dataclass
class FileScore:
    """AI likelihood score for a file."""
    file_path: str
    ai_probability: float
    stylometry_score: float
    structural_score: float
    feature_explanations: Dict[str, float]
    suspicious_snippets: List[Dict]


@dataclass
class RepoScore:
    """AI likelihood score for entire repository."""
    repo_path: str
    ai_probability: float
    confidence: float

    # Component scores
    stylometry_score: float
    structural_score: float
    history_score: float

    # File-level results
    file_scores: List[FileScore]
    top_suspicious_files: List[str]

    # Summary statistics
    total_files_analyzed: int
    total_lines_analyzed: int
    language_distribution: Dict[str, int]


class HeuristicAggregator:
    """Aggregates features into AI probability scores using heuristics.

    Phase 1 implementation - uses weighted heuristics.
    Phase 2 will replace with learned classifier.
    """

    def __init__(self, config: Dict = None):
        """Initialize aggregator.

        Args:
            config: Configuration with weights and thresholds
        """
        self.config = config or {}

        # Default weights (can be tuned)
        weights = self.config.get('weights', {})
        self.stylometry_weight = weights.get('stylometry', 0.4)
        self.structural_weight = weights.get('structural', 0.4)
        self.history_weight = weights.get('history', 0.2)

        # Thresholds
        thresholds = self.config.get('thresholds', {})
        self.file_threshold = thresholds.get('file_threshold', 0.6)
        self.repo_threshold = thresholds.get('repo_threshold', 0.5)

    def aggregate_file_features(
        self,
        stylometry: StylometricFeatures,
        structural: StructuralFeatures,
    ) -> FileScore:
        """Aggregate file-level features into AI probability.

        Args:
            stylometry: Stylometric features
            structural: Structural features

        Returns:
            FileScore with AI probability
        """
        # Calculate component scores
        stylometry_score = self._score_stylometry(stylometry)
        structural_score = self._score_structural(structural)

        # Weighted combination
        ai_probability = (
            self.stylometry_weight * stylometry_score +
            self.structural_weight * structural_score
        )

        # Normalize to 0-1
        ai_probability = max(0.0, min(1.0, ai_probability))

        # Extract feature explanations (top contributors)
        explanations = self._extract_explanations(stylometry, structural)

        return FileScore(
            file_path="",  # Set by caller
            ai_probability=ai_probability,
            stylometry_score=stylometry_score,
            structural_score=structural_score,
            feature_explanations=explanations,
            suspicious_snippets=[],  # TODO: Add snippet extraction
        )

    def aggregate_repo_features(
        self,
        file_scores: List[FileScore],
        history: HistoryFeatures,
        total_lines: int,
        language_dist: Dict[str, int],
    ) -> RepoScore:
        """Aggregate repository-level features into AI probability.

        Args:
            file_scores: List of FileScore for all files
            history: Repository history features
            total_lines: Total lines of code analyzed
            language_dist: Distribution of languages

        Returns:
            RepoScore with overall AI probability
        """
        if not file_scores:
            return self._empty_repo_score()

        # Calculate file-level statistics
        file_probs = [fs.ai_probability for fs in file_scores]
        mean_file_prob = np.mean(file_probs)
        max_file_prob = np.max(file_probs)
        median_file_prob = np.median(file_probs)

        # History score
        history_score = self._score_history(history)

        # Aggregate scores
        # Use combination of mean and max (high max with high mean is stronger signal)
        file_level_score = 0.6 * mean_file_prob + 0.4 * max_file_prob

        # Final repo probability
        repo_probability = (
            0.7 * file_level_score +
            0.3 * history_score
        )

        # Confidence based on consistency
        file_std = np.std(file_probs)
        confidence = 1.0 - min(file_std, 1.0)  # Lower variance = higher confidence

        # Find top suspicious files
        sorted_files = sorted(file_scores, key=lambda x: x.ai_probability, reverse=True)
        top_files = [fs.file_path for fs in sorted_files[:10]]

        # Average component scores
        avg_stylometry = np.mean([fs.stylometry_score for fs in file_scores])
        avg_structural = np.mean([fs.structural_score for fs in file_scores])

        return RepoScore(
            repo_path="",  # Set by caller
            ai_probability=repo_probability,
            confidence=confidence,
            stylometry_score=avg_stylometry,
            structural_score=avg_structural,
            history_score=history_score,
            file_scores=file_scores,
            top_suspicious_files=top_files,
            total_files_analyzed=len(file_scores),
            total_lines_analyzed=total_lines,
            language_distribution=language_dist,
        )

    def _score_stylometry(self, features: StylometricFeatures) -> float:
        """Convert stylometric features to 0-1 score.

        Higher score = more likely AI.
        """
        score = 0.0

        # Comment features (30%)
        # High comment ratio with boilerplate = AI
        if features.comment_to_code_ratio > 0.3:
            score += 0.1
        score += 0.1 * features.boilerplate_comment_score
        score += 0.1 * features.tutorial_comment_score

        # Naming features (30%)
        # Generic names + low entropy = AI
        score += 0.15 * features.generic_name_ratio
        if features.identifier_entropy < 2.0:
            score += 0.15

        # Formatting features (20%)
        # Perfect consistency might be AI
        if features.indentation_consistency > 0.95:
            score += 0.1
        if features.trailing_whitespace_ratio < 0.01:
            score += 0.1

        # Duplication features (20%)
        score += 0.1 * features.code_duplication_score
        score += 0.1 * features.intra_file_similarity

        return min(1.0, score)

    def _score_structural(self, features: StructuralFeatures) -> float:
        """Convert structural features to 0-1 score.

        Higher score = more likely AI.
        """
        score = 0.0

        # Complexity vs documentation (30%)
        # Over-explained simple code = AI
        score += 0.15 * features.over_explained_simple_functions
        if features.avg_cyclomatic_complexity < 3.0 and features.complexity_to_docstring_ratio > 50:
            score += 0.15

        # Error handling (30%)
        score += 0.15 * features.generic_exception_ratio
        score += 0.15 * features.print_error_pattern_score

        # Dead code (25%)
        score += 0.1 * features.unused_function_ratio
        score += 0.1 * features.unused_import_ratio
        score += 0.05 * features.unreachable_code_score

        # Missing cleanup (15%)
        score += 0.15 * features.missing_cleanup_score

        return min(1.0, score)

    def _score_history(self, features: HistoryFeatures) -> float:
        """Convert history features to 0-1 score.

        Higher score = more likely AI.
        """
        score = 0.0

        # Commit patterns (40%)
        score += 0.2 * features.commit_burst_score
        score += 0.1 * features.commit_gini_coefficient
        if features.files_created_in_burst > 0.7:
            score += 0.1

        # Author patterns (30%)
        # Low diversity + generic messages = AI
        if features.author_diversity < 0.3:
            score += 0.15
        score += 0.15 * features.generic_message_ratio

        # Time patterns (30%)
        # Very young repo or very dense commits = AI
        if features.repo_age_days < 7:
            score += 0.15
        if features.commits_per_day > 10:
            score += 0.15

        return min(1.0, score)

    def _extract_explanations(
        self,
        stylometry: StylometricFeatures,
        structural: StructuralFeatures,
    ) -> Dict[str, float]:
        """Extract top contributing features for explanation.

        Returns dict of feature_name -> normalized_contribution.
        """
        explanations = {}

        # Stylometry
        if stylometry.boilerplate_comment_score > 0.5:
            explanations['boilerplate_comments'] = stylometry.boilerplate_comment_score
        if stylometry.generic_name_ratio > 0.3:
            explanations['generic_naming'] = stylometry.generic_name_ratio
        if stylometry.code_duplication_score > 0.4:
            explanations['code_duplication'] = stylometry.code_duplication_score

        # Structural
        if structural.over_explained_simple_functions > 0.3:
            explanations['over_explained_functions'] = structural.over_explained_simple_functions
        if structural.generic_exception_ratio > 0.5:
            explanations['generic_exceptions'] = structural.generic_exception_ratio
        if structural.unused_function_ratio > 0.3:
            explanations['unused_functions'] = structural.unused_function_ratio

        return explanations

    def _empty_repo_score(self) -> RepoScore:
        """Return empty repo score."""
        return RepoScore(
            repo_path="",
            ai_probability=0.0,
            confidence=0.0,
            stylometry_score=0.0,
            structural_score=0.0,
            history_score=0.0,
            file_scores=[],
            top_suspicious_files=[],
            total_files_analyzed=0,
            total_lines_analyzed=0,
            language_distribution={},
        )
