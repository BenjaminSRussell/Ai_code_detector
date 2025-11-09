"""Enhanced detector integrating heuristics, ML classifier, and explanations."""

from pathlib import Path
from typing import Dict, List, Optional
import yaml

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kwargs):
        return iterable

from .ingest.git_loader import GitLoader, RepoInfo
from .ingest.file_filter import FileFilter, FileInfo
from .analysis.tokenizer import CodeTokenizer
from .analysis.ast_parser import ASTParserFactory, FileAST
from .analysis.metrics_stylometry import StylometryAnalyzer, StylometricFeatures
from .analysis.metrics_structural import StructuralAnalyzer, StructuralFeatures
from .analysis.metrics_history import HistoryAnalyzer, HistoryFeatures
from .model.aggregator import HeuristicAggregator, FileScore, RepoScore

# Phase 2 & 3 imports
from .model.embedder_mlx import get_embedder, CodeEmbedder
from .model.classifier import MLClassifier, EnsembleClassifier
from .model.explainer import get_explainer, ExplanationGenerator


class EnhancedAICodeDetector:
    """Detector with heuristics, ML classifier, and explanations."""

    def __init__(
        self,
        config_path: Optional[Path] = None,
        use_ml: bool = True,
        use_explanations: bool = True,
        embedder_backend: str = "hash",  # 'mlx' or 'hash'
        explainer_backend: str = "template",  # 'qwen' or 'template'
        ml_model_path: Optional[Path] = None,
        qwen_model_path: Optional[str] = None,
    ):
        """Initialize enhanced detector.

        Args:
            config_path: Path to config YAML
            use_ml: Whether to use ML classifier (Phase 2)
            use_explanations: Whether to generate explanations (Phase 3)
            embedder_backend: Embedding backend ('mlx' or 'hash')
            explainer_backend: Explanation backend ('qwen' or 'template')
            ml_model_path: Path to trained ML model
            qwen_model_path: Path to Qwen model
        """
        # Load configuration
        if config_path and config_path.exists():
            with open(config_path) as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._default_config()

        # Phase 1 components
        self.git_loader = GitLoader()

        self.file_filter = FileFilter(
            supported_extensions=self.config['ingestion']['supported_extensions'],
            excluded_dirs=self.config['ingestion']['excluded_dirs'],
            max_file_size_mb=self.config['ingestion']['max_file_size_mb'],
        )

        self.tokenizer = CodeTokenizer()

        self.stylometry_analyzer = StylometryAnalyzer(
            config=self.config.get('features', {}).get('stylometry', {})
        )

        self.structural_analyzer = StructuralAnalyzer(
            config=self.config.get('features', {}).get('structural', {})
        )

        self.history_analyzer = HistoryAnalyzer(
            config=self.config.get('features', {}).get('history', {})
        )

        self.heuristic_aggregator = HeuristicAggregator(
            config=self.config.get('scoring', {})
        )

        # Phase 2 components
        self.use_ml = use_ml
        self.embedder = None
        self.ml_classifier = None

        if use_ml:
            self.embedder = get_embedder(
                backend=embedder_backend,
                model_path=None,  # Use default
            )

            self.ml_classifier = MLClassifier(model_path=ml_model_path)

        # Phase 3 components
        self.use_explanations = use_explanations
        self.explainer = None

        if use_explanations:
            self.explainer = get_explainer(
                backend=explainer_backend,
                model_path=qwen_model_path,
            )

    def analyze_repo(self, source: str, verbose: bool = True) -> RepoScore:
        """Analyze repository with enhanced detection.

        Args:
            source: GitHub URL or local path
            verbose: Show progress

        Returns:
            RepoScore with enhanced analysis
        """
        # Step 1: Load repository
        if verbose:
            print(f"Loading repository: {source}")

        repo_info = self.git_loader.load(source)

        if verbose:
            print(f"Repository path: {repo_info.path}")
            if repo_info.is_git:
                print(f"Git repository with {repo_info.total_commits} commits")

        # Step 2: Scan files
        if verbose:
            print("Scanning for code files...")

        files = self.file_filter.scan_directory(repo_info.path)

        if not files:
            print("No code files found!")
            return self._empty_result(repo_info.path)

        if verbose:
            lang_dist = self.file_filter.get_language_distribution(files)
            print(f"Found {len(files)} code files")
            print(f"Languages: {dict(lang_dist)}")

        # Step 3: Analyze files (with enhanced features)
        file_scores = []
        total_lines = 0

        iterator = tqdm(files, desc="Analyzing files") if verbose else files

        for file_info in iterator:
            try:
                file_score = self._analyze_file_enhanced(file_info, repo_info.path)
                file_scores.append(file_score)
                total_lines += file_info.line_count
            except Exception as e:
                if verbose:
                    print(f"Error analyzing {file_info.path}: {e}")
                continue

        # Step 4: Analyze repository history
        if verbose:
            print("Analyzing git history...")

        history_features = self.history_analyzer.analyze_repo(repo_info)

        # Step 5: Aggregate into repository score
        if verbose:
            print("Computing final scores...")

        lang_dist = self.file_filter.get_language_distribution(files)

        repo_score = self.heuristic_aggregator.aggregate_repo_features(
            file_scores=file_scores,
            history=history_features,
            total_lines=total_lines,
            language_dist=lang_dist,
        )

        # Set repo path
        repo_score.repo_path = str(repo_info.path)

        if verbose:
            print(f"\nAnalysis complete!")
            print(f"AI Probability: {repo_score.ai_probability*100:.1f}%")

            if self.use_ml:
                print("  (Enhanced with ML classifier)")
            if self.use_explanations:
                print("  (Includes natural language explanations)")

        return repo_score

    def _analyze_file_enhanced(self, file_info: FileInfo, repo_root: Path) -> FileScore:
        """Analyze file with enhanced features (Phase 2 & 3).

        Args:
            file_info: File information
            repo_root: Repository root

        Returns:
            Enhanced FileScore
        """
        # Read file
        try:
            with open(file_info.path, 'r', encoding='utf-8', errors='ignore') as f:
                code = f.read()
        except Exception:
            return self._empty_file_score(file_info.relative_path)

        # Parse AST
        file_ast = None
        parser = ASTParserFactory.get_parser(file_info.language)
        if parser:
            try:
                file_ast = parser.parse_file(file_info.path, code)
            except Exception:
                pass

        # Phase 1: Extract features
        stylometry_features = self.stylometry_analyzer.analyze_file(
            code, file_info.language, file_ast
        )

        structural_features = self.structural_analyzer.analyze_file(
            code, file_info.language, file_ast
        )

        # Get heuristic score
        file_score = self.heuristic_aggregator.aggregate_file_features(
            stylometry=stylometry_features,
            structural=structural_features,
        )

        # Phase 2: ML enhancement
        if self.use_ml and self.embedder and self.ml_classifier:
            # Generate embedding
            embedding = self.embedder.embed(code)

            # Combine features into vector
            feature_vector = self._features_to_vector(
                stylometry_features,
                structural_features,
            )

            # Get ML prediction
            ml_prob = self.ml_classifier.predict(embedding, feature_vector)

            # Ensemble: combine heuristic + ML
            ensemble_prob = 0.4 * file_score.ai_probability + 0.6 * ml_prob
            file_score.ai_probability = ensemble_prob

        # Phase 3: Generate explanation
        if self.use_explanations and self.explainer and file_score.ai_probability > 0.5:
            explanation = self.explainer.explain(
                code=code,
                ai_probability=file_score.ai_probability,
                features=file_score.feature_explanations,
                top_n=3,
            )

            # Add explanation to file score
            file_score.explanation = explanation
        else:
            file_score.explanation = None

        # Set file path
        file_score.file_path = str(file_info.relative_path)

        return file_score

    def _features_to_vector(
        self,
        stylometry: StylometricFeatures,
        structural: StructuralFeatures,
    ) -> List[float]:
        """Convert features to vector for ML classifier.

        Args:
            stylometry: Stylometric features
            structural: Structural features

        Returns:
            Feature vector
        """
        return [
            # Stylometry (12)
            stylometry.comment_to_code_ratio,
            stylometry.avg_comment_length / 100.0,  # Normalize
            stylometry.boilerplate_comment_score,
            stylometry.tutorial_comment_score,
            stylometry.avg_identifier_length / 10.0,
            stylometry.generic_name_ratio,
            stylometry.identifier_entropy / 5.0,
            stylometry.indentation_consistency,
            stylometry.avg_line_length / 100.0,
            stylometry.trailing_whitespace_ratio,
            stylometry.code_duplication_score,
            stylometry.intra_file_similarity,
            # Structural (11)
            structural.avg_cyclomatic_complexity / 10.0,
            structural.complexity_to_docstring_ratio / 100.0,
            structural.over_explained_simple_functions,
            structural.generic_exception_ratio,
            structural.print_error_pattern_score,
            structural.try_except_ratio,
            structural.unused_function_ratio,
            structural.unused_import_ratio,
            structural.unreachable_code_score,
            structural.rare_api_combination_score,
            structural.missing_cleanup_score,
            # Note: History features not included at file level
        ]

    def _empty_file_score(self, file_path: Path) -> FileScore:
        """Create empty file score."""
        return FileScore(
            file_path=str(file_path),
            ai_probability=0.0,
            stylometry_score=0.0,
            structural_score=0.0,
            feature_explanations={},
            suspicious_snippets=[],
        )

    def _empty_result(self, repo_path: Path) -> RepoScore:
        """Create empty result."""
        return RepoScore(
            repo_path=str(repo_path),
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

    def _default_config(self) -> Dict:
        """Get default configuration."""
        return {
            'ingestion': {
                'supported_extensions': ['.py', '.js', '.ts', '.go', '.rs'],
                'excluded_dirs': ['node_modules', 'dist', 'build', '.git', '__pycache__', 'venv'],
                'max_file_size_mb': 1.0,
            },
            'features': {
                'stylometry': {},
                'structural': {},
                'history': {},
            },
            'scoring': {
                'weights': {
                    'stylometry': 0.4,
                    'structural': 0.4,
                    'history': 0.2,
                },
            },
        }


# Add explanation field to FileScore
# Monkey patch the dataclass
from dataclasses import dataclass, field

original_FileScore = FileScore

@dataclass
class EnhancedFileScore(original_FileScore):
    """Enhanced file score with explanation."""
    explanation: Optional[str] = None

# Replace in module
FileScore = EnhancedFileScore
