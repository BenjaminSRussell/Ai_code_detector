"""Main AI code detector class."""

from pathlib import Path
from typing import Dict, List, Optional
import yaml

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # Fallback: no progress bar
    def tqdm(iterable, **kwargs):
        return iterable

from ingest.git_loader import GitLoader, RepoInfo
from ingest.file_filter import FileFilter, FileInfo
from analysis.tokenizer import CodeTokenizer
from analysis.ast_parser import ASTParserFactory, FileAST
from analysis.metrics_stylometry import StylometryAnalyzer, StylometricFeatures
from analysis.metrics_structural import StructuralAnalyzer, StructuralFeatures
from analysis.metrics_history import HistoryAnalyzer, HistoryFeatures
from model.aggregator import HeuristicAggregator, FileScore, RepoScore


class AICodeDetector:
    """Main detector class that orchestrates the analysis pipeline."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize detector.

        Args:
            config_path: Path to config YAML file
        """
        # Load configuration
        if config_path and config_path.exists():
            with open(config_path) as f:
                self.config = yaml.safe_load(f)
        else:
            # Use default config
            self.config = self._default_config()

        # Initialize components
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

        self.aggregator = HeuristicAggregator(
            config=self.config.get('scoring', {})
        )

    def analyze_repo(self, source: str, verbose: bool = True) -> RepoScore:
        """Analyze repository for AI-generated code.

        Args:
            source: GitHub URL or local path
            verbose: Show progress bars

        Returns:
            RepoScore with detection results
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

        # Step 3: Analyze files
        file_scores = []
        total_lines = 0

        iterator = tqdm(files, desc="Analyzing files") if verbose else files

        for file_info in iterator:
            try:
                file_score = self._analyze_file(file_info, repo_info.path)
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

        repo_score = self.aggregator.aggregate_repo_features(
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

        return repo_score

    def _analyze_file(self, file_info: FileInfo, repo_root: Path) -> FileScore:
        """Analyze a single file.

        Args:
            file_info: File information
            repo_root: Repository root path

        Returns:
            FileScore
        """
        # Read file
        try:
            with open(file_info.path, 'r', encoding='utf-8', errors='ignore') as f:
                code = f.read()
        except Exception:
            # Return low-confidence score for unreadable files
            return FileScore(
                file_path=str(file_info.relative_path),
                ai_probability=0.0,
                stylometry_score=0.0,
                structural_score=0.0,
                feature_explanations={},
                suspicious_snippets=[],
            )

        # Parse AST if supported
        file_ast = None
        parser = ASTParserFactory.get_parser(file_info.language)
        if parser:
            try:
                file_ast = parser.parse_file(file_info.path, code)
            except Exception:
                # Return low-confidence score for files that fail to parse
                return FileScore(
                    file_path=str(file_info.relative_path),
                    ai_probability=0.0,
                    stylometry_score=0.0,
                    structural_score=0.0,
                    feature_explanations={"error": "AST parsing failed"},
                    suspicious_snippets=[],
                )

        # Extract features
        stylometry_features = self.stylometry_analyzer.analyze_file(
            code, file_info.language, file_ast
        )

        structural_features = self.structural_analyzer.analyze_file(
            code, file_info.language, file_ast
        )

        # Aggregate into file score
        file_score = self.aggregator.aggregate_file_features(
            stylometry=stylometry_features,
            structural=structural_features,
        )

        # Set file path
        file_score.file_path = str(file_info.relative_path)

        return file_score

    def _empty_result(self, repo_path: Path) -> RepoScore:
        """Create empty result for repos with no files."""
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
