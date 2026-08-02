"""Orchestrates the agent-prep scan pipeline.

Runs the existing AI-likelihood detector plus the repo-wide duplication,
AI-attribution, SATD, and performance-hotspot analyzers, and produces a
ScanFindings object for the report writers to render.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

        if config_path is None:
            packaged_default = Path(__file__).parent.parent / "configs" / "default.yaml"
            if packaged_default.exists():
                config_path = packaged_default

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
                if not timings and verbose:
                    print(
                        "Profiling produced no results (timeout, error, or no "
                        "matching functions were found)."
                    )
                performance_hotspots = merge_profiling_results(performance_hotspots, timings)
            elif verbose:
                print("No tests/ directory found — skipping dynamic profiling.")

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
        # Overlapping n-gram windows over the same duplicated region each produce
        # their own DuplicateBlock. Group blocks by the set of files they hit so
        # a single genuine duplication collapses into one finding instead of one
        # per overlapping window.
        grouped: Dict[Tuple[str, ...], List[Tuple[str, int]]] = {}
        for block in features.duplicate_blocks:
            files_hit = tuple(sorted({loc[0] for loc in block.locations}))
            merged_locations = grouped.setdefault(files_hit, [])
            for loc in block.locations:
                if loc not in merged_locations:
                    merged_locations.append(loc)

        findings = []
        for files_hit, locations in grouped.items():
            findings.append(Finding(
                type="duplication",
                file=files_hit[0],
                severity="warning",
                description=(
                    f"Code block duplicated across {len(files_hit)} files: "
                    f"{', '.join(files_hit)}"
                ),
                evidence={"locations": [list(loc) for loc in locations]},
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
