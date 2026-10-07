"""Enhanced reporters with natural language explanations."""

from pathlib import Path
from typing import Dict, Any
import json

from ..model.aggregator import RepoScore


class EnhancedJSONReporter:
    """JSON reporter with explanations."""

    def __init__(self, config: Dict = None):
        """Initialize reporter."""
        self.config = config or {}
        self.top_n_files = self.config.get('top_n_files', 10)

    def generate(self, repo_score: RepoScore, output_path: Path = None) -> Dict[str, Any]:
        """Generate enhanced JSON report with explanations."""
        report = {
            "summary": {
                "repo_path": str(repo_score.repo_path),
                "ai_probability": round(repo_score.ai_probability, 3),
                "confidence": round(repo_score.confidence, 3),
                "verdict": self._get_verdict(repo_score.ai_probability),
                "detection_mode": "enhanced",  # Indicates Phase 2+3 features
            },
            "scores": {
                "stylometry": round(repo_score.stylometry_score, 3),
                "structural": round(repo_score.structural_score, 3),
                "history": round(repo_score.history_score, 3),
            },
            "statistics": {
                "total_files": repo_score.total_files_analyzed,
                "total_lines": repo_score.total_lines_analyzed,
                "languages": repo_score.language_distribution,
            },
            "top_suspicious_files": [],
            "file_details": [],
        }

        # Add top suspicious files with details and explanations
        top_files = repo_score.file_scores[:self.top_n_files]

        for file_score in top_files:
            file_detail = {
                "path": str(file_score.file_path),
                "ai_probability": round(file_score.ai_probability, 3),
                "scores": {
                    "stylometry": round(file_score.stylometry_score, 3),
                    "structural": round(file_score.structural_score, 3),
                },
                "explanations": {
                    k: round(v, 3) for k, v in file_score.feature_explanations.items()
                },
            }

            # Add natural language explanation if available (Phase 3)
            if hasattr(file_score, 'explanation') and file_score.explanation:
                file_detail["natural_language_explanation"] = file_score.explanation

            report["file_details"].append(file_detail)
            report["top_suspicious_files"].append(str(file_score.file_path))

        # Save to file if path provided
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)

        return report

    def _get_verdict(self, probability: float) -> str:
        """Get verdict string."""
        if probability >= 0.8:
            return "Very likely AI-generated"
        elif probability >= 0.6:
            return "Likely AI-generated"
        elif probability >= 0.4:
            return "Possibly AI-assisted"
        elif probability >= 0.2:
            return "Possibly human-written"
        else:
            return "Likely human-written"


class EnhancedMarkdownReporter:
    """Markdown reporter with explanations."""

    def __init__(self, config: Dict = None):
        """Initialize reporter."""
        self.config = config or {}
        self.top_n_files = self.config.get('top_n_files', 10)

    def generate(self, repo_score: RepoScore, output_path: Path = None) -> str:
        """Generate enhanced Markdown report."""
        lines = []

        # Header
        lines.append("# AI Code Detection Report (Enhanced)")
        lines.append("")
        lines.append("*Generated with Phase 2 (ML Embeddings) + Phase 3 (Qwen Explanations)*")
        lines.append("")

        # Summary
        lines.append("## Summary")
        lines.append("")
        lines.append(f"**Repository:** `{repo_score.repo_path}`")
        lines.append("")

        # Verdict
        verdict = self._get_verdict(repo_score.ai_probability)
        prob_percent = repo_score.ai_probability * 100
        conf_percent = repo_score.confidence * 100

        lines.append(f"**AI Probability:** {prob_percent:.1f}%")
        lines.append(f"**Confidence:** {conf_percent:.1f}%")
        lines.append(f"**Verdict:** {verdict}")
        lines.append("")

        # Visual bar
        bar = self._create_probability_bar(repo_score.ai_probability)
        lines.append(bar)
        lines.append("")

        # Component scores
        lines.append("## Component Scores")
        lines.append("")
        lines.append("| Component | Score | Bar |")
        lines.append("|-----------|-------|-----|")

        components = [
            ("Stylometry", repo_score.stylometry_score),
            ("Structural", repo_score.structural_score),
            ("History", repo_score.history_score),
        ]

        for name, score in components:
            bar = self._create_small_bar(score)
            lines.append(f"| {name} | {score*100:.1f}% | {bar} |")

        lines.append("")

        # Statistics
        lines.append("## Repository Statistics")
        lines.append("")
        lines.append(f"- **Files Analyzed:** {repo_score.total_files_analyzed}")
        lines.append(f"- **Lines of Code:** {repo_score.total_lines_analyzed:,}")
        lines.append("")

        # Language distribution
        if repo_score.language_distribution:
            lines.append("**Languages:**")
            lines.append("")
            for lang, count in sorted(repo_score.language_distribution.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"- {lang}: {count} files")
            lines.append("")

        # Top suspicious files
        lines.append("## Top Suspicious Files")
        lines.append("")

        if repo_score.file_scores:
            for idx, file_score in enumerate(repo_score.file_scores[:self.top_n_files], 1):
                prob = file_score.ai_probability * 100

                lines.append(f"### {idx}. `{file_score.file_path}` ({prob:.1f}% AI)")
                lines.append("")

                # Natural language explanation (Phase 3)
                if hasattr(file_score, 'explanation') and file_score.explanation:
                    lines.append("**Analysis:**")
                    lines.append("")
                    lines.append(f"> {file_score.explanation}")
                    lines.append("")

                # Feature breakdown
                if file_score.feature_explanations:
                    lines.append("**Key Indicators:**")
                    lines.append("")
                    for feature, value in sorted(file_score.feature_explanations.items(), key=lambda x: x[1], reverse=True):
                        feature_name = self._format_feature_name(feature)
                        lines.append(f"- **{feature_name}:** {value*100:.0f}%")
                    lines.append("")

        # Footer
        lines.append("---")
        lines.append("")
        lines.append("*This enhanced report was generated by AI Code Detector (Phase 2+3)*")
        lines.append("")
        lines.append("**Detection Features:**")
        lines.append("- Heuristic pattern analysis (Phase 1)")
        lines.append("- ML-based classification with code embeddings (Phase 2)")
        lines.append("- Natural language explanations (Phase 3)")
        lines.append("")
        lines.append("**Note:** This is probabilistic forensics. Results should be combined with code review and other verification methods.")

        report = "\n".join(lines)

        # Save to file
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report)

        return report

    def _get_verdict(self, probability: float) -> str:
        """Get verdict string."""
        if probability >= 0.8:
            return "Very likely AI-generated"
        elif probability >= 0.6:
            return "Likely AI-generated"
        elif probability >= 0.4:
            return "Possibly AI-assisted"
        elif probability >= 0.2:
            return "Possibly human-written"
        else:
            return "Likely human-written"

    def _create_probability_bar(self, probability: float) -> str:
        """Create ASCII probability bar."""
        width = 40
        filled = int(probability * width)
        empty = width - filled

        bar = "█" * filled + "░" * empty
        return f"`[{bar}] {probability*100:.1f}%`"

    def _create_small_bar(self, score: float) -> str:
        """Create small ASCII bar."""
        width = 20
        filled = int(score * width)
        empty = width - filled

        return "█" * filled + "░" * empty

    def _format_feature_name(self, feature: str) -> str:
        """Format feature name for display."""
        return feature.replace('_', ' ').title()
