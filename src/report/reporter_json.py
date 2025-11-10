"""JSON report generator."""

import json
from pathlib import Path
from typing import Dict, Any

from model.aggregator import RepoScore, FileScore


class JSONReporter:
    """Generates JSON reports of AI detection results."""

    def __init__(self, config: Dict = None):
        """Initialize reporter.

        Args:
            config: Configuration dict
        """
        self.config = config or {}
        self.top_n_files = self.config.get('top_n_files', 10)

    def generate(self, repo_score: RepoScore, output_path: Path = None) -> Dict[str, Any]:
        """Generate JSON report.

        Args:
            repo_score: RepoScore with detection results
            output_path: Optional path to save JSON file

        Returns:
            Report dict
        """
        report = {
            "summary": {
                "repo_path": str(repo_score.repo_path),
                "ai_probability": round(repo_score.ai_probability, 3),
                "confidence": round(repo_score.confidence, 3),
                "verdict": self._get_verdict(repo_score.ai_probability),
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

        # Add top suspicious files with details
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
            report["file_details"].append(file_detail)
            report["top_suspicious_files"].append(str(file_score.file_path))

        # Save to file if path provided
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)

        return report

    def _get_verdict(self, probability: float) -> str:
        """Get human-readable verdict based on probability.

        Args:
            probability: AI probability (0-1)

        Returns:
            Verdict string
        """
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
