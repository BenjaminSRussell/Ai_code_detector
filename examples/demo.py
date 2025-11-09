"""Demo script showing programmatic usage of AI Code Detector."""

from pathlib import Path
from src.detector import AICodeDetector
from src.report.reporter_json import JSONReporter
from src.report.reporter_markdown import MarkdownReporter


def main():
    """Run demo analysis."""
    print("AI Code Detector - Demo")
    print("=" * 60)
    print()

    # Initialize detector with default config
    detector = AICodeDetector()

    # Example 1: Analyze current directory
    print("Example 1: Analyze current repository")
    print("-" * 60)

    current_repo = Path.cwd()
    repo_score = detector.analyze_repo(str(current_repo), verbose=True)

    print()
    print(f"AI Probability: {repo_score.ai_probability * 100:.1f}%")
    print(f"Confidence: {repo_score.confidence * 100:.1f}%")
    print()

    # Example 2: Generate reports
    print("Example 2: Generate reports")
    print("-" * 60)

    output_dir = Path("examples/output")
    output_dir.mkdir(exist_ok=True)

    # JSON report
    json_reporter = JSONReporter()
    json_path = output_dir / "demo_report.json"
    json_report = json_reporter.generate(repo_score, json_path)
    print(f"JSON report saved to: {json_path}")

    # Markdown report
    md_reporter = MarkdownReporter()
    md_path = output_dir / "demo_report.md"
    md_report = md_reporter.generate(repo_score, md_path)
    print(f"Markdown report saved to: {md_path}")

    print()

    # Example 3: Inspect top suspicious files
    print("Example 3: Top suspicious files")
    print("-" * 60)

    for i, file_path in enumerate(repo_score.top_suspicious_files[:5], 1):
        file_score = next(
            (fs for fs in repo_score.file_scores if fs.file_path == file_path),
            None
        )

        if file_score:
            print(f"{i}. {file_path}")
            print(f"   AI Probability: {file_score.ai_probability * 100:.1f}%")

            if file_score.feature_explanations:
                print("   Key indicators:")
                for feature, value in list(file_score.feature_explanations.items())[:3]:
                    print(f"     - {feature}: {value * 100:.1f}%")

            print()

    # Example 4: Programmatic decision-making
    print("Example 4: Programmatic decision-making")
    print("-" * 60)

    if repo_score.ai_probability >= 0.8:
        print("🔴 Action: Flag for manual review (Very high AI likelihood)")
    elif repo_score.ai_probability >= 0.6:
        print("🟠 Action: Request explanation from author")
    elif repo_score.ai_probability >= 0.4:
        print("🟡 Action: Note for future reference")
    else:
        print("✅ Action: No action needed")

    print()
    print("=" * 60)
    print("Demo complete!")


if __name__ == '__main__':
    main()
