"""Command-line interface for AI code detector."""

import sys
from pathlib import Path
import click

from .detector import AICodeDetector
from .report.reporter_json import JSONReporter
from .report.reporter_markdown import MarkdownReporter


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
    help='Output directory for reports'
)
@click.option(
    '--format', '-f',
    type=click.Choice(['json', 'markdown', 'both'], case_sensitive=False),
    default='both',
    help='Report format'
)
@click.option(
    '--quiet', '-q',
    is_flag=True,
    help='Suppress progress output'
)
def main(source: str, config: Path, output: Path, format: str, quiet: bool):
    """Analyze a GitHub repository or local directory for AI-generated code.

    SOURCE can be:
      - GitHub URL (e.g., https://github.com/user/repo)
      - Local path (e.g., /path/to/repo)

    Examples:
      ai-code-detector https://github.com/user/repo
      ai-code-detector /path/to/local/repo --format json
      ai-code-detector ./my_project -o ./reports --config custom.yaml
    """
    if not quiet:
        print("=" * 60)
        print("AI Code Detector")
        print("=" * 60)
        print()

    # Initialize detector
    detector = AICodeDetector(config_path=config)

    # Run analysis
    try:
        repo_score = detector.analyze_repo(source, verbose=not quiet)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    # Generate reports
    if not quiet:
        print()
        print("=" * 60)
        print("Generating reports...")
        print("=" * 60)
        print()

    # Determine output paths
    if output:
        output_dir = output
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path.cwd()

    json_reporter = JSONReporter()
    markdown_reporter = MarkdownReporter()

    # Generate requested formats
    if format in ['json', 'both']:
        json_path = output_dir / 'ai_detection_report.json'
        json_reporter.generate(repo_score, json_path)
        if not quiet:
            print(f"JSON report saved to: {json_path}")

    if format in ['markdown', 'both']:
        md_path = output_dir / 'ai_detection_report.md'
        markdown_reporter.generate(repo_score, md_path)
        if not quiet:
            print(f"Markdown report saved to: {md_path}")

    # Print summary to console
    if not quiet:
        print()
        print("=" * 60)
        print("Summary")
        print("=" * 60)
        print()

    # Always print summary
    prob_percent = repo_score.ai_probability * 100
    conf_percent = repo_score.confidence * 100

    click.echo(f"Repository: {repo_score.repo_path}")
    click.echo(f"AI Probability: {prob_percent:.1f}%")
    click.echo(f"Confidence: {conf_percent:.1f}%")
    click.echo()

    # Color-coded verdict
    if prob_percent >= 80:
        click.secho("Verdict: Very likely AI-generated", fg='red', bold=True)
    elif prob_percent >= 60:
        click.secho("Verdict: Likely AI-generated", fg='yellow', bold=True)
    elif prob_percent >= 40:
        click.secho("Verdict: Possibly AI-assisted", fg='yellow')
    else:
        click.secho("Verdict: Likely human-written", fg='green', bold=True)

    click.echo()

    # Show top suspicious files
    if repo_score.top_suspicious_files:
        click.echo("Top suspicious files:")
        for i, file_path in enumerate(repo_score.top_suspicious_files[:5], 1):
            file_score = next((fs for fs in repo_score.file_scores if fs.file_path == file_path), None)
            if file_score:
                click.echo(f"  {i}. {file_path} ({file_score.ai_probability*100:.1f}%)")

    click.echo()
    click.echo("Note: This is a probabilistic analysis. Use as one signal among many.")

    # Exit code based on threshold
    if prob_percent >= 70:
        sys.exit(2)  # High AI probability
    elif prob_percent >= 50:
        sys.exit(1)  # Moderate AI probability
    else:
        sys.exit(0)  # Low AI probability


if __name__ == '__main__':
    main()
