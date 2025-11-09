"""Enhanced CLI with ML classifier and explanations."""

import sys
from pathlib import Path
import click

from .detector_enhanced import EnhancedAICodeDetector
from .report.reporter_enhanced import EnhancedJSONReporter, EnhancedMarkdownReporter


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
    '--mode',
    type=click.Choice(['basic', 'enhanced'], case_sensitive=False),
    default='enhanced',
    help='Detection mode: basic (Phase 1 only) or enhanced (Phase 1+2+3)'
)
@click.option(
    '--no-ml',
    is_flag=True,
    help='Disable ML classifier (Phase 2)'
)
@click.option(
    '--no-explanations',
    is_flag=True,
    help='Disable natural language explanations (Phase 3)'
)
@click.option(
    '--embedder',
    type=click.Choice(['hash', 'mlx'], case_sensitive=False),
    default='hash',
    help='Embedding backend for Phase 2'
)
@click.option(
    '--explainer',
    type=click.Choice(['template', 'qwen'], case_sensitive=False),
    default='template',
    help='Explanation backend for Phase 3'
)
@click.option(
    '--quiet', '-q',
    is_flag=True,
    help='Suppress progress output'
)
def main(
    source: str,
    config: Path,
    output: Path,
    format: str,
    mode: str,
    no_ml: bool,
    no_explanations: bool,
    embedder: str,
    explainer: str,
    quiet: bool
):
    """Analyze a GitHub repository or local directory for AI-generated code.

    Enhanced version with ML classifier and natural language explanations.

    SOURCE can be:
      - GitHub URL (e.g., https://github.com/user/repo)
      - Local path (e.g., /path/to/repo)

    Examples:
      # Basic analysis (Phase 1 only)
      ai-code-detector-enhanced ./repo --mode basic

      # Enhanced analysis (Phase 1+2+3)
      ai-code-detector-enhanced ./repo --mode enhanced

      # With MLX embeddings
      ai-code-detector-enhanced ./repo --embedder mlx

      # With Qwen explanations
      ai-code-detector-enhanced ./repo --explainer qwen
    """
    if not quiet:
        print("=" * 60)
        print("AI Code Detector (Enhanced)")
        print("=" * 60)
        print()

        if mode == 'enhanced':
            print("Mode: Enhanced (Phase 1 + Phase 2 + Phase 3)")
            if not no_ml:
                print("  ML classifier enabled")
                print(f"  Embedder: {embedder}")
            if not no_explanations:
                print("  Explanations enabled")
                print(f"  Explainer: {explainer}")
        else:
            print("Mode: Basic (Phase 1 only)")

        print()

    # Initialize enhanced detector
    use_ml = (mode == 'enhanced') and not no_ml
    use_explanations = (mode == 'enhanced') and not no_explanations

    detector = EnhancedAICodeDetector(
        config_path=config,
        use_ml=use_ml,
        use_explanations=use_explanations,
        embedder_backend=embedder,
        explainer_backend=explainer,
    )

    # Run analysis
    try:
        repo_score = detector.analyze_repo(source, verbose=not quiet)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        import traceback
        if not quiet:
            traceback.print_exc()
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

    json_reporter = EnhancedJSONReporter()
    markdown_reporter = EnhancedMarkdownReporter()

    # Generate requested formats
    if format in ['json', 'both']:
        json_path = output_dir / 'ai_detection_report_enhanced.json'
        json_reporter.generate(repo_score, json_path)
        if not quiet:
            print(f"JSON report saved to: {json_path}")

    if format in ['markdown', 'both']:
        md_path = output_dir / 'ai_detection_report_enhanced.md'
        markdown_reporter.generate(repo_score, md_path)
        if not quiet:
            print(f"Markdown report saved to: {md_path}")

    # Print summary
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

    # Show top suspicious files with explanations
    if repo_score.top_suspicious_files:
        click.echo("Top suspicious files:")
        for i, file_path in enumerate(repo_score.top_suspicious_files[:5], 1):
            file_score = next((fs for fs in repo_score.file_scores if fs.file_path == file_path), None)
            if file_score:
                click.echo(f"  {i}. {file_path} ({file_score.ai_probability*100:.1f}%)")

                # Show explanation if available (Phase 3)
                if hasattr(file_score, 'explanation') and file_score.explanation:
                    click.echo(f"     └─ {file_score.explanation[:100]}...")

    click.echo()

    # Mode indicator
    if mode == 'enhanced':
        click.secho("Enhanced detection with ML + explanations", fg='cyan')
    else:
        click.secho("Basic detection (use --mode enhanced for ML + explanations)", fg='blue')

    click.echo()
    click.echo("Note: This is probabilistic analysis. Use as one signal among many.")

    # Exit code based on threshold
    if prob_percent >= 70:
        sys.exit(2)  # High AI probability
    elif prob_percent >= 50:
        sys.exit(1)  # Moderate AI probability
    else:
        sys.exit(0)  # Low AI probability


if __name__ == '__main__':
    main()
