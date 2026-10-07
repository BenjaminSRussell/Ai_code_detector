"""Command-line interface for the agent-prep scanner."""

import sys
from pathlib import Path
import click

from .agent_scan import AgentPrepScanner
from .report.reporter_findings import FindingsMarkdownWriter, FindingsJSONWriter
from .report.reporter_sarif import FindingsSarifWriter


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
    help='Output directory for findings (defaults to the scanned repo root)'
)
@click.option(
    '--profile',
    is_flag=True,
    help='Also run an opt-in dynamic profiling pass (executes code from the scanned repo)'
)
@click.option(
    '--sarif',
    is_flag=True,
    help='Also write ai_scan_findings.sarif (SARIF 2.1.0 for GitHub code scanning)'
)
@click.option(
    '--suppressions',
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help='Suppressions file (default: <repo>/.aicdignore when present)'
)
@click.option(
    '--no-suppressions',
    is_flag=True,
    help='Ignore .aicdignore and scan every file'
)
@click.option(
    '--quiet', '-q',
    is_flag=True,
    help='Suppress progress output'
)
def main(source: str, config: Path, output: Path, profile: bool, sarif: bool, suppressions: Path, no_suppressions: bool, quiet: bool):
    """Scan a repository and produce agent-readable findings.

    SOURCE can be a GitHub URL or a local path.
    """
    scanner = AgentPrepScanner(config_path=config, enable_profiling=profile)
    scanner.file_filter.suppressions_file = suppressions
    scanner.file_filter.use_suppressions = not no_suppressions

    try:
        findings = scanner.scan(source, verbose=not quiet)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    output_dir = output if output else Path(findings.repo_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    md_path = output_dir / 'AI_SCAN_FINDINGS.md'
    json_path = output_dir / 'ai_scan_findings.json'

    FindingsMarkdownWriter().generate(findings, md_path)
    FindingsJSONWriter().generate(findings, json_path)
    sarif_path = output_dir / 'ai_scan_findings.sarif'
    if sarif:
        FindingsSarifWriter().generate(findings, sarif_path)

    if not quiet:
        click.echo("\nFindings written to:")
        click.echo(f"  {md_path}")
        click.echo(f"  {json_path}")
        if sarif:
            click.echo(f"  {sarif_path}")
        click.echo(
            f"\n{len(findings.findings)} findings across "
            f"{len({f.type for f in findings.findings})} categories."
        )


if __name__ == '__main__':
    main()
