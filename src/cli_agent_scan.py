"""Command-line interface for the agent-prep scanner."""

from pathlib import Path
import click

from agent_scan import AgentPrepScanner
from report.reporter_findings import FindingsMarkdownWriter, FindingsJSONWriter


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
    '--quiet', '-q',
    is_flag=True,
    help='Suppress progress output'
)
def main(source: str, config: Path, output: Path, profile: bool, quiet: bool):
    """Scan a repository and produce agent-readable findings.

    SOURCE can be a GitHub URL or a local path.
    """
    scanner = AgentPrepScanner(config_path=config, enable_profiling=profile)
    findings = scanner.scan(source, verbose=not quiet)

    output_dir = output if output else Path(findings.repo_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    md_path = output_dir / 'AI_SCAN_FINDINGS.md'
    json_path = output_dir / 'ai_scan_findings.json'

    FindingsMarkdownWriter().generate(findings, md_path)
    FindingsJSONWriter().generate(findings, json_path)

    if not quiet:
        click.echo("\nFindings written to:")
        click.echo(f"  {md_path}")
        click.echo(f"  {json_path}")
        click.echo(
            f"\n{len(findings.findings)} findings across "
            f"{len({f.type for f in findings.findings})} categories."
        )


if __name__ == '__main__':
    main()
