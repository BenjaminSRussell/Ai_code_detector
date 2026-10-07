"""``aicd gate``: score only the given files (e.g. a PR diff) and fail when any
file's AI probability exceeds a threshold (#12).

Used by ``.github/workflows/ai-scan.yml`` (changed files on a pull request) and
by the ``aicd-gate`` pre-commit hook (staged files). Honors ``.aicdignore``.

Exit codes: 0 = nothing above threshold (or nothing to scan), 1 = at least one
file flagged, 2 = usage / runtime error.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import click

DEFAULT_THRESHOLD = 0.7
GATE_EXTENSIONS = {".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs"}


@dataclass
class GateResult:
    threshold: float
    mode: str
    files: List[dict] = field(default_factory=list)       # scored: {path, ai_probability, flagged}
    suppressed: List[dict] = field(default_factory=list)  # {path, pattern}
    skipped: List[dict] = field(default_factory=list)     # {path, reason}

    @property
    def flagged(self) -> List[dict]:
        return [f for f in self.files if f["flagged"]]

    def to_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "mode": self.mode,
            "passed": not self.flagged,
            "flagged": [f["path"] for f in self.flagged],
            "files": self.files,
            "suppressed": self.suppressed,
            "skipped": self.skipped,
        }

    def markdown(self) -> str:
        lines = ["## AI code gate", ""]
        verdict = (f"**{len(self.flagged)} file(s) above {self.threshold:.2f}**" if self.flagged
                   else f"No changed file above {self.threshold:.2f}")
        lines.append(f"{verdict} (mode: `{self.mode}`, scanned {len(self.files)}, "
                     f"suppressed {len(self.suppressed)}, skipped {len(self.skipped)}).")
        lines.append("")
        if self.files:
            lines += ["| | File | AI probability |", "|---|---|---|"]
            for f in sorted(self.files, key=lambda x: -x["ai_probability"]):
                mark = "🚩" if f["flagged"] else "✅"
                lines.append(f"| {mark} | `{f['path']}` | {f['ai_probability'] * 100:.1f}% |")
            lines.append("")
        if self.skipped:
            lines.append("<details><summary>Skipped</summary>\n")
            lines += [f"- `{s['path']}`: {s['reason']}" for s in self.skipped]
            lines.append("\n</details>\n")
        if self.suppressed:
            lines.append("<details><summary>Suppressed by .aicdignore</summary>\n")
            lines += [f"- `{s['path']}` ({s['pattern']})" for s in self.suppressed]
            lines.append("\n</details>\n")
        return "\n".join(lines)


def run_gate(paths, root: Path, threshold: float = DEFAULT_THRESHOLD, mode: str = "basic",
             config: Optional[Path] = None, use_suppressions: bool = True,
             suppressions_file: Optional[Path] = None) -> GateResult:
    from .detector_enhanced import EnhancedAICodeDetector
    from .ingest.file_filter import FileFilter, FileInfo
    from .ingest.suppressions import Suppressions

    root = Path(root).resolve()
    result = GateResult(threshold=threshold, mode=mode)
    sup = Suppressions.load(root, suppressions_file) if use_suppressions else Suppressions()
    det = None

    for raw in dict.fromkeys(str(p) for p in paths):  # de-dupe, keep order
        p = Path(raw)
        abs_p = (p if p.is_absolute() else root / p).resolve()
        try:
            rel = abs_p.relative_to(root).as_posix()
        except ValueError:
            rel = raw
        if not abs_p.is_file():
            result.skipped.append({"path": rel, "reason": "missing (deleted or renamed)"})
            continue
        if abs_p.suffix not in GATE_EXTENSIONS:
            result.skipped.append({"path": rel, "reason": f"unsupported extension {abs_p.suffix or '(none)'}"})
            continue
        rule = sup.match(rel) if sup.rules else None
        if rule is not None:
            result.suppressed.append({"path": rel, "pattern": rule.pattern})
            continue
        if det is None:
            det = EnhancedAICodeDetector(config_path=config, use_ml=(mode == "enhanced"),
                                         use_explanations=False)
        text = abs_p.read_text(encoding="utf-8", errors="ignore")
        info = FileInfo(path=abs_p, relative_path=Path(rel),
                        language=FileFilter.LANGUAGE_MAP.get(abs_p.suffix, "unknown"),
                        size_bytes=abs_p.stat().st_size, line_count=text.count("\n") + 1)
        score = det._analyze_file_enhanced(info, root)
        prob = round(float(score.ai_probability), 4)
        result.files.append({"path": rel, "ai_probability": prob, "flagged": prob > threshold})
    return result


def _env_threshold() -> float:
    raw = os.environ.get("AICD_THRESHOLD", "").strip()
    try:
        return float(raw) if raw else DEFAULT_THRESHOLD
    except ValueError:
        raise click.BadParameter(f"AICD_THRESHOLD={raw!r} is not a number", param_hint="AICD_THRESHOLD")


@click.command("gate")
@click.argument("paths", nargs=-1, type=click.Path(path_type=Path))
@click.option("--threshold", "-t", type=click.FloatRange(0.0, 1.0), default=None,
              help=f"Fail when a file's AI probability is above this (default: $AICD_THRESHOLD or {DEFAULT_THRESHOLD})")
@click.option("--mode", type=click.Choice(["basic", "enhanced"]), default="basic", show_default=True,
              help="basic = heuristics only (stable, fast); enhanced adds the ML classifier")
@click.option("--root", type=click.Path(exists=True, file_okay=False, path_type=Path), default=Path("."),
              show_default=True, help="Repository root (paths and .aicdignore are relative to it)")
@click.option("--paths-from", type=click.File("r"), default=None,
              help="Read newline-separated paths from a file ('-' for stdin), e.g. `git diff --name-only`")
@click.option("--config", "-c", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--json", "json_out", type=click.Path(dir_okay=False, path_type=Path), help="Write a JSON report")
@click.option("--sarif", "sarif_out", type=click.Path(dir_okay=False, path_type=Path), help="Write SARIF 2.1.0")
@click.option("--summary", "summary_out", type=click.Path(dir_okay=False, path_type=Path),
              default=lambda: os.environ.get("GITHUB_STEP_SUMMARY") or None,
              help="Append a Markdown table (default: $GITHUB_STEP_SUMMARY when set)")
@click.option("--suppressions", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--no-suppressions", is_flag=True)
@click.option("--quiet", "-q", is_flag=True)
def gate(paths, threshold, mode, root, paths_from, config, json_out, sarif_out, summary_out,
         suppressions, no_suppressions, quiet):
    """Score PATHS (e.g. a PR's changed files) and exit 1 if any is above the threshold."""
    threshold = _env_threshold() if threshold is None else threshold
    all_paths = list(paths)
    if paths_from is not None:
        all_paths += [Path(l.strip()) for l in paths_from if l.strip()]
    try:
        result = run_gate(all_paths, root, threshold=threshold, mode=mode, config=config,
                          use_suppressions=not no_suppressions, suppressions_file=suppressions)
    except Exception as exc:  # pragma: no cover - surfaced to CI log
        click.echo(f"aicd gate: error: {exc}", err=True)
        raise SystemExit(2)

    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    if sarif_out:
        from .report.reporter_sarif import SarifReporter
        from .model.aggregator import FileScore, RepoScore
        scores = [FileScore(file_path=f["path"], ai_probability=f["ai_probability"], stylometry_score=0.0,
                            structural_score=0.0, feature_explanations={}, suspicious_snippets=[])
                  for f in result.files]
        repo = RepoScore(repo_path=str(Path(root).resolve()), ai_probability=max(
            [f["ai_probability"] for f in result.files], default=0.0), confidence=0.0,
            stylometry_score=0.0, structural_score=0.0, history_score=0.0, file_scores=scores,
            top_suspicious_files=[f["path"] for f in result.flagged], total_files_analyzed=len(scores),
            total_lines_analyzed=0, language_distribution={},
            suppressed={"count": len(result.suppressed)})
        SarifReporter().generate(repo, sarif_out)
    md = result.markdown()
    if summary_out:
        with open(summary_out, "a", encoding="utf-8") as fh:
            fh.write(md + "\n")
    if not quiet or result.flagged:  # --quiet (pre-commit) stays silent only when passing
        click.echo(md)
    raise SystemExit(1 if result.flagged else 0)
