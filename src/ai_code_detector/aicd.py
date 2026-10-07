"""Unified ``aicd`` command line (#9).

    aicd scan SOURCE [--mode basic|enhanced] ...   repository AI-likelihood report
    aicd agent-scan SOURCE ...                      agent-readable findings
    aicd explain FILE ...                           score + explanation for one file
    aicd train DATASET.jsonl -o model.json          fit the Phase 2 classifier
    aicd gate FILES... [--threshold 0.7]            fail if any file is above threshold (PR/pre-commit)
    aicd history [--repo PATH] [--scan ID]          scan runs stored with `aicd scan --store`
    aicd eval --manifest datasets/manifest.yaml     ROC AUC / calibration on a labeled set

``scan`` is the former ``ai-code-detector-enhanced`` command, with the same
options, outputs, and exit codes. ``agent-scan`` is the former
``ai-code-detector-agent-scan``. The old console scripts and modules still work
for one release but print a one-line deprecation notice.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from . import cli as _cli_basic
from . import cli_agent_scan as _cli_agent_scan
from . import cli_enhanced as _cli_enhanced

SUBCOMMANDS = ("scan", "agent-scan", "explain", "train", "gate", "eval", "history")


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(package_name="ai-code-detector", message="aicd %(version)s")
def main():
    """AI code detector: one entrypoint, several subcommands."""


main.add_command(_cli_enhanced.main, name="scan")
main.add_command(_cli_agent_scan.main, name="agent-scan")

from .gate import gate as _gate  # noqa: E402
from .store import history as _history  # noqa: E402
from .evaluation import main as _eval  # noqa: E402

main.add_command(_gate, name="gate")
main.add_command(_history, name="history")
main.add_command(_eval, name="eval")


def _file_info(path: Path):
    from .ingest.file_filter import FileFilter, FileInfo
    text = path.read_text(encoding="utf-8", errors="ignore")
    return FileInfo(
        path=path,
        relative_path=Path(path.name),
        language=FileFilter.LANGUAGE_MAP.get(path.suffix, "unknown"),
        size_bytes=path.stat().st_size,
        line_count=text.count("\n") + 1,
    )


@main.command()
@click.argument("file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--config", "-c", type=click.Path(exists=True, path_type=Path), help="Path to config YAML file")
@click.option("--explainer", type=click.Choice(["template", "qwen"], case_sensitive=False), default="template")
@click.option("--no-ml", is_flag=True, help="Heuristics only (skip the Phase 2 classifier)")
@click.option("--model", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Classifier weights from `aicd train`")
@click.option("--json", "as_json", is_flag=True, help="Print a JSON object instead of text")
def explain(file: Path, config, explainer: str, no_ml: bool, model, as_json: bool):
    """Score a single FILE and explain the top contributing features."""
    from .detector_enhanced import EnhancedAICodeDetector

    det = EnhancedAICodeDetector(config_path=config, use_ml=not no_ml, use_explanations=True,
                                 explainer_backend=explainer, ml_model_path=model)
    score = det._analyze_file_enhanced(_file_info(file.resolve()), file.resolve().parent)
    explanation = score.explanation
    if explanation is None and det.explainer is not None:
        # The detector only explains files above 0.5; `explain` always answers.
        code = file.read_text(encoding="utf-8", errors="ignore")
        explanation = det.explainer.explain(code=code, ai_probability=score.ai_probability,
                                            features=score.feature_explanations, top_n=3)
    result = {
        "file": str(file),
        "ai_probability": round(float(score.ai_probability), 4),
        "confidence": round(float(getattr(score, "confidence", 0.0)), 4),
        "explanation": explanation,
    }
    if as_json:
        click.echo(json.dumps(result, indent=2))
    else:
        click.echo(f"{file}: AI probability {result['ai_probability'] * 100:.1f}%")
        if explanation:
            click.echo(explanation)


@main.command()
@click.argument("dataset", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--output", "-o", type=click.Path(dir_okay=False, path_type=Path), required=True,
              help="Where to write classifier weights (JSON)")
@click.option("--epochs", type=int, default=100, show_default=True)
@click.option("--learning-rate", type=float, default=0.01, show_default=True)
@click.option("--config", "-c", type=click.Path(exists=True, path_type=Path))
def train(dataset: Path, output: Path, epochs: int, learning_rate: float, config):
    """Train the Phase 2 classifier from a labeled JSONL DATASET.

    Each line is ``{"path": "file.py", "label": 0|1}`` (1 = AI). Relative paths
    are resolved against the dataset's directory. Use the weights with
    ``aicd explain --model``.
    """
    from .detector_enhanced import EnhancedAICodeDetector
    from .evaluation import featurize
    from .model.classifier import MLClassifier

    det = EnhancedAICodeDetector(config_path=config, use_ml=True, use_explanations=False)
    embeddings, features, labels = [], [], []
    for lineno, line in enumerate(dataset.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            label = int(row["label"])
            if label not in (0, 1):
                raise ValueError("label must be 0 or 1")
            path = Path(row["path"])
        except (ValueError, KeyError, TypeError) as exc:
            raise click.BadParameter(f"line {lineno}: {exc}", param_hint="DATASET")
        if not path.is_absolute():
            path = dataset.parent / path
        if not path.is_file():
            raise click.BadParameter(f"line {lineno}: no such file {path}", param_hint="DATASET")
        emb, feat = featurize(det, path)
        embeddings.append(emb)
        features.append(feat)
        labels.append(label)
    if len(set(labels)) < 2:
        raise click.UsageError("dataset needs at least one example of each label (0 and 1)")

    clf = MLClassifier(embedding_dim=len(embeddings[0]), feature_dim=len(features[0]))
    history = clf.train(embeddings, features, labels, epochs=epochs, learning_rate=learning_rate)
    clf.save(output)
    click.echo(f"trained on {len(labels)} files: final loss {history['loss'][-1]:.4f}, "
               f"accuracy {history['accuracy'][-1]:.3f} -> {output}")


# ---- deprecated entrypoints (one release) ---------------------------------

def _deprecated(old: str, new: str, command):
    def runner():
        click.echo(f"warning: `{old}` is deprecated; use `{new}`", err=True)
        command.main(prog_name=old)
    return runner


legacy_basic = _deprecated("ai-code-detector", "aicd scan --mode basic", _cli_basic.main)
legacy_enhanced = _deprecated("ai-code-detector-enhanced", "aicd scan", _cli_enhanced.main)
legacy_agent_scan = _deprecated("ai-code-detector-agent-scan", "aicd agent-scan", _cli_agent_scan.main)


def module_main(argv=None):
    """``python -m ai_code_detector``: aicd, plus the old ``<repo>`` form."""
    args = list(sys.argv[1:] if argv is None else argv)
    if args and not args[0].startswith("-") and args[0] not in SUBCOMMANDS:
        click.echo("warning: `python -m ai_code_detector <repo>` is deprecated; "
                   "use `python -m ai_code_detector scan <repo> --mode basic`", err=True)
        return _cli_basic.main.main(args=args, prog_name="python -m ai_code_detector")
    return main.main(args=args, prog_name="aicd")
