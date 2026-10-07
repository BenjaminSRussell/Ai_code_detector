"""Labeled evaluation harness: ROC / calibration / confusion matrix (#11).

    python scripts/eval_harness.py --manifest datasets/manifest.yaml
    aicd eval --manifest datasets/manifest.yaml --train

It writes these files to ``--out`` (default ``reports/eval/``):
``metrics.json``, ``predictions.csv``, ``roc_<scorer>.csv``,
``calibration_<scorer>.csv`` and ``summary.md``.

Scorers:
  basic    the heuristic aggregator (what ``aicd scan --mode basic`` and
           ``aicd gate`` use), evaluated on every entry (it is not trained).
  trained  only with ``--train``. It fits the Phase 2 classifier on the
           ``train`` split with the same features ``aicd train`` uses, then
           scores the held-out ``test`` split in enhanced mode
           (0.4 heuristic + 0.6 ML).
  model    only with ``--model``. A previously trained classifier, also
           scored on the ``test`` split.

The metrics are written in plain numpy so their definitions stay auditable.
They describe the detector's behaviour on this dataset only. Probabilities
are not proof of authorship.
"""
from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import click
import yaml

LABELS = {"human": 0, "ai": 1, 0: 0, 1: 1, "0": 0, "1": 1}
CAVEATS = [
    "Scores are probabilistic signals, not proof of authorship.",
    "Metrics only describe this dataset; small or synthetic sets (like datasets/manifest.yaml) "
    "say nothing about real-world accuracy.",
    "Style-based detectors are easy to evade and can flag terse or heavily commented human code.",
]


class ManifestError(ValueError):
    pass


@dataclass
class Entry:
    path: Path
    rel: str
    label: int
    split: str
    source: str = ""


def _auto_split(rel: str, test_fraction: float) -> str:
    h = int(hashlib.sha1(rel.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if h < test_fraction else "train"


def load_manifest(path: Path) -> List[Entry]:
    path = Path(path)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ManifestError(f"cannot read manifest {path}: {exc}")
    if not isinstance(data, dict) or not isinstance(data.get("entries"), list):
        raise ManifestError("manifest needs a top-level `entries:` list")
    base = path.parent
    test_fraction = float(data.get("test_fraction", 0.3))
    out: Dict[str, Entry] = {}
    for i, raw in enumerate(data["entries"], 1):
        if not isinstance(raw, dict):
            raise ManifestError(f"entry {i}: expected a mapping")
        if raw.get("label") not in LABELS:
            raise ManifestError(f"entry {i}: label must be human|ai (got {raw.get('label')!r})")
        split = raw.get("split")
        if split not in (None, "train", "test"):
            raise ManifestError(f"entry {i}: split must be train|test (got {split!r})")
        if "path" in raw:
            files = [base / raw["path"]]
            if not files[0].is_file():
                raise ManifestError(f"entry {i}: no such file {raw['path']}")
        elif "glob" in raw:
            files = sorted(p for p in base.glob(raw["glob"]) if p.is_file())
            if not files:
                raise ManifestError(f"entry {i}: glob {raw['glob']!r} matched nothing")
        else:
            raise ManifestError(f"entry {i}: needs `path` or `glob`")
        for f in files:
            f = f.resolve()
            rel = Path(os.path.relpath(f, base.resolve())).as_posix()
            out[str(f)] = Entry(path=f, rel=rel, label=LABELS[raw["label"]],
                                split=split or _auto_split(rel, test_fraction),
                                source=str(raw.get("source", "")))
    if not out:
        raise ManifestError("manifest has no files")
    return list(out.values())


# ---- metrics -----------------------------------------------------------------

def roc_auc(y: Sequence[int], p: Sequence[float]) -> Optional[float]:
    """Mann-Whitney AUC: P(score_ai > score_human), with ties counted as 1/2."""
    pos = [s for s, t in zip(p, y) if t == 1]
    neg = [s for s, t in zip(p, y) if t == 0]
    if not pos or not neg:
        return None
    wins = sum(1.0 if a > b else 0.5 if a == b else 0.0 for a in pos for b in neg)
    return wins / (len(pos) * len(neg))


def roc_curve(y: Sequence[int], p: Sequence[float]) -> List[Dict[str, float]]:
    P = sum(y)
    N = len(y) - P
    points = [{"threshold": None, "fpr": 0.0, "tpr": 0.0}]  # (0,0): nothing flagged
    for t in sorted(set(p), reverse=True):
        tp = sum(1 for s, l in zip(p, y) if s >= t and l == 1)
        fp = sum(1 for s, l in zip(p, y) if s >= t and l == 0)
        points.append({"threshold": float(t), "fpr": fp / N if N else 0.0, "tpr": tp / P if P else 0.0})
    return points


def confusion(y: Sequence[int], p: Sequence[float], threshold: float) -> Dict[str, float]:
    tp = sum(1 for s, l in zip(p, y) if s >= threshold and l == 1)
    fp = sum(1 for s, l in zip(p, y) if s >= threshold and l == 0)
    fn = sum(1 for s, l in zip(p, y) if s < threshold and l == 1)
    tn = sum(1 for s, l in zip(p, y) if s < threshold and l == 0)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"threshold": threshold, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "accuracy": (tp + tn) / len(y) if y else 0.0, "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}


def calibration(y: Sequence[int], p: Sequence[float], bins: int = 10) -> Dict[str, object]:
    """Reliability table over equal-width bins, plus ECE and the Brier score."""
    rows = []
    n = len(y)
    ece = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, s in enumerate(p) if (lo <= s < hi) or (b == bins - 1 and s == 1.0)]
        if not idx:
            rows.append({"bin_lo": lo, "bin_hi": hi, "count": 0, "mean_predicted": None, "fraction_ai": None})
            continue
        mp = sum(p[i] for i in idx) / len(idx)
        fa = sum(y[i] for i in idx) / len(idx)
        ece += len(idx) / n * abs(mp - fa)
        rows.append({"bin_lo": lo, "bin_hi": hi, "count": len(idx), "mean_predicted": mp, "fraction_ai": fa})
    brier = sum((s - l) ** 2 for s, l in zip(p, y)) / n if n else None
    return {"bins": rows, "ece": ece, "brier": brier}


def metrics(y: Sequence[int], p: Sequence[float], threshold: float = 0.5, bins: int = 10) -> Dict[str, object]:
    cal = calibration(y, p, bins)
    return {"n": len(y), "n_ai": int(sum(y)), "n_human": len(y) - int(sum(y)),
            "roc_auc": roc_auc(y, p), "brier": cal["brier"], "ece": cal["ece"],
            "confusion": confusion(y, p, threshold), "calibration": cal["bins"], "roc": roc_curve(y, p)}


# ---- scoring -----------------------------------------------------------------

def _file_info(path: Path):
    from .ingest.file_filter import FileFilter, FileInfo
    text = path.read_text(encoding="utf-8", errors="ignore")
    return FileInfo(path=path, relative_path=Path(path.name),
                    language=FileFilter.LANGUAGE_MAP.get(path.suffix, "unknown"),
                    size_bytes=path.stat().st_size, line_count=text.count("\n") + 1)


def featurize(det, path: Path):
    """(embedding, feature_vector) exactly as ``aicd train`` / enhanced scoring compute them."""
    from .analysis.ast_parser import ASTParserFactory
    info = _file_info(path)
    code = path.read_text(encoding="utf-8", errors="ignore")
    tree = None
    parser = ASTParserFactory.get_parser(info.language)
    if parser:
        try:
            tree = parser.parse_file(path, code)
        except Exception:
            tree = None
    sty = det.stylometry_analyzer.analyze_file(code, info.language, tree)
    struct = det.structural_analyzer.analyze_file(code, info.language, tree)
    return list(det.embedder.embed(code)), det._features_to_vector(sty, struct)


def score_files(paths: Sequence[Path], use_ml: bool, model_path: Optional[Path] = None,
                config: Optional[Path] = None) -> List[float]:
    from .detector_enhanced import EnhancedAICodeDetector
    with contextlib.redirect_stdout(io.StringIO()):
        det = EnhancedAICodeDetector(config_path=config, use_ml=use_ml, use_explanations=False,
                                     ml_model_path=model_path)
        return [float(det._analyze_file_enhanced(_file_info(p), p.parent).ai_probability) for p in paths]


def train_model(entries: Sequence[Entry], out: Path, epochs: int = 200, learning_rate: float = 0.05,
                config: Optional[Path] = None) -> Dict[str, float]:
    from .detector_enhanced import EnhancedAICodeDetector
    from .model.classifier import MLClassifier
    if len({e.label for e in entries}) < 2:
        raise ManifestError("train split needs at least one human and one ai file")
    with contextlib.redirect_stdout(io.StringIO()):
        det = EnhancedAICodeDetector(config_path=config, use_ml=True, use_explanations=False)
        X = [featurize(det, e.path) for e in entries]
        clf = MLClassifier(embedding_dim=len(X[0][0]), feature_dim=len(X[0][1]))
        hist = clf.train([x[0] for x in X], [x[1] for x in X], [e.label for e in entries],
                         epochs=epochs, learning_rate=learning_rate)
        clf.save(out)
    return {"n_train": len(entries), "epochs": epochs, "learning_rate": learning_rate,
            "final_loss": hist["loss"][-1], "train_accuracy": hist["accuracy"][-1]}


# ---- harness -----------------------------------------------------------------

def run(manifest: Path, out: Path, threshold: float = 0.5, bins: int = 10, train: bool = False,
        model: Optional[Path] = None, epochs: int = 200, config: Optional[Path] = None) -> Dict[str, object]:
    entries = load_manifest(manifest)
    out.mkdir(parents=True, exist_ok=True)
    y_all = [e.label for e in entries]
    preds: Dict[str, Dict[str, float]] = {e.rel: {} for e in entries}
    result: Dict[str, object] = {
        "manifest": str(manifest), "threshold": threshold, "bins": bins,
        "dataset": {"n": len(entries), "n_ai": sum(y_all), "n_human": len(y_all) - sum(y_all),
                    "train": sum(e.split == "train" for e in entries),
                    "test": sum(e.split == "test" for e in entries)},
        "scorers": {}, "caveats": CAVEATS,
    }

    basic = score_files([e.path for e in entries], use_ml=False, config=config)
    result["scorers"]["basic"] = {"evaluated_on": "all", **metrics(y_all, basic, threshold, bins)}
    for e, s in zip(entries, basic):
        preds[e.rel]["basic"] = s

    test = [e for e in entries if e.split == "test"]
    ml_runs = []
    if train:
        model_out = out / "model.json"
        info = train_model([e for e in entries if e.split == "train"], model_out, epochs=epochs, config=config)
        ml_runs.append(("trained", model_out, info))
    if model:
        ml_runs.append(("model", Path(model), {"model": str(model)}))
    for name, mpath, info in ml_runs:
        if not test:
            raise ManifestError(f"{name}: manifest has no `test` split to evaluate on")
        scores = score_files([e.path for e in test], use_ml=True, model_path=mpath, config=config)
        result["scorers"][name] = {"evaluated_on": "test", "model_path": str(mpath), **info,
                                   **metrics([e.label for e in test], scores, threshold, bins)}
        for e, s in zip(test, scores):
            preds[e.rel][name] = s

    _write_outputs(out, entries, preds, result)
    return result


def _fmt(v, pct=False):
    if v is None:
        return "n/a"
    return f"{v * 100:.1f}%" if pct else f"{v:.3f}"


def _write_outputs(out: Path, entries, preds, result):
    (out / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    scorers = list(result["scorers"])
    with open(out / "predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "split", "source", *scorers])
        for e in entries:
            w.writerow([e.rel, "ai" if e.label else "human", e.split, e.source,
                        *[f"{preds[e.rel][s]:.4f}" if s in preds[e.rel] else "" for s in scorers]])
    for name, m in result["scorers"].items():
        with open(out / f"roc_{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["threshold", "fpr", "tpr"])
            w.writeheader()
            w.writerows(m["roc"])
        with open(out / f"calibration_{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["bin_lo", "bin_hi", "count", "mean_predicted", "fraction_ai"])
            w.writeheader()
            w.writerows(m["calibration"])
    d = result["dataset"]
    lines = [f"# aicd evaluation: `{result['manifest']}`", "",
             f"{d['n']} files ({d['n_ai']} ai / {d['n_human']} human; {d['train']} train / {d['test']} test). "
             f"Confusion matrix at threshold {result['threshold']}.", "",
             "| scorer | evaluated on | n | ROC AUC | Brier | ECE | accuracy | precision | recall | TP | FP | TN | FN |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, m in result["scorers"].items():
        c = m["confusion"]
        lines.append(f"| {name} | {m['evaluated_on']} | {m['n']} | {_fmt(m['roc_auc'])} | {_fmt(m['brier'])} | "
                     f"{_fmt(m['ece'])} | {_fmt(c['accuracy'], True)} | {_fmt(c['precision'], True)} | "
                     f"{_fmt(c['recall'], True)} | {c['tp']} | {c['fp']} | {c['tn']} | {c['fn']} |")
    lines += ["", "**Caveats**", ""] + [f"- {c}" for c in result["caveats"]] + [""]
    (out / "summary.md").write_text("\n".join(lines))


@click.command("eval")
@click.option("--manifest", "-m", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              default=Path("datasets/manifest.yaml"), show_default=True)
@click.option("--out", "-o", type=click.Path(file_okay=False, path_type=Path), default=Path("reports/eval"),
              show_default=True)
@click.option("--threshold", type=click.FloatRange(0, 1), default=0.5, show_default=True,
              help="Decision threshold for the confusion matrix")
@click.option("--bins", type=click.IntRange(2, 50), default=10, show_default=True, help="Calibration bins")
@click.option("--train", is_flag=True, help="Fit the classifier on the train split; score the test split")
@click.option("--epochs", type=int, default=200, show_default=True)
@click.option("--model", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Also evaluate an existing `aicd train` model on the test split")
@click.option("--min-auc", type=click.FloatRange(0, 1), default=None,
              help="Exit 1 if any scorer's ROC AUC is below this (regression guard)")
@click.option("--config", "-c", type=click.Path(exists=True, path_type=Path))
@click.option("--quiet", "-q", is_flag=True)
def main(manifest, out, threshold, bins, train, epochs, model, min_auc, config, quiet):
    """Evaluate the detector on a labeled manifest (ROC AUC, calibration, confusion)."""
    try:
        result = run(manifest, out, threshold=threshold, bins=bins, train=train, model=model,
                     epochs=epochs, config=config)
    except ManifestError as exc:
        raise click.ClickException(str(exc))
    if not quiet:
        click.echo((out / "summary.md").read_text())
        click.echo(f"wrote {out}/metrics.json")
    if min_auc is not None:
        low = {k: v["roc_auc"] for k, v in result["scorers"].items()
               if v["roc_auc"] is not None and v["roc_auc"] < min_auc}
        if low:
            click.echo(f"ROC AUC below {min_auc}: {low}", err=True)
            raise SystemExit(1)


if __name__ == "__main__":
    main()
