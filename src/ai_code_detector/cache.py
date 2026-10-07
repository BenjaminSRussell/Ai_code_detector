"""Incremental scan cache (#8).

``aicd scan PATH --incremental`` hashes each eligible file. If the same
content was already analyzed under the same *analysis key*, the stored
FileScore is reused and nothing is recomputed. The key covers:

* the detector version
* the config fingerprint (weights, thresholds, feature settings)
* the phases (ML / explanations) and the embedder and explainer backends
* the classifier weights file, by its content hash

Changing any of these invalidates every cached entry. Results live in the
``file_cache`` table of the scan store (``--store`` path, or
``~/.cache/ai_code_detector/scans.db``). Because the key is the file's
content rather than its path, renames and copies are cache hits too.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

from .model.aggregator import FileScore

# FileScore attributes set outside the dataclass fields (explanations, #6 provenance).
_EXTRA_ATTRS = ("explanation", "language", "features", "ml_used")


def content_sha256(code: str) -> str:
    """Hash of the decoded text the analyzers see (same as FileScore.content_sha256)."""
    return hashlib.sha256(code.encode("utf-8", "surrogatepass")).hexdigest()


def _file_sha(path: Optional[Path]) -> Optional[str]:
    if not path:
        return None
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def analysis_key(*, detector_version: str, config_fingerprint: str, use_ml: bool, use_explanations: bool,
                 embedder: str, explainer: str, ml_model_path: Optional[Path] = None,
                 feature_dim: int = 0) -> str:
    material = {
        "detector_version": detector_version, "config": config_fingerprint, "use_ml": bool(use_ml),
        "use_explanations": bool(use_explanations), "embedder": embedder if use_ml else None,
        "explainer": explainer if use_explanations else None,
        "model": _file_sha(ml_model_path) if use_ml else None, "feature_dim": feature_dim,
    }
    return hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()[:24]


def score_to_dict(fs: FileScore) -> Dict[str, Any]:
    d = dataclasses.asdict(fs)
    d.pop("file_path", None)
    for attr in _EXTRA_ATTRS:
        if hasattr(fs, attr):
            d[attr] = getattr(fs, attr)
    return d


def score_from_dict(d: Dict[str, Any], file_path: str, sha: str) -> FileScore:
    names = {f.name for f in dataclasses.fields(FileScore)}
    fs = FileScore(file_path=file_path, **{k: v for k, v in d.items() if k in names and k != "file_path"})
    for attr in _EXTRA_ATTRS:
        if attr in d:
            setattr(fs, attr, d[attr])
    fs.content_sha256 = sha
    return fs


class FileCache:
    """Thin adapter over ScanStore.cache_get / cache_put, with hit/miss counters."""

    def __init__(self, store, key: str, force_full: bool = False):
        self.store = store
        self.key = key
        self.force_full = force_full
        self.recomputed: list = []
        self.cached = 0

    def get(self, sha: str) -> Optional[Dict[str, Any]]:
        if self.force_full:
            return None
        return self.store.cache_get(sha, self.key)

    def put(self, sha: str, fs: FileScore) -> None:
        self.store.cache_put(sha, self.key, score_to_dict(fs))

    def stats(self) -> Dict[str, Any]:
        return {"enabled": True, "force_full": self.force_full, "analysis_key": self.key,
                "recomputed": len(self.recomputed), "cached": self.cached,
                "recomputed_paths": self.recomputed[:200]}
