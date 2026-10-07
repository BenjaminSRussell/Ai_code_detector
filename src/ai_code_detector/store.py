"""SQLite history of scan runs (#6).

    aicd scan ./repo --store ./scans.db       # or bare --store -> ~/.cache/ai_code_detector/scans.db
    aicd history --store ./scans.db [--repo PATH] [--scan ID] [--json]

Every run appends one ``scans`` row. Existing rows are never updated or
deleted, so earlier runs stay comparable. The schema is created with
``CREATE TABLE IF NOT EXISTS``, and ``PRAGMA user_version`` tracks
migrations (see ``SCHEMA_VERSION`` / ``_MIGRATIONS``).

Tables:
  scans     one row per run: repo, source, mode/phase flags, detector
            version, config fingerprint, repo-level scores
  files     one row per analyzed path in a scan: probability, component
            scores, phase flags (heuristics / ML / explanation), content
            sha256
  features  raw stylometry + structural feature values, additive feature
            contributions and indicators, as JSON (one row per file)
  verdicts  the repo verdict, plus a verdict for each file at or above the
            file threshold, with the natural-language explanation when
            there is one
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_STORE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "ai_code_detector" / "scans.db"
SCHEMA_VERSION = 1

_SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS scans (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    repo_path          TEXT NOT NULL,
    source             TEXT NOT NULL,
    started_at         TEXT NOT NULL,
    finished_at        TEXT NOT NULL,
    mode               TEXT NOT NULL,
    use_ml             INTEGER NOT NULL,
    use_explanations   INTEGER NOT NULL,
    detector_version   TEXT NOT NULL,
    config_fingerprint TEXT NOT NULL,
    ai_probability     REAL NOT NULL,
    confidence         REAL NOT NULL,
    stylometry_score   REAL NOT NULL,
    structural_score   REAL NOT NULL,
    history_score      REAL NOT NULL,
    total_files        INTEGER NOT NULL,
    total_lines        INTEGER NOT NULL,
    languages          TEXT NOT NULL,          -- JSON {language: count}
    suppressed_count   INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_scans_repo ON scans(repo_path, id);

CREATE TABLE IF NOT EXISTS files (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_id           INTEGER NOT NULL REFERENCES scans(id) ON DELETE CASCADE,
    path              TEXT NOT NULL,
    language          TEXT,
    ai_probability    REAL NOT NULL,
    stylometry_score  REAL NOT NULL,
    structural_score  REAL NOT NULL,
    parse_failed      INTEGER NOT NULL DEFAULT 0,
    phase_heuristic   INTEGER NOT NULL DEFAULT 1,
    phase_ml          INTEGER NOT NULL DEFAULT 0,
    phase_explanation INTEGER NOT NULL DEFAULT 0,
    content_sha256    TEXT,
    UNIQUE (scan_id, path)
);
CREATE INDEX IF NOT EXISTS idx_files_path ON files(path, scan_id);
CREATE INDEX IF NOT EXISTS idx_files_sha ON files(content_sha256);

CREATE TABLE IF NOT EXISTS features (
    file_id       INTEGER PRIMARY KEY REFERENCES files(id) ON DELETE CASCADE,
    features      TEXT NOT NULL,   -- JSON {"stylometry": {...}, "structural": {...}}
    contributions TEXT NOT NULL,   -- JSON {feature: share of ai_probability}
    indicators    TEXT NOT NULL    -- JSON FileScore.feature_explanations
);

CREATE TABLE IF NOT EXISTS verdicts (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_id        INTEGER NOT NULL REFERENCES scans(id) ON DELETE CASCADE,
    file_id        INTEGER REFERENCES files(id) ON DELETE CASCADE,  -- NULL = repo-level
    level          TEXT NOT NULL CHECK (level IN ('repo', 'file')),
    verdict        TEXT NOT NULL,
    ai_probability REAL NOT NULL,
    threshold      REAL,
    explanation    TEXT
);
CREATE INDEX IF NOT EXISTS idx_verdicts_scan ON verdicts(scan_id);
"""

# version -> SQL that upgrades (version - 1) -> version. Append; never edit.
_MIGRATIONS = {1: _SCHEMA_V1}


def verdict(p: float) -> str:
    if p >= 0.8:
        return "Very likely AI-generated"
    if p >= 0.6:
        return "Likely AI-generated"
    if p >= 0.4:
        return "Possibly AI-assisted"
    return "Likely human-written"


def config_fingerprint(config: Any) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _detector_version() -> str:
    try:
        from importlib.metadata import version
        return version("ai-code-detector")
    except Exception:
        return "unknown"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ScanStore:
    def __init__(self, path: Path = DEFAULT_STORE):
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.migrate()

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @property
    def schema_version(self) -> int:
        return self.conn.execute("PRAGMA user_version").fetchone()[0]

    def migrate(self) -> int:
        current = self.schema_version
        if current > SCHEMA_VERSION:
            raise RuntimeError(f"{self.path} has schema v{current}; this aicd only knows v{SCHEMA_VERSION}. Upgrade aicd.")
        for v in range(current + 1, SCHEMA_VERSION + 1):
            with self.conn:
                self.conn.executescript(_MIGRATIONS[v])
                self.conn.execute(f"PRAGMA user_version = {v}")
        return self.schema_version

    # ---- write ----------------------------------------------------------

    def record(self, repo_score, *, source: str, mode: str, use_ml: bool, use_explanations: bool,
               config: Any = None, file_threshold: float = 0.6, started_at: Optional[str] = None) -> int:
        """Append one scan (and its files, features, verdicts). Returns the new scan id."""
        sup = getattr(repo_score, "suppressed", None) or {}
        with self.conn:
            cur = self.conn.execute(
                """INSERT INTO scans (repo_path, source, started_at, finished_at, mode, use_ml, use_explanations,
                       detector_version, config_fingerprint, ai_probability, confidence, stylometry_score,
                       structural_score, history_score, total_files, total_lines, languages, suppressed_count)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (str(repo_score.repo_path), str(source), started_at or _now(), _now(), mode, int(use_ml),
                 int(use_explanations), _detector_version(), config_fingerprint(config),
                 float(repo_score.ai_probability), float(repo_score.confidence),
                 float(repo_score.stylometry_score), float(repo_score.structural_score),
                 float(repo_score.history_score), int(repo_score.total_files_analyzed),
                 int(repo_score.total_lines_analyzed), json.dumps(dict(repo_score.language_distribution or {})),
                 int(sup.get("count", 0) or 0)))
            scan_id = cur.lastrowid
            self.conn.execute(
                "INSERT INTO verdicts (scan_id, file_id, level, verdict, ai_probability, threshold) VALUES (?,?,?,?,?,?)",
                (scan_id, None, "repo", verdict(float(repo_score.ai_probability)), float(repo_score.ai_probability), None))
            for fs in repo_score.file_scores:
                explanation = getattr(fs, "explanation", None)
                prob = float(fs.ai_probability)
                fcur = self.conn.execute(
                    """INSERT INTO files (scan_id, path, language, ai_probability, stylometry_score, structural_score,
                           parse_failed, phase_heuristic, phase_ml, phase_explanation, content_sha256)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                    (scan_id, str(fs.file_path), getattr(fs, "language", None), prob, float(fs.stylometry_score),
                     float(fs.structural_score), int(bool(getattr(fs, "parse_failed", False))), 1,
                     int(bool(getattr(fs, "ml_used", False))), int(bool(explanation)),
                     getattr(fs, "content_sha256", None)))
                file_id = fcur.lastrowid
                indicators = fs.feature_explanations if isinstance(fs.feature_explanations, dict) else {}
                self.conn.execute(
                    "INSERT INTO features (file_id, features, contributions, indicators) VALUES (?,?,?,?)",
                    (file_id, json.dumps(getattr(fs, "features", None) or {}, default=str),
                     json.dumps(getattr(fs, "feature_contributions", None) or {}),
                     json.dumps(indicators, default=str)))
                if prob >= file_threshold:
                    self.conn.execute(
                        """INSERT INTO verdicts (scan_id, file_id, level, verdict, ai_probability, threshold, explanation)
                           VALUES (?,?,?,?,?,?,?)""",
                        (scan_id, file_id, "file", verdict(prob), prob, file_threshold, explanation))
        return scan_id

    # ---- read -----------------------------------------------------------

    def list_scans(self, repo: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        q = "SELECT * FROM scans"
        args: list = []
        if repo is not None:
            q += " WHERE repo_path = ? OR source = ?"
            args += [repo, repo]
        q += " ORDER BY id DESC LIMIT ?"
        args.append(limit)
        return [self._scan_row(r) for r in self.conn.execute(q, args)]

    def get_scan(self, scan_id: int) -> Optional[Dict[str, Any]]:
        r = self.conn.execute("SELECT * FROM scans WHERE id = ?", (scan_id,)).fetchone()
        return self._scan_row(r) if r else None

    def files(self, scan_id: int) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            """SELECT f.*, x.features, x.contributions, x.indicators FROM files f
               LEFT JOIN features x ON x.file_id = f.id WHERE f.scan_id = ?
               ORDER BY f.ai_probability DESC, f.path""", (scan_id,))
        out = []
        for r in rows:
            d = dict(r)
            for k in ("features", "contributions", "indicators"):
                d[k] = json.loads(d[k]) if d[k] else {}
            out.append(d)
        return out

    def verdicts(self, scan_id: int) -> List[Dict[str, Any]]:
        return [dict(r) for r in self.conn.execute(
            """SELECT v.*, f.path FROM verdicts v LEFT JOIN files f ON f.id = v.file_id
               WHERE v.scan_id = ? ORDER BY v.level DESC, v.ai_probability DESC""", (scan_id,))]

    def file_history(self, repo: str, path: str) -> List[Dict[str, Any]]:
        return [dict(r) for r in self.conn.execute(
            """SELECT s.id AS scan_id, s.finished_at, s.mode, f.ai_probability, f.content_sha256
               FROM files f JOIN scans s ON s.id = f.scan_id
               WHERE (s.repo_path = ? OR s.source = ?) AND f.path = ? ORDER BY s.id""", (repo, repo, path))]

    @staticmethod
    def _scan_row(r) -> Dict[str, Any]:
        d = dict(r)
        d["languages"] = json.loads(d["languages"]) if d.get("languages") else {}
        return d


# ---- CLI -------------------------------------------------------------------

import click  # noqa: E402


@click.command("history")
@click.option("--store", type=click.Path(dir_okay=False, path_type=Path), default=DEFAULT_STORE, show_default=True)
@click.option("--repo", help="Only scans of this repo (local path or source URL)")
@click.option("--scan", "scan_id", type=int, help="Show one scan's files and verdicts")
@click.option("--file", "file_path", help="With --repo: this path's probability across scans")
@click.option("--limit", type=int, default=20, show_default=True)
@click.option("--json", "as_json", is_flag=True)
def history(store: Path, repo, scan_id, file_path, limit, as_json):
    """List stored scan runs (from `aicd scan --store`)."""
    if not Path(store).expanduser().exists():
        raise click.ClickException(f"no scan store at {store} (run `aicd scan PATH --store {store}` first)")
    if repo and Path(repo).exists():
        repo = str(Path(repo).resolve())
    with ScanStore(store) as db:
        if scan_id is not None:
            scan = db.get_scan(scan_id)
            if scan is None:
                raise click.ClickException(f"no scan #{scan_id} in {store}")
            data: Any = {"scan": scan, "files": db.files(scan_id), "verdicts": db.verdicts(scan_id)}
        elif file_path:
            if not repo:
                raise click.UsageError("--file needs --repo")
            data = db.file_history(repo, file_path)
        else:
            data = db.list_scans(repo, limit)
    if as_json:
        click.echo(json.dumps(data, indent=2))
        return
    if scan_id is not None:
        s = data["scan"]
        click.echo(f"scan #{s['id']}  {s['finished_at']}  {s['mode']}  {s['repo_path']}  "
                   f"AI {s['ai_probability'] * 100:.1f}%  ({s['total_files']} files)")
        for f in data["files"]:
            flags = "".join(c for c, on in (("H", f["phase_heuristic"]), ("M", f["phase_ml"]),
                                            ("E", f["phase_explanation"])) if on)
            click.echo(f"  {f['ai_probability'] * 100:5.1f}%  [{flags:<3}]  {f['path']}")
        return
    if file_path:
        for r in data:
            click.echo(f"  scan #{r['scan_id']}  {r['finished_at']}  {r['mode']:<8}  "
                       f"{r['ai_probability'] * 100:5.1f}%  {(r['content_sha256'] or '')[:12]}")
        return
    if not data:
        click.echo("no scans stored" + (f" for {repo}" if repo else ""))
    for s in data:
        click.echo(f"#{s['id']:<5} {s['finished_at']}  {s['mode']:<8}  AI {s['ai_probability'] * 100:5.1f}%  "
                   f"{s['total_files']:>5} files  {s['source']}")
