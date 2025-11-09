"""Git repository loader with commit history analysis."""

import os
import tempfile
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass
from datetime import datetime

try:
    import git
    from git import Repo
    HAS_GITPYTHON = True
except ImportError:
    HAS_GITPYTHON = False


@dataclass
class CommitInfo:
    """Information about a single commit."""
    sha: str
    author: str
    email: str
    timestamp: datetime
    message: str
    files_changed: List[str]
    lines_added: int
    lines_deleted: int


@dataclass
class RepoInfo:
    """Repository metadata and history."""
    path: Path
    is_git: bool
    remote_url: Optional[str]
    commits: List[CommitInfo]
    authors: List[str]
    total_commits: int
    first_commit_date: Optional[datetime]
    last_commit_date: Optional[datetime]


class GitLoader:
    """Loads git repositories from URL or local path."""

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize git loader.

        Args:
            cache_dir: Directory for cloning remote repos
        """
        if not HAS_GITPYTHON:
            raise ImportError("GitPython is required. Install with: pip install GitPython")

        self.cache_dir = cache_dir or Path(tempfile.gettempdir()) / "ai_code_detector_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load(self, source: str) -> RepoInfo:
        """Load repository from URL or local path.

        Args:
            source: GitHub URL or local path

        Returns:
            RepoInfo with repository metadata
        """
        if self._is_url(source):
            repo_path = self._clone_repo(source)
            remote_url = source
        else:
            repo_path = Path(source).resolve()
            if not repo_path.exists():
                raise ValueError(f"Path does not exist: {source}")
            remote_url = None

        return self._analyze_repo(repo_path, remote_url)

    def _is_url(self, source: str) -> bool:
        """Check if source is a URL."""
        return source.startswith(("http://", "https://", "git@"))

    def _clone_repo(self, url: str) -> Path:
        """Clone repository to cache directory.

        Args:
            url: Git repository URL

        Returns:
            Path to cloned repository
        """
        # Extract repo name from URL
        repo_name = url.rstrip("/").split("/")[-1]
        if repo_name.endswith(".git"):
            repo_name = repo_name[:-4]

        target_path = self.cache_dir / repo_name

        if target_path.exists():
            print(f"Repository already cloned at {target_path}")
            try:
                repo = Repo(target_path)
                repo.remotes.origin.pull()
                print("Updated existing repository")
            except Exception as e:
                print(f"Warning: Could not update repo: {e}")
        else:
            print(f"Cloning {url} to {target_path}")
            Repo.clone_from(url, target_path)

        return target_path

    def _analyze_repo(self, repo_path: Path, remote_url: Optional[str]) -> RepoInfo:
        """Analyze repository history and metadata.

        Args:
            repo_path: Path to repository
            remote_url: Remote URL if cloned

        Returns:
            RepoInfo with analysis results
        """
        try:
            repo = Repo(repo_path)
            is_git = True
        except Exception:
            # Not a git repo, just a directory
            return RepoInfo(
                path=repo_path,
                is_git=False,
                remote_url=remote_url,
                commits=[],
                authors=[],
                total_commits=0,
                first_commit_date=None,
                last_commit_date=None,
            )

        commits = []
        authors_set = set()

        try:
            for commit in repo.iter_commits():
                author = commit.author.name
                email = commit.author.email
                authors_set.add(author)

                # Get file changes
                files_changed = []
                lines_added = 0
                lines_deleted = 0

                try:
                    if commit.parents:
                        diffs = commit.parents[0].diff(commit, create_patch=True)
                        for diff in diffs:
                            if diff.a_path:
                                files_changed.append(diff.a_path)
                            # Count line changes
                            if diff.diff:
                                diff_text = diff.diff.decode('utf-8', errors='ignore')
                                for line in diff_text.split('\n'):
                                    if line.startswith('+') and not line.startswith('+++'):
                                        lines_added += 1
                                    elif line.startswith('-') and not line.startswith('---'):
                                        lines_deleted += 1
                except Exception:
                    pass  # Skip if we can't get diff

                commits.append(CommitInfo(
                    sha=commit.hexsha,
                    author=author,
                    email=email,
                    timestamp=datetime.fromtimestamp(commit.committed_date),
                    message=commit.message.strip(),
                    files_changed=files_changed,
                    lines_added=lines_added,
                    lines_deleted=lines_deleted,
                ))

        except Exception as e:
            print(f"Warning: Could not fully analyze git history: {e}")

        return RepoInfo(
            path=repo_path,
            is_git=is_git,
            remote_url=remote_url,
            commits=commits,
            authors=sorted(authors_set),
            total_commits=len(commits),
            first_commit_date=commits[-1].timestamp if commits else None,
            last_commit_date=commits[0].timestamp if commits else None,
        )
