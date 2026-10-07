"""Git history feature extraction for AI detection."""

from typing import Dict, List
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import Counter
import math

from ..ingest.git_loader import RepoInfo, CommitInfo


@dataclass
class HistoryFeatures:
    """Git history features for AI detection."""
    # Commit patterns
    commit_burst_score: float
    avg_lines_per_commit: float
    commit_gini_coefficient: float

    # Author patterns
    author_diversity: float
    commit_message_entropy: float
    avg_message_length: float
    generic_message_ratio: float

    # File patterns
    avg_edits_per_file: float
    files_created_in_burst: float
    repo_age_days: float
    commits_per_day: float


class HistoryAnalyzer:
    """Analyzes git history for AI generation patterns."""

    # Generic commit messages common in AI-generated repos
    GENERIC_MESSAGES = {
        'initial commit',
        'first commit',
        'update',
        'updates',
        'fix',
        'fixes',
        'bug fix',
        'minor fix',
        'changes',
        'minor changes',
        'improvements',
        'refactor',
        'cleanup',
        'wip',
        'work in progress',
        'temp',
        'test',
        'testing',
        'add files',
        'add code',
        'update files',
        'update code',
    }

    def __init__(self, config: Dict = None):
        """Initialize analyzer.

        Args:
            config: Configuration dict
        """
        self.config = config or {}
        self.recent_days = self.config.get('recent_days_threshold', 30)
        self.burst_threshold = self.config.get('burst_threshold', 1000)

    def analyze_repo(self, repo_info: RepoInfo) -> HistoryFeatures:
        """Analyze repository history.

        Args:
            repo_info: Repository information with commit history

        Returns:
            HistoryFeatures
        """
        if not repo_info.is_git or not repo_info.commits:
            return self._default_features()

        commits = repo_info.commits

        # Commit patterns
        burst_score = self._calculate_burst_score(commits)
        avg_lines = self._calculate_avg_lines_per_commit(commits)
        gini = self._calculate_gini_coefficient(commits)

        # Author patterns
        author_diversity = self._calculate_author_diversity(repo_info)
        message_entropy = self._calculate_message_entropy(commits)
        avg_msg_len = self._calculate_avg_message_length(commits)
        generic_ratio = self._calculate_generic_message_ratio(commits)

        # File patterns
        avg_edits = self._calculate_avg_edits_per_file(commits)
        burst_files = self._calculate_burst_file_creation(commits)

        # Time patterns
        repo_age = self._calculate_repo_age(repo_info)
        commits_per_day = self._calculate_commits_per_day(repo_info)

        return HistoryFeatures(
            commit_burst_score=burst_score,
            avg_lines_per_commit=avg_lines,
            commit_gini_coefficient=gini,
            author_diversity=author_diversity,
            commit_message_entropy=message_entropy,
            avg_message_length=avg_msg_len,
            generic_message_ratio=generic_ratio,
            avg_edits_per_file=avg_edits,
            files_created_in_burst=burst_files,
            repo_age_days=repo_age,
            commits_per_day=commits_per_day,
        )

    def _default_features(self) -> HistoryFeatures:
        """Return default features for non-git repos."""
        return HistoryFeatures(
            commit_burst_score=0.0,
            avg_lines_per_commit=0.0,
            commit_gini_coefficient=0.0,
            author_diversity=1.0,
            commit_message_entropy=0.0,
            avg_message_length=0.0,
            generic_message_ratio=0.0,
            avg_edits_per_file=0.0,
            files_created_in_burst=0.0,
            repo_age_days=0.0,
            commits_per_day=0.0,
        )

    def _calculate_burst_score(self, commits: List[CommitInfo]) -> float:
        """Calculate commit burst score.

        High score indicates large amounts of code added in few commits.
        Returns 0-1.
        """
        if not commits:
            return 0.0

        # Find commits with LOC > threshold
        burst_commits = [c for c in commits if c.lines_added > self.burst_threshold]

        return min(1.0, len(burst_commits) / len(commits))

    def _calculate_avg_lines_per_commit(self, commits: List[CommitInfo]) -> float:
        """Calculate average lines changed per commit."""
        if not commits:
            return 0.0

        total_lines = sum(c.lines_added + c.lines_deleted for c in commits)
        return total_lines / len(commits)

    def _calculate_gini_coefficient(self, commits: List[CommitInfo]) -> float:
        """Calculate Gini coefficient of LOC distribution.

        Higher value means fewer commits created most of the code (AI pattern).
        Returns 0-1.
        """
        if not commits:
            return 0.0

        # Get LOC per commit
        loc_values = [c.lines_added + c.lines_deleted for c in commits]
        loc_values.sort()

        n = len(loc_values)
        if n == 0:
            return 0.0

        # Calculate Gini
        cumsum = 0
        for i, value in enumerate(loc_values):
            cumsum += value * (n - i)

        if sum(loc_values) == 0:
            return 0.0

        return 1 - (2 * cumsum) / (n * sum(loc_values))

    def _calculate_author_diversity(self, repo_info: RepoInfo) -> float:
        """Calculate author diversity.

        Returns ratio of unique authors to total commits.
        Higher is more diverse (less likely AI).
        """
        if not repo_info.commits:
            return 1.0

        unique_authors = len(repo_info.authors)
        total_commits = repo_info.total_commits

        return unique_authors / max(total_commits, 1)

    def _calculate_message_entropy(self, commits: List[CommitInfo]) -> float:
        """Calculate Shannon entropy of commit messages.

        Low entropy suggests repetitive messages (AI pattern).
        """
        if not commits:
            return 0.0

        # Tokenize messages
        all_words = []
        for commit in commits:
            words = commit.message.lower().split()
            all_words.extend(words)

        if not all_words:
            return 0.0

        # Calculate entropy
        word_counts = Counter(all_words)
        total = len(all_words)

        entropy = 0.0
        for count in word_counts.values():
            prob = count / total
            if prob > 0:
                entropy -= prob * math.log2(prob)

        return entropy

    def _calculate_avg_message_length(self, commits: List[CommitInfo]) -> float:
        """Calculate average commit message length."""
        if not commits:
            return 0.0

        total_length = sum(len(c.message) for c in commits)
        return total_length / len(commits)

    def _calculate_generic_message_ratio(self, commits: List[CommitInfo]) -> float:
        """Calculate ratio of generic commit messages.

        Returns 0-1, higher means more generic messages.
        """
        if not commits:
            return 0.0

        generic_count = 0

        for commit in commits:
            msg_lower = commit.message.lower().strip()
            if msg_lower in self.GENERIC_MESSAGES:
                generic_count += 1

        return generic_count / len(commits)

    def _calculate_avg_edits_per_file(self, commits: List[CommitInfo]) -> float:
        """Calculate average number of commits touching each file.

        Low value suggests files created and never edited (AI pattern).
        """
        if not commits:
            return 0.0

        # Track files and their edit counts
        file_edits = Counter()

        for commit in commits:
            for file_path in commit.files_changed:
                file_edits[file_path] += 1

        if not file_edits:
            return 0.0

        return sum(file_edits.values()) / len(file_edits)

    def _calculate_burst_file_creation(self, commits: List[CommitInfo]) -> float:
        """Calculate ratio of files created in single commit.

        High value suggests repo created all at once (AI pattern).
        Returns 0-1.
        """
        if not commits or len(commits) < 2:
            return 0.0

        # Track unique files per commit
        all_files = set()
        max_files_in_commit = 0

        for commit in commits:
            files_in_commit = len(commit.files_changed)
            max_files_in_commit = max(max_files_in_commit, files_in_commit)
            all_files.update(commit.files_changed)

        if not all_files:
            return 0.0

        return max_files_in_commit / len(all_files)

    def _calculate_repo_age(self, repo_info: RepoInfo) -> float:
        """Calculate repository age in days."""
        if not repo_info.first_commit_date or not repo_info.last_commit_date:
            return 0.0

        delta = repo_info.last_commit_date - repo_info.first_commit_date
        return delta.total_seconds() / 86400  # Convert to days

    def _calculate_commits_per_day(self, repo_info: RepoInfo) -> float:
        """Calculate average commits per day."""
        age_days = self._calculate_repo_age(repo_info)

        if age_days == 0:
            return 0.0

        return repo_info.total_commits / max(age_days, 1)
