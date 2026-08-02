"""Repository-wide (cross-file) code duplication detection."""

from typing import Dict, List, Tuple
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class DuplicateBlock:
    """A block of code duplicated across two or more files."""
    lines: Tuple[str, ...]
    locations: List[Tuple[str, int]]


@dataclass
class RepoDuplicationFeatures:
    """Cross-file duplication features for a repository."""
    duplication_ratio: float
    duplicate_blocks: List[DuplicateBlock]


class RepoDuplicationAnalyzer:
    """Detects code blocks duplicated across multiple files in a repository."""

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.ngram_size = self.config.get('ngram_size', 3)
        self.top_n_blocks = self.config.get('top_n_blocks', 20)

    def analyze_repo(self, file_contents: Dict[str, str]) -> RepoDuplicationFeatures:
        """Find duplicate code blocks across files.

        Args:
            file_contents: Mapping of relative file path to source code

        Returns:
            RepoDuplicationFeatures
        """
        ngram_locations: Dict[Tuple[str, ...], List[Tuple[str, int]]] = defaultdict(list)
        total_ngrams = 0
        n = self.ngram_size

        for file_path, code in file_contents.items():
            lines = [line.strip() for line in code.split('\n') if line.strip()]

            for i in range(len(lines) - n + 1):
                ngram = tuple(lines[i:i + n])
                ngram_locations[ngram].append((file_path, i + 1))
                total_ngrams += 1

        duplicate_blocks = []
        cross_file_duplicate_count = 0

        for ngram, locations in ngram_locations.items():
            distinct_files = {loc[0] for loc in locations}
            if len(distinct_files) >= 2:
                cross_file_duplicate_count += len(locations)
                duplicate_blocks.append(DuplicateBlock(lines=ngram, locations=locations))

        duplicate_blocks.sort(key=lambda b: len(b.locations), reverse=True)

        duplication_ratio = (
            min(1.0, cross_file_duplicate_count / total_ngrams) if total_ngrams else 0.0
        )

        return RepoDuplicationFeatures(
            duplication_ratio=duplication_ratio,
            duplicate_blocks=duplicate_blocks[:self.top_n_blocks],
        )
