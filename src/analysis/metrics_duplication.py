"""Repository-wide (cross-file) code duplication detection."""

from typing import Dict, List, Tuple
from dataclasses import dataclass
from collections import defaultdict
import re


@dataclass
class DuplicateBlock:
    """A block of code duplicated across two or more files."""
    lines: Tuple[str, ...]
    locations: List[Tuple[str, int]]  # (path, start_line)
    end_lines: List[Tuple[str, int]] = None  # optional (path, end_line)


@dataclass
class RepoDuplicationFeatures:
    """Cross-file duplication features for a repository."""
    duplication_ratio: float
    duplicate_blocks: List[DuplicateBlock]


_IMPORT_RE = re.compile(
    r"^(?:from\s+\S+\s+import\s+.+|import\s+\S+|using\s+\S+|\#include\s*[<\"].*[>\"])$"
)
_TRIVIAL_RE = re.compile(
    r"^(?:"
    r"else\s*:?"
    r"|elif\s+.*:"
    r"|else\s*\{?"
    r"|try\s*:?"
    r"|finally\s*:?"
    r"|pass"
    r"|return(?:\s+None)?"
    r"|break|continue"
    r"|[\{\}\(\)\[\];,]+"
    r"|\)\s*;?"
    r")$"
)


def _is_meaningful_line(line: str, min_chars: int = 8) -> bool:
    s = line.strip()
    if len(s) < min_chars:
        return False
    if _IMPORT_RE.match(s):
        return False
    if _TRIVIAL_RE.match(s):
        return False
    return True


class RepoDuplicationAnalyzer:
    """Detects code blocks duplicated across multiple files in a repository."""

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.ngram_size = self.config.get('ngram_size', 5)
        self.top_n_blocks = self.config.get('top_n_blocks', 20)
        self.min_line_chars = self.config.get('min_line_chars', 8)

    def analyze_repo(self, file_contents: Dict[str, str]) -> RepoDuplicationFeatures:
        """Find duplicate code blocks across files."""
        ngram_locations: Dict[Tuple[str, ...], List[Tuple[str, int]]] = defaultdict(list)
        total_ngrams = 0
        n = self.ngram_size

        for file_path, code in file_contents.items():
            numbered_lines = [
                (idx + 1, line.strip())
                for idx, line in enumerate(code.split('\n'))
                if _is_meaningful_line(line, self.min_line_chars)
            ]

            for i in range(len(numbered_lines) - n + 1):
                ngram = tuple(numbered_lines[j][1] for j in range(i, i + n))
                start_line = numbered_lines[i][0]
                ngram_locations[ngram].append((file_path, start_line))
                total_ngrams += 1

        # Cross-file windows only
        raw_blocks: List[DuplicateBlock] = []
        cross_file_duplicate_count = 0
        for ngram, locations in ngram_locations.items():
            distinct_files = {loc[0] for loc in locations}
            if len(distinct_files) >= 2:
                cross_file_duplicate_count += len(locations)
                raw_blocks.append(DuplicateBlock(lines=ngram, locations=locations))

        merged = self._merge_overlapping(raw_blocks)
        # Rank by region length * occurrences
        merged.sort(
            key=lambda b: (len(b.lines) * len(b.locations), len(b.locations)),
            reverse=True,
        )

        duplication_ratio = (
            min(1.0, cross_file_duplicate_count / total_ngrams) if total_ngrams else 0.0
        )

        return RepoDuplicationFeatures(
            duplication_ratio=duplication_ratio,
            duplicate_blocks=merged[: self.top_n_blocks],
        )

    def _merge_overlapping(self, blocks: List[DuplicateBlock]) -> List[DuplicateBlock]:
        """Merge contiguous overlapping windows into maximal regions per file-set."""
        # Group by frozenset of files
        by_files: Dict[frozenset, List[DuplicateBlock]] = defaultdict(list)
        for block in blocks:
            files = frozenset(loc[0] for loc in block.locations)
            by_files[files].append(block)

        merged: List[DuplicateBlock] = []
        for files, group in by_files.items():
            # Per file, collect start lines and merge contiguous runs spaced by ~1 line
            # Build map file -> sorted starts from all windows
            starts_by_file: Dict[str, List[int]] = defaultdict(list)
            line_bags: List[Tuple[str, ...]] = []
            for block in group:
                line_bags.append(block.lines)
                for path, start in block.locations:
                    starts_by_file[path].append(start)

            # Representative lines: longest ngram then extend conceptually by count of windows
            # Use union of unique consecutive content by taking max-length lines tuple
            best_lines = max(line_bags, key=len) if line_bags else tuple()
            # Expand region length estimate: number of overlapping windows + ngram_size - 1
            region_len = len(best_lines) + max(0, len(group) - 1)

            locations = []
            end_lines = []
            for path in sorted(files):
                starts = sorted(set(starts_by_file[path]))
                if not starts:
                    continue
                # Merge starts within ngram_size into runs
                run_start = starts[0]
                prev = starts[0]
                for s in starts[1:]:
                    if s <= prev + self.ngram_size:
                        prev = s
                    else:
                        locations.append((path, run_start))
                        end_lines.append((path, prev + self.ngram_size - 1))
                        run_start = s
                        prev = s
                locations.append((path, run_start))
                end_lines.append((path, prev + self.ngram_size - 1))

            # Pad representative lines to region_len for ranking
            padded = best_lines + tuple([""] * max(0, region_len - len(best_lines)))
            merged.append(
                DuplicateBlock(lines=padded[:region_len] or best_lines, locations=locations, end_lines=end_lines)
            )
        return merged
