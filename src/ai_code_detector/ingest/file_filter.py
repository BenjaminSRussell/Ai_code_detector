"""File filtering for supported code files."""

import os
from pathlib import Path
from typing import List, Set, Dict, Optional
from dataclasses import dataclass

from .suppressions import Suppressions, SuppressionSummary


@dataclass
class FileInfo:
    """Information about a source code file."""
    path: Path
    relative_path: Path
    language: str
    size_bytes: int
    line_count: int


class FileFilter:
    """Filters and categorizes source code files."""

    # Language detection by extension
    LANGUAGE_MAP = {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".go": "go",
        ".rs": "rust",
        ".c": "c",
        ".cpp": "cpp",
        ".cc": "cpp",
        ".cxx": "cpp",
        ".h": "c",
        ".hpp": "cpp",
        ".java": "java",
        ".rb": "ruby",
        ".php": "php",
        ".cs": "csharp",
        ".swift": "swift",
        ".kt": "kotlin",
        ".scala": "scala",
        ".r": "r",
        ".m": "objective-c",
        ".sh": "bash",
        ".pl": "perl",
        ".lua": "lua",
    }

    def __init__(
        self,
        supported_extensions: List[str],
        excluded_dirs: List[str],
        max_file_size_mb: float = 1.0,
    ):
        """Initialize file filter.

        Args:
            supported_extensions: List of file extensions to include (e.g., ['.py', '.js'])
            excluded_dirs: List of directory names to exclude
            max_file_size_mb: Maximum file size in megabytes
        """
        self.supported_extensions = set(supported_extensions)
        self.excluded_dirs = set(excluded_dirs)
        self.max_file_size_bytes = int(max_file_size_mb * 1024 * 1024)
        # Suppressions: None -> auto-load <root>/.aicdignore; a path -> that file.
        self.suppressions_file: Optional[Path] = None
        self.use_suppressions: bool = True
        self.last_suppressed: SuppressionSummary = SuppressionSummary()

    def scan_directory(self, root_path: Path) -> List[FileInfo]:
        """Scan directory for supported code files.

        Args:
            root_path: Root directory to scan

        Returns:
            List of FileInfo for valid code files
        """
        files = []
        if self.use_suppressions:
            suppressions = Suppressions.load(root_path, self.suppressions_file)
        else:
            suppressions = Suppressions()
        summary = SuppressionSummary(source=suppressions.source, rules=list(suppressions.rules))
        self.last_suppressed = summary

        for dirpath, dirnames, filenames in os.walk(root_path):
            # Filter out excluded directories
            dirnames[:] = [d for d in dirnames if d not in self.excluded_dirs]

            for filename in filenames:
                file_path = Path(dirpath) / filename

                # Check extension
                if file_path.suffix not in self.supported_extensions:
                    continue

                relative_path = file_path.relative_to(root_path)
                if suppressions.rules:
                    rule = suppressions.match(relative_path)
                    if rule is not None:
                        summary.record(relative_path.as_posix(), rule)
                        continue

                # Check file size
                try:
                    size = file_path.stat().st_size
                    if size > self.max_file_size_bytes:
                        continue
                except Exception:
                    continue

                # Count lines
                # (#10) was hard-coded to 0, so every report said "0 lines".
                try:
                    data = file_path.read_bytes()
                    line_count = data.count(b"\n") + (1 if data and not data.endswith(b"\n") else 0)
                except OSError:
                    line_count = 0

                # Detect language
                language = self.LANGUAGE_MAP.get(file_path.suffix, "unknown")

                files.append(FileInfo(
                    path=file_path,
                    relative_path=relative_path,
                    language=language,
                    size_bytes=size,
                    line_count=line_count,
                ))

        return files

    def get_language_distribution(self, files: List[FileInfo]) -> Dict[str, int]:
        """Get distribution of languages in file list.

        Args:
            files: List of FileInfo

        Returns:
            Dict mapping language to file count
        """
        distribution = {}
        for file_info in files:
            lang = file_info.language
            distribution[lang] = distribution.get(lang, 0) + 1
        return distribution
