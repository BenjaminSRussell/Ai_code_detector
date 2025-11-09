"""File filtering for supported code files."""

import os
from pathlib import Path
from typing import List, Set, Dict
from dataclasses import dataclass


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

    def scan_directory(self, root_path: Path) -> List[FileInfo]:
        """Scan directory for supported code files.

        Args:
            root_path: Root directory to scan

        Returns:
            List of FileInfo for valid code files
        """
        files = []

        for dirpath, dirnames, filenames in os.walk(root_path):
            # Filter out excluded directories
            dirnames[:] = [d for d in dirnames if d not in self.excluded_dirs]

            for filename in filenames:
                file_path = Path(dirpath) / filename

                # Check extension
                if file_path.suffix not in self.supported_extensions:
                    continue

                # Check file size
                try:
                    size = file_path.stat().st_size
                    if size > self.max_file_size_bytes:
                        continue
                except Exception:
                    continue

                # Count lines
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        line_count = sum(1 for _ in f)
                except Exception:
                    line_count = 0

                # Detect language
                language = self.LANGUAGE_MAP.get(file_path.suffix, "unknown")

                files.append(FileInfo(
                    path=file_path,
                    relative_path=file_path.relative_to(root_path),
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
