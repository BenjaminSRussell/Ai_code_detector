"""Test for bug fix in AI Code Detector."""

import pytest
from pathlib import Path
import sys
import tempfile
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from detector import AICodeDetector
from ingest.file_filter import FileInfo

def test_syntactically_incorrect_file():
    """Test that a syntactically incorrect file gets a low score."""
    detector = AICodeDetector()

    with tempfile.TemporaryDirectory() as tmpdir:
        repo_root = Path(tmpdir)
        file_path = repo_root / "bad_code.py"

        # Create a Python file with a syntax error
        with open(file_path, "w") as f:
            f.write("def = 1")

        file_info = FileInfo(
            path=file_path,
            relative_path=Path("bad_code.py"),
            language="python",
            line_count=1,
            size_bytes=10,
        )

        # Analyze the file
        file_score = detector._analyze_file(file_info, repo_root)

        # The score should be low because of the syntax error
        assert file_score.ai_probability == 0.0
        assert "error" in file_score.feature_explanations
        assert file_score.feature_explanations["error"] == "AST parsing failed"
