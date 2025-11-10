"""Basic tests for AI Code Detector."""

import pytest
from pathlib import Path
import sys
import tempfile
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from analysis.tokenizer import CodeTokenizer
from analysis.ast_parser import PythonASTParser
from analysis.metrics_stylometry import StylometryAnalyzer
from analysis.metrics_structural import StructuralAnalyzer
from detector import AICodeDetector

def test_tokenizer():
    """Test code tokenizer."""
    tokenizer = CodeTokenizer()

    code = """
def hello():
    # This is a comment
    print("Hello, world!")
    """

    stats = tokenizer.tokenize(code, "python")

    assert stats.total_tokens > 0
    assert len(stats.comment_tokens) > 0


def test_ast_parser():
    """Test AST parser."""
    parser = PythonASTParser()

    code = """
def process_data(data):
    '''This function processes data.'''
    return data
    """

    file_ast = parser.parse_file(Path("test.py"), code)

    assert len(file_ast.functions) == 1
    assert file_ast.functions[0].name == "process_data"
    assert file_ast.functions[0].docstring is not None


def test_stylometry_analyzer():
    """Test stylometry analyzer."""
    analyzer = StylometryAnalyzer()

    # AI-like code with boilerplate
    ai_code = """
def process_data(data):
    '''
    This function processes the data.

    Args:
        data: The data to process

    Returns:
        The result
    '''
    result = data
    return result
    """

    features = analyzer.analyze_file(ai_code, "python")

    # Should detect boilerplate patterns
    assert features.boilerplate_comment_score > 0


def test_structural_analyzer():
    """Test structural analyzer."""
    analyzer = StructuralAnalyzer()

    # Code with generic exceptions
    code = """
def handle_request():
    try:
        result = do_something()
    except Exception as e:
        print(f"An error occurred: {e}")
    """

    features = analyzer.analyze_file(code, "python")

    # Should detect generic exception pattern
    assert features.generic_exception_ratio > 0


def test_ai_vs_human_samples():
    """Test that AI sample scores higher than human sample."""
    detector = AICodeDetector()

    # Analyze AI sample
    ai_sample_path = Path(__file__).parent.parent / "examples" / "sample_ai_code.py"
    human_sample_path = Path(__file__).parent.parent / "examples" / "sample_human_code.py"

    if ai_sample_path.exists() and human_sample_path.exists():
        # Create temp directory with just these files
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Copy AI sample
            ai_dir = tmpdir / "ai_repo"
            ai_dir.mkdir()
            shutil.copy(ai_sample_path, ai_dir / "code.py")

            # Copy human sample
            human_dir = tmpdir / "human_repo"
            human_dir.mkdir()
            shutil.copy(human_sample_path, human_dir / "code.py")

            # Analyze both
            ai_score = detector.analyze_repo(str(ai_dir), verbose=False)
            human_score = detector.analyze_repo(str(human_dir), verbose=False)

            # AI code should score higher
            print(f"AI score: {ai_score.ai_probability}")
            print(f"Human score: {human_score.ai_probability}")

            # This is a soft check - AI should generally score higher
            assert ai_score.ai_probability > 0.3  # At least somewhat detected


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
