"""Quick test of the detector."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from analysis.tokenizer import CodeTokenizer
from analysis.metrics_stylometry import StylometryAnalyzer

# Test 1: Tokenizer
print("Test 1: Tokenizer")
print("-" * 40)

tokenizer = CodeTokenizer()

code = """
def hello():
    # This is a comment
    print("Hello, world!")
"""

stats = tokenizer.tokenize(code, "python")
print(f"Total tokens: {stats.total_tokens}")
print(f"Comments: {len(stats.comment_tokens)}")
print(f"Identifiers: {len(stats.identifier_tokens)}")
print("✓ Tokenizer works!")
print()

# Test 2: Stylometry
print("Test 2: Stylometry Analyzer")
print("-" * 40)

analyzer = StylometryAnalyzer()

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
print(f"Boilerplate score: {features.boilerplate_comment_score:.2f}")
print(f"Comment ratio: {features.comment_to_code_ratio:.2f}")
print(f"Generic name ratio: {features.generic_name_ratio:.2f}")
print("✓ Stylometry analyzer works!")
print()

# Test 3: Aggregator
print("Test 3: Aggregator")
print("-" * 40)

from model.aggregator import HeuristicAggregator
from analysis.metrics_stylometry import StylometricFeatures
from analysis.metrics_structural import StructuralFeatures

aggregator = HeuristicAggregator()

# Create sample features
stylometry = StylometricFeatures(
    comment_to_code_ratio=0.5,
    avg_comment_length=100,
    boilerplate_comment_score=0.8,
    tutorial_comment_score=0.3,
    avg_identifier_length=6.5,
    generic_name_ratio=0.4,
    identifier_entropy=2.5,
    indentation_consistency=0.98,
    avg_line_length=60,
    trailing_whitespace_ratio=0.01,
    code_duplication_score=0.3,
    intra_file_similarity=0.2,
)

structural = StructuralFeatures(
    avg_cyclomatic_complexity=2.5,
    complexity_to_docstring_ratio=80,
    over_explained_simple_functions=0.5,
    generic_exception_ratio=0.6,
    print_error_pattern_score=0.4,
    try_except_ratio=0.7,
    unused_function_ratio=0.3,
    unused_import_ratio=0.2,
    unreachable_code_score=0.1,
    rare_api_combination_score=0.0,
    missing_cleanup_score=0.2,
)

file_score = aggregator.aggregate_file_features(stylometry, structural)
print(f"AI Probability: {file_score.ai_probability*100:.1f}%")
print(f"Stylometry score: {file_score.stylometry_score:.2f}")
print(f"Structural score: {file_score.structural_score:.2f}")
print("✓ Aggregator works!")

print()
print("=" * 40)
print("All tests passed!")
