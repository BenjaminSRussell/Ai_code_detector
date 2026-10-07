"""Weight math of the real HeuristicAggregator (#17).

The previous version stubbed `analysis.*` in sys.modules (with `object` as
the feature classes), which broke every test collected after it, and only
re-derived the formula inline. This exercises the actual class.
"""
from ai_code_detector.model.aggregator import FileScore, HeuristicAggregator


def make(weights, s=1.0, t=1.0, h=0.0):
    agg = HeuristicAggregator({"weights": weights})
    agg._score_stylometry = lambda _f: s
    agg._score_structural = lambda _f: t
    agg._score_history = lambda _f: h
    agg._extract_explanations = lambda *_a: []
    return agg


def test_maxed_file_reaches_one_with_renormalized_weights():
    agg = make({"stylometry": 0.4, "structural": 0.4, "history": 0.2})
    assert abs(agg.aggregate_file_features(None, None).ai_probability - 1.0) < 1e-9


def test_zero_file_weights_do_not_divide_by_zero():
    agg = make({"stylometry": 0.0, "structural": 0.0})
    assert agg.aggregate_file_features(None, None).ai_probability == 0.0


def test_history_weight_is_applied_to_repo_score():
    agg = make({"history": 0.2}, h=0.0)
    files = [FileScore(file_path="a.py", ai_probability=1.0, stylometry_score=1.0,
                       structural_score=1.0, feature_explanations=[], suspicious_snippets=[])]
    repo = agg.aggregate_repo_features(files, history=None, total_lines=10, language_dist={"python": 1})
    assert abs(repo.ai_probability - 0.8) < 1e-9
