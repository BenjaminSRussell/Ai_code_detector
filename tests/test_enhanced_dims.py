"""Regression: enhanced mode used to build MLClassifier(768, 34) while the hash
embedder emits 256 dims and _features_to_vector 23, so every file failed with
"shapes not aligned" and was silently dropped (0 files, 0% AI)."""
import shutil
from pathlib import Path

import pytest

from ai_code_detector.detector_enhanced import EnhancedAICodeDetector
from ai_code_detector.model.classifier import MLClassifier

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def test_enhanced_mode_scores_every_file(tmp_path):
    for name in ("sample_ai_code.py", "sample_human_code.py"):
        shutil.copy(EXAMPLES / name, tmp_path / name)
    det = EnhancedAICodeDetector(use_ml=True, use_explanations=False, embedder_backend="hash")
    score = det.analyze_repo(str(tmp_path), verbose=False)
    assert score.total_files_analyzed == 2
    assert all(0.0 < fs.ai_probability < 1.0 for fs in score.file_scores)


def test_mismatched_model_fails_loudly(tmp_path):
    model = tmp_path / "m.json"
    MLClassifier(embedding_dim=768, feature_dim=34).save(model)
    with pytest.raises(ValueError, match="Retrain"):
        EnhancedAICodeDetector(use_ml=True, use_explanations=False, embedder_backend="hash",
                               ml_model_path=model)
