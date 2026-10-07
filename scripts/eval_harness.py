#!/usr/bin/env python3
"""Labeled eval harness (#11). Same as `aicd eval`; see src/ai_code_detector/evaluation.py.

    python scripts/eval_harness.py --manifest datasets/manifest.yaml [--train]
"""
import sys
from pathlib import Path

try:
    from ai_code_detector.evaluation import main
except ImportError:  # running from a checkout without `pip install -e .`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from ai_code_detector.evaluation import main

if __name__ == "__main__":
    main(prog_name="eval_harness.py")
