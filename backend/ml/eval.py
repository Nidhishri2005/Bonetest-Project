"""Baseline evaluation and model metric evaluation entry point.

Wraps evaluation routines for legacy baselines and ensemble models.
Reproduces legacy CNN baseline MAE on the frozen test holdout split.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root and backend to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
for p in [str(ROOT_DIR), str(BACKEND_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from ml.eval.evaluate_baseline import evaluate_baseline, main as run_baseline_eval

if __name__ == "__main__":
    run_baseline_eval()
