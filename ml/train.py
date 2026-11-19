"""Model training entry point.

Wraps training routines for modern EfficientNet-B3 Sex-Fusion models.
"""

from __future__ import annotations

import sys
from pathlib import Path

def get_project_root() -> Path:
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "configs").is_dir() and (current / "backend").is_dir():
            return current
        current = current.parent
    return Path(__file__).resolve().parent


ROOT_DIR = get_project_root()
BACKEND_DIR = ROOT_DIR / "backend"
for p in [str(ROOT_DIR), str(BACKEND_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from ml.train.train import main as run_train, train

if __name__ == "__main__":
    run_train()
