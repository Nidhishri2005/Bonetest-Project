"""Diagnostic: inspect hand_validator_best.pt and hand_validator.onnx (read-only).

Reports:
  - checkpoint metadata (classes, val_acc, epoch)
  - strict key comparison against HandXRayValidator (missing / unexpected keys)
  - whether loading with strict=False would leave layers randomly initialised
  - PyTorch(.pt) vs ONNX output agreement on a sample input
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
for p in (str(ROOT), str(BACKEND)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch

from app.models.validator import HandXRayValidator, VALIDATOR_CLASSES

CKPT = BACKEND / "checkpoints" / "hand_validator_best.pt"
ONNX = BACKEND / "checkpoints" / "hand_validator.onnx"


def main() -> None:
    print(f"Checkpoint path : {CKPT}")
    print(f"Exists          : {CKPT.exists()}  size={CKPT.stat().st_size if CKPT.exists() else 'n/a'} bytes")
    print(f"ONNX path       : {ONNX}")
    print(f"Exists          : {ONNX.exists()}  size={ONNX.stat().st_size if ONNX.exists() else 'n/a'} bytes")

    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    meta = {k: v for k, v in ckpt.items() if k != "model_state_dict"}
    print("\nCheckpoint metadata:")
    for k, v in meta.items():
        print(f"  {k}: {v}")

    sd = ckpt.get("model_state_dict", ckpt)
    model = HandXRayValidator(num_classes=len(VALIDATOR_CLASSES))
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(sd.keys())
    missing = sorted(model_keys - ckpt_keys)
    unexpected = sorted(ckpt_keys - model_keys)
    shape_mismatch = [
        k for k in model_keys & ckpt_keys if tuple(sd[k].shape) != tuple(model.state_dict()[k].shape)
    ]
    print(f"\nModel keys: {len(model_keys)}  Checkpoint keys: {len(ckpt_keys)}")
    print(f"Missing in checkpoint (would stay RANDOM with strict=False): {len(missing)}")
    for k in missing[:20]:
        print(f"   - {k}")
    print(f"Unexpected in checkpoint (silently ignored): {len(unexpected)}")
    for k in unexpected[:20]:
        print(f"   - {k}")
    print(f"Shape mismatches: {shape_mismatch}")

    model.load_state_dict(sd, strict=True)
    print("strict=True load: OK")

    ckpt_classes = meta.get("classes")
    print(f"\nCode VALIDATOR_CLASSES      : {VALIDATOR_CLASSES}")
    print(f"Checkpoint 'classes' field  : {ckpt_classes}")
    print(f"Class order identical       : {list(ckpt_classes or []) == list(VALIDATOR_CLASSES)}")

    # PT vs ONNX agreement
    if ONNX.exists():
        import onnxruntime as ort

        model.eval()
        torch.manual_seed(0)
        x = torch.randn(4, 3, 224, 224)
        with torch.no_grad():
            pt = model(x).numpy()
        sess = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])
        on = sess.run(None, {"image": x.numpy()})[0]
        print(f"\nPT vs ONNX max |logit diff| on random input: {float(np.max(np.abs(pt - on))):.3e}")


if __name__ == "__main__":
    main()
