"""Quantification INT8 d'un modèle ONNX (sans PyTorch)."""

from pathlib import Path

import numpy as np
import onnxruntime as ort

from src.utils.quantize import quantize_int8

DEMO = Path(__file__).resolve().parent.parent / "backend" / "demo_model" / "deepguard-demo.onnx"


def test_quantize_demo_model(tmp_path):
    dst = tmp_path / "demo-int8.onnx"
    result = quantize_int8(DEMO, dst, size=224)
    if result is None:  # opérateurs non supportés par cette version d'onnxruntime : repli sur FP32
        assert not dst.exists()
        return
    assert dst.exists() and result["int8_max_prob_diff"] < 0.25
    sess = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])
    logits, cam = sess.run(None, {"image": np.zeros((1, 3, 224, 224), dtype=np.float32)})
    assert logits.shape == (1, 2) and cam.shape == (1, 7, 7)
