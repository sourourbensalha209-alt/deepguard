"""Quantification INT8 dynamique d'un modèle ONNX et mesure de son impact.

Sans PyTorch : n'utilise que onnxruntime et numpy.
"""

import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _p50_ms(session, x, n=30, warmup=5):
    for _ in range(warmup):
        session.run(None, {"image": x})
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        session.run(None, {"image": x})
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.percentile(times, 50))


def quantize_int8(src, dst, size=224):
    """Quantifie les poids de `src` en INT8 vers `dst`.

    Retourne un dict (taille, latence p50, écart maximal de probabilité avec le modèle FP32) ou None si
    ce modèle ne supporte pas la quantification dynamique (fichier `dst` supprimé dans ce cas).
    L'écart de probabilité est mesuré sur des entrées aléatoires : valide-le sur le jeu de test avant de
    déployer la version INT8.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    src, dst = Path(src), Path(dst)
    try:
        quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)
        fp32 = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"])
        int8 = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])
        x = np.random.default_rng(0).normal(size=(4, 3, size, size)).astype(np.float32)
        ref = _softmax(fp32.run(["logits"], {"image": x})[0])
        out = _softmax(int8.run(["logits"], {"image": x})[0])
        one = x[:1]
        return {
            "int8_size_mb": round(dst.stat().st_size / 1e6, 2),
            "int8_onnx_cpu_p50_ms": round(_p50_ms(int8, one), 1),
            "int8_max_prob_diff": round(float(np.abs(ref - out).max()), 4),
        }
    except Exception as exc:  # opérateur non supporté, etc. : on garde simplement le modèle FP32
        print(f"[info] quantification INT8 ignorée : {exc}")
        dst.unlink(missing_ok=True)
        return None
