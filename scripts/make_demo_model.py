"""Génère le modèle ONNX de démonstration (poids ALÉATOIRES, sans PyTorch) : python scripts/make_demo_model.py

Il respecte le contrat de l'API (entrée « image » 1x3x224x224 -> sorties « logits » et « cam »)
pour que toute l'application fonctionne avant l'entraînement. Ses prédictions n'ont AUCUNE valeur.
"""

import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT = Path(__file__).resolve().parent.parent / "backend" / "demo_model"


def main():
    rng = np.random.default_rng(42)
    w1 = (rng.normal(0, 0.02, (8, 3, 32, 32))).astype(np.float32)  # « patchify » 32x32 -> carte 7x7
    w2 = rng.normal(0, 1.0, (1, 8, 1, 1)).astype(np.float32)  # CAM : combinaison des 8 canaux
    w3 = rng.normal(0, 1.5, (2, 8)).astype(np.float32)  # classifieur linéaire (classe 0 = fake)
    b3 = np.zeros(2, dtype=np.float32)
    axes = np.array([1], dtype=np.int64)

    inits = [
        numpy_helper.from_array(w1, "w1"),
        numpy_helper.from_array(w2, "w2"),
        numpy_helper.from_array(w3, "w3"),
        numpy_helper.from_array(b3, "b3"),
        numpy_helper.from_array(axes, "axes"),
    ]
    nodes = [
        helper.make_node("Conv", ["image", "w1"], ["c1"], kernel_shape=[32, 32], strides=[32, 32]),
        helper.make_node("Relu", ["c1"], ["feat"]),
        helper.make_node("Conv", ["feat", "w2"], ["cam4"], kernel_shape=[1, 1]),
        helper.make_node("Squeeze", ["cam4", "axes"], ["cam"]),
        helper.make_node("GlobalAveragePool", ["feat"], ["pool"]),
        helper.make_node("Flatten", ["pool"], ["flat"], axis=1),
        helper.make_node("Gemm", ["flat", "w3", "b3"], ["logits"], transB=1),
    ]
    graph = helper.make_graph(
        nodes,
        "deepguard_demo",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, ["batch", 3, 224, 224])],
        [
            helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 2]),
            helper.make_tensor_value_info("cam", TensorProto.FLOAT, ["batch", 7, 7]),
        ],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)

    OUT.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(OUT / "deepguard-demo.onnx"))
    meta = {
        "name": "DeepGuard (démo)",
        "architecture": "démo : poids aléatoires, non entraîné",
        "classes": ["fake", "real"],
        "input_size": 224,
        "demo": True,
        "version": "demo",
        "note": "Modèle de démonstration : ses prédictions n'ont aucune valeur.",
    }
    (OUT / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Écrit :", OUT)


if __name__ == "__main__":
    main()
