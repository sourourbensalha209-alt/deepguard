"""Export ONNX, parité PyTorch/ONNX, benchmark et quantification INT8 : python -m src.models.export_onnx

Le modèle exporté a UNE entrée « image » et DEUX sorties :
  - logits : (N, 2), classe 0 = fake ;
  - cam    : (N, h, w), carte d'explication (CAM pour un CNN, Attention Rollout pour un ViT).
Ainsi l'API n'a besoin que d'ONNX Runtime pour prédire ET expliquer, sans PyTorch.
"""

import json
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

from src.data.dataset import CLASSES, MEAN, STD
from src.models.architecture import build_model
from src.utils.config import git_sha, load_params
from src.utils.quantize import quantize_int8


class CNNWithCAM(nn.Module):
    """CNN : sorties = (logits, carte CAM de la classe « fake »)."""

    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, x):
        f = self.m.forward_features(x)
        logits = self.m.forward_head(f)
        cam = torch.einsum("bchw,c->bhw", f, self.m.classifier.weight[0])
        return logits, cam


class ViTWithRollout(nn.Module):
    """ViT timm : sorties = (logits, Attention Rollout sur la grille de patchs, ex. 14x14)."""

    def __init__(self, model):
        super().__init__()
        self.m = model
        self.maps = []
        self.capture = False
        for blk in model.blocks:
            blk.attn.fused_attn = False  # expose la matrice d'attention
            blk.attn.attn_drop.register_forward_hook(self._store)
        self.grid = tuple(model.patch_embed.grid_size)
        self.n_prefix = model.num_prefix_tokens

    def _store(self, module, inputs, output):
        if self.capture:
            self.maps.append(output)

    def forward(self, x):
        self.maps.clear()
        self.capture = True
        logits = self.m(x)
        self.capture = False
        result = None
        for a in self.maps:
            a = a.mean(1)  # moyenne des têtes : (B, N, N)
            a = a + torch.eye(a.shape[-1], dtype=a.dtype, device=a.device)  # connexion résiduelle
            a = a / a.sum(-1, keepdim=True)
            result = a if result is None else a @ result
        mask = result[:, 0, self.n_prefix :]  # token CLS -> patchs
        return logits, mask.reshape(-1, *self.grid)


def make_wrapper(model):
    is_vit = hasattr(model, "blocks") and hasattr(model, "patch_embed")
    return (ViTWithRollout(model) if is_vit else CNNWithCAM(model)).eval()


def _bench(fn, n=50, warmup=5):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.percentile(times, 50)), float(np.percentile(times, 95))


def export_model(model, size, path, bench=True, quantize=True):
    """Exporte `model` vers `path`, vérifie la parité avec PyTorch, mesure la latence, quantifie en INT8."""
    wrapper = make_wrapper(model)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, size, size)
    torch.onnx.export(
        wrapper,
        dummy,
        str(path),
        input_names=["image"],
        output_names=["logits", "cam"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}, "cam": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(str(path)))

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = torch.randn(2, 3, size, size)
    with torch.no_grad():
        ref_logits, ref_cam = wrapper(x)
    onnx_logits, onnx_cam = sess.run(None, {"image": x.numpy()})
    logit_diff = float(np.abs(ref_logits.numpy() - onnx_logits).max())  # écart absolu
    cam_diff = float(
        np.abs(ref_cam.numpy() - onnx_cam).max() / (np.abs(ref_cam.numpy()).max() + 1e-12)
    )  # relatif
    max_diff = max(logit_diff, cam_diff)
    assert max_diff < 1e-3, f"Écart PyTorch/ONNX trop grand : logits {logit_diff}, carte {cam_diff}"

    metrics = {"parity_max_diff": round(max_diff, 6), "onnx_size_mb": round(path.stat().st_size / 1e6, 2)}
    if bench:
        one = x[:1]
        with torch.no_grad():
            tp50, tp95 = _bench(lambda: wrapper(one))
        op50, op95 = _bench(lambda: sess.run(None, {"image": one.numpy()}))
        metrics.update(
            torch_cpu_p50_ms=round(tp50, 1),
            torch_cpu_p95_ms=round(tp95, 1),
            onnx_cpu_p50_ms=round(op50, 1),
            onnx_cpu_p95_ms=round(op95, 1),
        )
    if quantize:
        metrics.update(quantize_int8(path, path.with_name(path.stem + "-int8.onnx"), size) or {})
    return metrics


def write_metadata(path, architecture, size, extra=None):
    sha = git_sha()
    meta = {
        "name": "DeepGuard",
        "architecture": architecture,
        "classes": CLASSES,
        "input_size": size,
        "mean": list(MEAN),
        "std": list(STD),
        "git_sha": sha,
        "version": f"{architecture}-{sha}",
        "demo": False,
        **(extra or {}),
    }
    Path(path).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main():
    params = load_params()
    t = params["train"]
    size = t["img_size"]

    model = build_model(t["model"], pretrained=False)
    model.load_state_dict(torch.load("checkpoints/best.pt", map_location="cpu"))
    model.eval()

    out = Path("models")
    metrics = export_model(model, size, out / "deepguard.onnx")
    Path("metrics").mkdir(exist_ok=True)
    Path("metrics/export.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    train_file, test_file = Path("metrics/train.json"), Path("metrics/test.json")
    train_info = json.loads(train_file.read_text(encoding="utf-8")) if train_file.exists() else {}
    test_info = json.loads(test_file.read_text(encoding="utf-8")) if test_file.exists() else {}
    write_metadata(
        out / "metadata.json",
        t["model"],
        size,
        {
            "mlflow_run_id": train_info.get("run_id"),
            "validation_metrics": {k: v for k, v in train_info.items() if k.startswith("val_")},
            "metrics": test_info,
            "export": metrics,
        },
    )
    print(metrics)


if __name__ == "__main__":
    main()
