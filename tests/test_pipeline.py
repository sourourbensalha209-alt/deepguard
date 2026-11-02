"""Test de bout en bout du pipeline (prepare -> train -> evaluate -> gradcam -> export) sur des
images synthétiques, sur CPU, en quelques secondes. Exécuté par la CI (GitHub Actions)."""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

torch = pytest.importorskip("torch")


def _make_split(root: Path, split: str, n: int, rng):
    for cls, (lo, hi) in {"fake": (10, 70), "real": (170, 230)}.items():
        folder = root / split / cls
        folder.mkdir(parents=True)
        for i in range(n):
            arr = rng.integers(lo, hi, (64, 64, 3), dtype=np.uint8)
            Image.fromarray(arr).save(folder / f"{i}.jpg")


@pytest.fixture()
def workdir(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    data = tmp_path / "raw"
    for split, n in {"train": 16, "valid": 8, "test": 8}.items():
        _make_split(data, split, n, rng)
    params = {
        "seed": 0,
        "data_dir": str(data),
        "external_dir": str(tmp_path / "absent"),
        "prepare": {"train_limit": 0, "val_limit": 0, "test_limit": 0},
        "train": {
            "model": "baseline",
            "epochs": 2,
            "batch_size": 8,
            "lr": 0.001,
            "img_size": 64,
            "num_workers": 0,
        },
    }
    (tmp_path / "params.yaml").write_text(yaml.safe_dump(params), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DATA_DIR", raising=False)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    return tmp_path


def test_full_pipeline(workdir):
    from src.data import prepare
    from src.models import evaluate, export_onnx, register, train
    from src.utils import gradcam

    prepare.main()
    assert len(Path("data/processed/train.csv").read_text().splitlines()) == 33  # 32 images + en-tête

    train.main()
    assert Path("checkpoints/best.pt").exists()
    summary = json.loads(Path("metrics/train.json").read_text())
    assert summary["model"] == "baseline" and 0 <= summary["val_roc_auc"] <= 1

    evaluate.main()
    results = json.loads(Path("metrics/test.json").read_text())
    assert {"test_accuracy", "test_f1", "test_roc_auc"} <= set(results)
    assert Path("reports/confusion_matrix.png").exists() and Path("reports/roc_curve.png").exists()

    gradcam.main(n_per_class=2)
    assert Path("reports/heatmaps.png").exists()

    export_onnx.main()
    meta = json.loads(Path("models/metadata.json").read_text())
    assert meta["architecture"] == "baseline" and meta["demo"] is False
    assert meta["version"].startswith("baseline-") and "val_f1" in meta["validation_metrics"]
    export = json.loads(Path("metrics/export.json").read_text())
    assert export["parity_max_diff"] < 1e-3

    # Contrat avec l'API : entrée « image », sorties « logits » et « cam »
    import onnxruntime as ort

    sess = ort.InferenceSession("models/deepguard.onnx", providers=["CPUExecutionProvider"])
    assert [i.name for i in sess.get_inputs()] == ["image"]
    logits, cam = sess.run(None, {"image": np.zeros((2, 3, 64, 64), dtype=np.float32)})
    assert logits.shape == (2, 2) and cam.shape[0] == 2 and cam.ndim == 3

    # Registre MLflow : première version -> staging + production
    register.main()
    assert json.loads(Path("metrics/registry.json").read_text())["aliases"] == ["staging", "production"]


def test_cnn_cam_matches_forward():
    from src.models.architecture import BaselineCNN
    from src.utils.gradcam import cnn_cam

    model = BaselineCNN().eval()
    x = torch.randn(2, 3, 64, 64)
    logits, cam = cnn_cam(model, x)
    assert torch.allclose(logits, model(x), atol=1e-6)
    assert cam.shape == (2, 4, 4) and bool((cam >= 0).all())


def test_vit_rollout_shape():
    timm = pytest.importorskip("timm")
    from src.utils.gradcam import vit_rollout

    model = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=2).eval()
    logits, mask = vit_rollout(model, torch.randn(1, 3, 224, 224))
    assert logits.shape == (1, 2)
    assert mask.shape == (14, 14)
    assert float(mask.min()) >= 0 and float(mask.max()) <= 1


def test_vit_onnx_export_matches_rollout(tmp_path):
    """Le ViT est exporté AVEC son Attention Rollout : l'API peut l'afficher sans PyTorch."""
    timm = pytest.importorskip("timm")
    import onnxruntime as ort

    from src.models.architecture import describe_vit
    from src.models.export_onnx import export_model
    from src.utils.gradcam import vit_rollout

    model = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=2).eval()
    info = describe_vit(model)
    assert info["vit_patch_size"] == 16 and info["vit_grid"] == "14x14" and info["vit_num_patches"] == 196
    assert info["vit_pos_embed_shape"] == "(1, 197, 192)"  # 1 token CLS + 196 patchs

    path = tmp_path / "vit.onnx"
    metrics = export_model(model, 224, path, bench=False, quantize=True)
    assert metrics["parity_max_diff"] < 1e-3

    x = torch.randn(1, 3, 224, 224)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    logits, cam = sess.run(None, {"image": x.numpy()})
    assert logits.shape == (1, 2) and cam.shape == (1, 14, 14)
    _, mask = vit_rollout(model, x)
    c = cam[0]
    c = (c - c.min()) / (c.max() - c.min() + 1e-8)
    assert np.allclose(c, mask.numpy(), atol=1e-3)


def test_metrics_positive_class_is_fake():
    from src.utils.metrics import compute_metrics

    y = np.array([0, 0, 1, 1])  # 0 = fake, 1 = real
    p_fake = np.array([0.9, 0.8, 0.2, 0.1])
    m = compute_metrics(y, p_fake)
    assert m["accuracy"] == 1.0 and m["f1"] == 1.0 and m["roc_auc"] == 1.0
