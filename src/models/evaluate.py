"""Évaluation sur le test + généralisation sur un 2e dataset : python -m src.models.evaluate"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from torchvision import datasets  # noqa: E402

import mlflow  # noqa: E402
from src.data.dataset import FaceDataset, get_transform  # noqa: E402
from src.models.architecture import build_model  # noqa: E402
from src.utils.config import get_device, load_params  # noqa: E402
from src.utils.metrics import compute_metrics, predict  # noqa: E402


def main():
    params = load_params()
    t = params["train"]
    device = get_device()
    model = build_model(t["model"], pretrained=False)
    model.load_state_dict(torch.load("checkpoints/best.pt", map_location="cpu"))
    model.to(device)
    tf = get_transform(t["img_size"])

    def loader(ds):
        return DataLoader(ds, batch_size=t["batch_size"], num_workers=t["num_workers"])

    test_ds = FaceDataset("data/processed/test.csv", params["data_dir"], tf)
    y, p_fake = predict(model, loader(test_ds), device)
    results = {f"test_{k}": v for k, v in compute_metrics(y, p_fake).items()}

    Path("reports").mkdir(exist_ok=True)
    Path("metrics").mkdir(exist_ok=True)
    y_fake = (y == 0).astype(int)
    ConfusionMatrixDisplay.from_predictions(
        y_fake, (p_fake >= 0.5).astype(int), display_labels=["real", "fake"]
    )
    plt.savefig("reports/confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    RocCurveDisplay.from_predictions(y_fake, p_fake)
    plt.savefig("reports/roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Généralisation : 2e dataset (dossiers training_fake / training_real -> classes 0 / 1)
    ext = Path(params["external_dir"])
    if ext.exists():
        y_ext, p_ext = predict(model, loader(datasets.ImageFolder(ext, tf)), device)
        results.update({f"external_{k}": v for k, v in compute_metrics(y_ext, p_ext).items()})
    else:
        print(f"[info] external_dir introuvable ({ext}) : test de généralisation ignoré.")

    results = {k: round(float(v), 4) for k, v in results.items()}
    Path("metrics/test.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    mlflow.set_experiment("deepguard")
    with mlflow.start_run(run_name=f"eval-{t['model']}"):
        mlflow.log_metrics(results)
        mlflow.log_artifact("reports/confusion_matrix.png")
        mlflow.log_artifact("reports/roc_curve.png")
    print(results)


if __name__ == "__main__":
    main()
