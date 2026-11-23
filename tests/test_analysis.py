"""Tests des analyses du rapport (src/analysis) sur des prédictions synthétiques : sans GPU ni PyTorch."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image


def _fake_run(folder: Path, n=400, noise=1.0, seed=0, epochs=3):
    """Crée un dossier reports/runs/<tag>_s<graine>/ réaliste : marges 2 fois trop confiantes."""
    rng = np.random.default_rng(seed)
    folder.mkdir(parents=True)
    for split in ("val", "test"):
        y = np.array([0, 1] * (n // 2))  # 0 = fake
        signal = np.where(y == 0, 1.0, -1.0) + rng.normal(0, noise, n)  # marge "vraie"
        z = 2 * (2 / noise**2) * signal  # log-rapport de vraisemblance exact = (2 / noise²) * signal
        pd.DataFrame(
            {
                "path": [f"img_{i}.jpg" for i in range(n)],
                "label": y,
                "margin": z,
                "p_fake": 1 / (1 + np.exp(-z)),
            }
        ).to_csv(folder / f"preds_{split}.csv", index=False)
    hist = {
        "model": folder.name,
        "steps": [{"epoch": e / 10, "train_loss": 0.7 / (1 + e)} for e in range(1, 10 * epochs + 1)],
        "epochs": [
            {
                "epoch": e,
                "train_loss": 0.5 / e,
                "val_loss": 0.4 / e,
                "val_f1": 0.8 + 0.05 * e,
                "val_accuracy": 0.8,
                "val_roc_auc": 0.9,
            }
            for e in range(1, epochs + 1)
        ],
    }
    (folder / "history.json").write_text(json.dumps(hist))
    (folder / "test.json").write_text(
        json.dumps({"test_accuracy": 0.9 + seed / 100, "test_f1": 0.9, "test_roc_auc": 0.95})
    )


@pytest.fixture()
def runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "reports" / "runs"
    _fake_run(root / "efficientnet_s42", noise=1.2, seed=1)
    _fake_run(root / "vit_s42", noise=0.5, seed=2)
    _fake_run(root / "vit_s1", noise=0.5, seed=3)
    return tmp_path


def test_bootstrap_ci_contains_point():
    from src.analysis.stats import bootstrap_ci

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = np.clip(np.where(y == 0, 0.8, 0.2) + rng.normal(0, 0.25, 500), 0, 1)
    ci = bootstrap_ci(y, p, n_boot=300)
    for m in ci.values():
        assert m["low"] <= m["value"] <= m["high"]
        assert m["high"] - m["low"] < 0.2


def test_mcnemar_detects_difference():
    from src.analysis.stats import mcnemar

    y = np.zeros(200, dtype=int)  # tout est fake
    good = np.full(200, 0.9)
    bad = good.copy()
    bad[:40] = 0.1  # B se trompe sur 40 images
    res = mcnemar(y, good, bad)
    assert res["only_a_correct"] == 40 and res["only_b_correct"] == 0
    assert res["p_value"] < 1e-6
    assert mcnemar(y, good, good)["p_value"] == 1.0


def test_temperature_scaling_fixes_overconfidence():
    from src.analysis.calibration import ece, fit_temperature

    rng = np.random.default_rng(0)
    n = 4000
    y_fake = rng.integers(0, 2, n)
    true_z = np.where(y_fake == 1, 1.5, -1.5) + rng.normal(0, 1.5, n)
    z = 5 * true_z  # modèle trop confiant d'un facteur 5
    T = fit_temperature(y_fake, z)
    assert 3.5 < T < 6.5
    before, after = ece(y_fake, 1 / (1 + np.exp(-z))), ece(y_fake, 1 / (1 + np.exp(-z / T)))
    assert after < before


def test_analysis_scripts_end_to_end(runs):
    from src.analysis import calibration, learning_curves, stats

    stats.main(n_boot=200)
    s = json.loads(Path("metrics/stats.json").read_text())
    assert set(s["bootstrap"]) == {"efficientnet", "vit"}
    assert "vit_vs_efficientnet" in s["mcnemar"]
    assert s["seeds"]["vit"]["seeds"] == [1, 42]

    calibration.main()
    c = json.loads(Path("metrics/calibration.json").read_text())
    assert c["vit"]["temperature"] > 1 and c["vit"]["accuracy_unchanged"]

    learning_curves.main()
    for stem in ("stats", "calibration", "learning_curves"):
        assert Path(f"docs/rapport/generated/{stem}.tex").exists()
    for fig in ("calibration", "learning_curves"):
        assert Path(f"docs/rapport/figures/{fig}.png").exists()


def test_robustness_report(runs):
    from src.analysis import robustness

    for tag, acc in (("vit", 0.6), ("vit_scale", 0.9)):
        res = {
            n: {"accuracy": 0.99 if n == "original" else acc, "fake_recall": acc}
            for n in robustness.PERTURBATIONS
        }
        out = Path(f"reports/runs/{tag}_s42")
        out.mkdir(parents=True, exist_ok=True)
        (out / "robustness.json").write_text(json.dumps({"n_images": 100, "results": res}))
    robustness.report()
    tex = Path("docs/rapport/generated/robustness.tex").read_text(encoding="utf-8")
    assert "améliore" in tex


@pytest.mark.parametrize(
    "name", ["JPEG q20", "résolution 32 px", "recadrage 60 %", "dézoom 80 %", "décalage 10 %"]
)
def test_perturbations_keep_rgb_image(name):
    from src.analysis.robustness import PERTURBATIONS

    img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8))
    out = PERTURBATIONS[name](img)
    assert out.mode == "RGB" and min(out.size) > 0
