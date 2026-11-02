"""MLflow Model Registry : versions, alias staging / production (sans PyTorch)."""

import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("mlflow")
from mlflow import MlflowClient  # noqa: E402
from src.models import register  # noqa: E402

DEMO = Path(__file__).resolve().parent.parent / "backend" / "demo_model" / "deepguard-demo.onnx"


@pytest.fixture()
def workdir(tmp_path, monkeypatch):
    (tmp_path / "models").mkdir()
    (tmp_path / "metrics").mkdir()
    shutil.copy(DEMO, tmp_path / "models" / "deepguard.onnx")
    (tmp_path / "models" / "metadata.json").write_text(
        json.dumps({"architecture": "baseline", "git_sha": "abc1234", "version": "baseline-abc1234"})
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    return tmp_path


def _train(f1):
    Path("metrics/train.json").write_text(json.dumps({"model": "baseline", "run_id": "r1", "val_f1": f1}))


def test_registry_promotion_rules(workdir):
    client = MlflowClient()

    _train(0.90)  # première version : staging ET production
    register.main()
    assert json.loads(Path("metrics/registry.json").read_text())["aliases"] == ["staging", "production"]
    assert int(client.get_model_version_by_alias("deepguard", "production").version) == 1

    _train(0.80)  # moins bonne : staging seulement, production inchangée
    register.main()
    assert json.loads(Path("metrics/registry.json").read_text())["aliases"] == ["staging"]
    assert int(client.get_model_version_by_alias("deepguard", "staging").version) == 2
    assert int(client.get_model_version_by_alias("deepguard", "production").version) == 1

    register.main(promote=True)  # promotion forcée
    assert int(client.get_model_version_by_alias("deepguard", "production").version) == 3
    assert client.get_model_version_by_alias("deepguard", "production").tags["architecture"] == "baseline"
