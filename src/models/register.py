"""Enregistre le modèle exporté dans le MLflow Model Registry : python -m src.models.register [--promote]

Chaque exécution crée une nouvelle version du modèle « deepguard » avec l'alias `staging`.
Elle reçoit en plus l'alias `production` si elle est au moins aussi bonne (F1 en validation) que la version
actuellement en production, ou si --promote est passé.
(Les « stages » MLflow Staging / Production sont dépréciés au profit des alias.)

Torch-free : n'utilise que mlflow, onnx et les fichiers JSON produits par le pipeline.
"""

import argparse
import json
from pathlib import Path

import onnx
from mlflow.exceptions import MlflowException

import mlflow
from mlflow import MlflowClient

MODEL_NAME = "deepguard"


def _read(path):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def main(promote=False):
    meta = _read("models/metadata.json")
    train = _read("metrics/train.json")
    test = _read("metrics/test.json")
    val_f1 = float(train.get("val_f1", 0.0))

    numeric = {k: v for k, v in {**train, **test}.items() if isinstance(v, int | float)}
    mlflow.set_experiment("deepguard")
    with mlflow.start_run(run_name=f"register-{meta.get('architecture', 'model')}"):
        mlflow.set_tags({"git_sha": meta.get("git_sha", "unknown"), "source_run_id": train.get("run_id", "")})
        mlflow.log_metrics(numeric)
        info = mlflow.onnx.log_model(
            onnx.load("models/deepguard.onnx"), name="model", registered_model_name=MODEL_NAME
        )

    client = MlflowClient()
    version = str(info.registered_model_version)
    for key, value in {
        "val_f1": val_f1,
        "architecture": meta.get("architecture", "inconnue"),
        "git_sha": meta.get("git_sha", "unknown"),
        "version": meta.get("version", ""),
    }.items():
        client.set_model_version_tag(MODEL_NAME, version, key, str(value))

    client.set_registered_model_alias(MODEL_NAME, "staging", version)
    aliases = ["staging"]
    try:
        current = client.get_model_version_by_alias(MODEL_NAME, "production")
        current_f1 = float(current.tags.get("val_f1", -1))
    except MlflowException:
        current_f1 = -1.0
    if promote or val_f1 >= current_f1:
        client.set_registered_model_alias(MODEL_NAME, "production", version)
        aliases.append("production")

    result = {"registered_model": MODEL_NAME, "version": int(version), "aliases": aliases, "val_f1": val_f1}
    Path("metrics").mkdir(exist_ok=True)
    Path("metrics/registry.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--promote", action="store_true", help="forcer l'alias production")
    main(parser.parse_args().promote)
