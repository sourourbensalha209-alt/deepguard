"""Télécharge depuis le MLflow Model Registry le modèle portant un alias (production par défaut) vers models/.

python scripts/pull_registry_model.py [alias]     (MLFLOW_TRACKING_URI doit pointer vers le serveur)
"""

import json
import shutil
import sys
from pathlib import Path

import mlflow
from mlflow import MlflowClient

NAME = "deepguard"


def main(alias="production"):
    client = MlflowClient()
    mv = client.get_model_version_by_alias(NAME, alias)
    tmp = Path("models/_download")
    shutil.rmtree(tmp, ignore_errors=True)
    mlflow.artifacts.download_artifacts(artifact_uri=f"models:/{NAME}@{alias}", dst_path=str(tmp))
    onnx_files = sorted(tmp.rglob("*.onnx"))
    if not onnx_files:
        raise SystemExit("Aucun fichier .onnx trouvé dans le modèle téléchargé.")
    Path("models").mkdir(exist_ok=True)
    shutil.copy(onnx_files[0], "models/deepguard.onnx")
    shutil.rmtree(tmp, ignore_errors=True)

    tags = dict(mv.tags)
    meta = {
        "name": "DeepGuard",
        "architecture": tags.get("architecture", "inconnue"),
        "classes": ["fake", "real"],
        "git_sha": tags.get("git_sha", "unknown"),
        "version": tags.get("version") or f"registry-v{mv.version}",
        "registry": {"name": NAME, "version": int(mv.version), "alias": alias},
        "validation_metrics": {"val_f1": float(tags.get("val_f1", 0))},
        "demo": False,
    }
    Path("models/metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"models/deepguard.onnx : version {mv.version} (alias {alias})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "production")
