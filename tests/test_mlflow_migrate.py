"""Migration de runs MLflow d'un magasin vers un autre (sans PyTorch)."""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("mlflow")
from mlflow.entities import Metric  # noqa: E402

from mlflow import MlflowClient  # noqa: E402

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "mlflow_migrate.py"
spec = importlib.util.spec_from_file_location("mlflow_migrate", SCRIPT)
mlflow_migrate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mlflow_migrate)


@pytest.fixture()
def stores(tmp_path):
    return f"sqlite:///{tmp_path / 'source.db'}", f"sqlite:///{tmp_path / 'dest.db'}"


def _make_source(uri):
    client = MlflowClient(tracking_uri=uri)
    exp = client.create_experiment("deepguard")
    for name, f1 in [("baseline", 0.84), ("vit", 0.998)]:
        run = client.create_run(exp, run_name=name, tags={"git_sha": "abc1234"})
        client.log_batch(
            run.info.run_id,
            params=[mlflow_migrate.Param("model", name), mlflow_migrate.Param("lr", "0.001")],
            metrics=[Metric("val_f1", f1 * (0.9 + 0.05 * s), 1_700_000_000_000 + s, s) for s in range(3)],
        )
        client.set_terminated(run.info.run_id, status="FINISHED")


def test_runs_are_copied_with_params_metrics_and_history(stores):
    source, dest = stores
    _make_source(source)

    assert mlflow_migrate.migrate(source, dest) == {"deepguard": 2}

    client = MlflowClient(tracking_uri=dest)
    exp = client.get_experiment_by_name("deepguard")
    runs = {r.info.run_name: r for r in client.search_runs([exp.experiment_id])}
    assert set(runs) == {"baseline", "vit"}
    vit = runs["vit"]
    assert vit.data.params["model"] == "vit" and vit.data.tags["git_sha"] == "abc1234"
    assert vit.info.status == "FINISHED"
    history = client.get_metric_history(vit.info.run_id, "val_f1")
    assert [m.step for m in history] == [
        0,
        1,
        2,
    ]  # l'historique par pas est conservé, pas seulement la dernière valeur


def test_migration_is_idempotent(stores):
    source, dest = stores
    _make_source(source)
    mlflow_migrate.migrate(source, dest)
    assert mlflow_migrate.migrate(source, dest) == {"deepguard": 0}  # relancer ne crée pas de doublons

    client = MlflowClient(tracking_uri=dest)
    exp = client.get_experiment_by_name("deepguard")
    assert len(client.search_runs([exp.experiment_id])) == 2
