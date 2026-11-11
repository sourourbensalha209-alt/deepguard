"""Copie les runs MLflow d'un magasin vers un autre.

Cas d'usage : récupérer les runs d'entraînement faits sur Kaggle (fichier mlflow.db) pour les afficher dans
une interface MLflow locale propre, ou les envoyer sur le serveur MLflow de DagsHub.

    python scripts/mlflow_migrate.py --source sqlite:///kaggle_mlflow.db --dest sqlite:///mlflow-data/mlflow.db
    python scripts/mlflow_migrate.py --source sqlite:///kaggle_mlflow.db \
        --dest https://dagshub.com/<utilisateur>/<depot>.mlflow
        (identifiants DagsHub dans les variables MLFLOW_TRACKING_USERNAME et MLFLOW_TRACKING_PASSWORD)

Copie : expériences (par nom), runs (nom, statut, dates), paramètres, métriques (avec pas et horodatage)
et tags. Ne copie ni les artefacts (fichiers) ni le registre de modèles. Les runs déjà copiés sont reconnus
(tag `migrated_from_run_id`) et ignorés : le script peut être relancé sans créer de doublons.
"""

import argparse

from mlflow.entities import Metric, Param, RunTag

from mlflow import MlflowClient

CHUNK = 100


def _batches(items, size=CHUNK):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def migrate(source, dest, only_experiment=None):
    """Retourne le nombre de runs copiés par expérience."""
    src, dst = MlflowClient(tracking_uri=source), MlflowClient(tracking_uri=dest)
    copied = {}
    for exp in src.search_experiments():
        if only_experiment and exp.name != only_experiment:
            continue
        runs = src.search_runs([exp.experiment_id], max_results=1000)
        if not runs:
            continue
        target = dst.get_experiment_by_name(exp.name)
        target_id = target.experiment_id if target else dst.create_experiment(exp.name)
        already = {
            r.data.tags.get("migrated_from_run_id") for r in dst.search_runs([target_id], max_results=1000)
        }

        copied[exp.name] = 0
        for run in sorted(runs, key=lambda r: r.info.start_time):
            if run.info.run_id in already:
                continue
            tags = {k: v for k, v in run.data.tags.items() if not k.startswith("mlflow.")}
            tags["migrated_from_run_id"] = run.info.run_id
            new = dst.create_run(
                target_id, start_time=run.info.start_time, tags=tags, run_name=run.info.run_name
            )
            run_id = new.info.run_id
            for chunk in _batches([Param(k, str(v)) for k, v in run.data.params.items()]):
                dst.log_batch(run_id, params=chunk)
            metrics = []
            for key in run.data.metrics:
                metrics += [
                    Metric(key, m.value, m.timestamp, m.step)
                    for m in src.get_metric_history(run.info.run_id, key)
                ]
            for chunk in _batches(metrics, 500):
                dst.log_batch(run_id, metrics=chunk)
            for chunk in _batches([RunTag(k, str(v)) for k, v in tags.items()]):
                dst.log_batch(run_id, tags=chunk)
            dst.set_terminated(run_id, status=run.info.status, end_time=run.info.end_time)
            copied[exp.name] += 1
    return copied


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--source", required=True, help="URI du magasin source (ex. sqlite:///kaggle_mlflow.db)"
    )
    parser.add_argument(
        "--dest", required=True, help="URI du magasin cible (sqlite:/// ou https://dagshub.com/...)"
    )
    parser.add_argument("--experiment", help="ne copier que cette expérience (par défaut : toutes)")
    args = parser.parse_args()
    result = migrate(args.source, args.dest, args.experiment)
    for name, n in result.items():
        print(f"expérience « {name} » : {n} run(s) copié(s)")
    if not result:
        print("Aucun run trouvé dans la source.")


if __name__ == "__main__":
    main()
