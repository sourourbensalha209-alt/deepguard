"""Lecture des résultats d'entraînement rangés par le notebook dans reports/runs/<tag>_s<graine>/.

Chaque dossier contient : train.json, history.json, test.json, preds_val.csv, preds_test.csv
(copiés après `python -m src.models.train` puis `python -m src.models.evaluate`).
Exemple : reports/runs/vit_s42/, reports/runs/vit_scale_s42/, reports/runs/efficientnet_s1/.
"""

import json
import re
from pathlib import Path

import pandas as pd

RUNS = Path("reports/runs")
MAIN_SEED = 42
ORDER = ["baseline", "efficientnet", "vit", "vit_scale"]


def list_runs(root=RUNS):
    """Retourne {tag: {graine: dossier}}."""
    runs = {}
    for d in sorted(Path(root).glob("*_s*")):
        m = re.fullmatch(r"(.+)_s(\d+)", d.name)
        if m and d.is_dir():
            runs.setdefault(m.group(1), {})[int(m.group(2))] = d
    return dict(sorted(runs.items(), key=lambda kv: ORDER.index(kv[0]) if kv[0] in ORDER else 99))


def main_run(runs, tag):
    seeds = runs[tag]
    return seeds.get(MAIN_SEED) or seeds[min(seeds)]


def load_preds(folder, split="test"):
    return pd.read_csv(Path(folder) / f"preds_{split}.csv")


def load_json(folder, name):
    path = Path(folder) / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
