"""Copie 4 images du jeu de test (2 réelles, 2 générées) dans frontend/public/samples/.

Usage : python scripts/export_samples.py   (après `python -m src.data.prepare`)
Les images viennent du dataset (FFHQ / StyleGAN) : aucune photo de personne réelle n'est ajoutée au dépôt
sans que tu l'aies choisie.
"""

import shutil
from pathlib import Path

import pandas as pd

from src.utils.config import load_params

OUT = Path("frontend/public/samples")


def main():
    params = load_params()
    root = Path(params["data_dir"])
    df = pd.read_csv("data/processed/test.csv")
    picks = [("réelle", df[df.label == 1].head(2)), ("générée", df[df.label == 0].head(2))]
    OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    for kind, rows in picks:
        for _, row in rows.iterrows():
            n += 1
            shutil.copy(root / row["path"], OUT / f"sample_{n}.jpg")
            print(f"sample_{n}.jpg : image {kind} ({row['path']})")
    print(f"{n} images copiées dans {OUT}")


if __name__ == "__main__":
    main()
