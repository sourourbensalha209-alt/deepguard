"""Construit les listes de fichiers train/val/test (CSV) : python -m src.data.prepare"""

import random
from pathlib import Path

import pandas as pd

from src.data.dataset import CLASSES
from src.utils.config import load_params

SPLITS = {"train": "train", "val": "valid", "test": "test"}  # nom logique -> dossier du dataset


def main():
    params = load_params()
    root = Path(params["data_dir"])
    rng = random.Random(params["seed"])
    out = Path("data/processed")
    out.mkdir(parents=True, exist_ok=True)

    for name, folder in SPLITS.items():
        limit = params["prepare"][f"{name}_limit"]
        rows = []
        for label, cls in enumerate(CLASSES):
            files = sorted((root / folder / cls).glob("*.jpg"))
            if not files:
                raise SystemExit(f"Aucune image trouvée dans {root / folder / cls} : vérifie data_dir.")
            if limit:
                files = rng.sample(files, min(len(files), limit // 2))  # classes équilibrées
            rows += [(f.relative_to(root).as_posix(), label) for f in files]
        rng.shuffle(rows)
        pd.DataFrame(rows, columns=["path", "label"]).to_csv(out / f"{name}.csv", index=False)
        print(f"{name}: {len(rows)} images")


if __name__ == "__main__":
    main()
