"""Pipeline de données PyTorch : transformations et Dataset lu depuis des CSV.

Pas de fuite de données : le split train / valid / test est fourni par le dataset, la normalisation
est celle d'ImageNet (statistiques fixes) et l'augmentation n'est appliquée qu'au train.
"""

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

CLASSES = ["fake", "real"]  # index 0 = fake, 1 = real (ordre alphabétique, comme ImageFolder)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def get_transform(size, train=False):
    ops = [transforms.Resize((size, size))]
    if train:
        ops += [transforms.RandomHorizontalFlip(), transforms.ColorJitter(0.1, 0.1, 0.1)]
    ops += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(ops)


class FaceDataset(Dataset):
    """Colonnes du CSV : path (relatif à `root`), label (0 = fake, 1 = real)."""

    def __init__(self, csv_path, root, transform):
        df = pd.read_csv(csv_path)
        self.paths = df["path"].tolist()
        self.labels = df["label"].tolist()
        self.root = Path(root)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.root / self.paths[i]).convert("RGB")
        return self.transform(img), self.labels[i]
