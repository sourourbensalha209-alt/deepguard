"""Inférence par lots et métriques de classification (le positif est la classe « fake »)."""

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


@torch.no_grad()
def predict(model, loader, device):
    """Retourne (labels, P(fake)). Classe 0 = fake."""
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            logits = model(x.to(device, non_blocking=True))
        ps.append(torch.softmax(logits.float(), dim=1)[:, 0].cpu())
        ys.append(torch.as_tensor(y))
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def compute_metrics(y, p_fake):
    y_fake = (np.asarray(y) == 0).astype(int)
    pred = (np.asarray(p_fake) >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_fake, pred),
        "f1": f1_score(y_fake, pred),
        "roc_auc": roc_auc_score(y_fake, p_fake),
    }
