"""Inférence par lots et métriques de classification (le positif est la classe « fake »)."""

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score


@torch.no_grad()
def predict_margin(model, loader, device):
    """Retourne (labels, marge) avec marge = logit(fake) - logit(real).

    P(fake) = sigmoid(marge). Garder la marge (et pas seulement la probabilité) évite que les
    probabilités saturées à 0 ou 1 en float32 rendent la calibration impossible.
    """
    model.eval()
    ys, zs = [], []
    for x, y in loader:
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            logits = model(x.to(device, non_blocking=True))
        logits = logits.float()
        zs.append((logits[:, 0] - logits[:, 1]).cpu())
        ys.append(torch.as_tensor(y))
    return torch.cat(ys).numpy(), torch.cat(zs).double().numpy()


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


@torch.no_grad()
def predict(model, loader, device):
    """Retourne (labels, P(fake)). Classe 0 = fake."""
    y, z = predict_margin(model, loader, device)
    return y, sigmoid(z)


def compute_metrics(y, p_fake):
    y_fake = (np.asarray(y) == 0).astype(int)
    pred = (np.asarray(p_fake) >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_fake, pred),
        "f1": f1_score(y_fake, pred),
        "roc_auc": roc_auc_score(y_fake, p_fake),
    }


def binary_log_loss(y, p_fake):
    """Entropie croisée (perte de validation), classe positive « fake »."""
    y_fake = (np.asarray(y) == 0).astype(int)
    p = np.clip(np.asarray(p_fake, dtype=np.float64), 1e-7, 1 - 1e-7)
    return float(log_loss(y_fake, p, labels=[0, 1]))
