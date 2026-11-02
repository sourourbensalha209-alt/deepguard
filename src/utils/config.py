"""Paramètres du projet (params.yaml), version Git, périphérique de calcul."""

import os
import subprocess

import torch
import yaml


def load_params(path="params.yaml"):
    with open(path, encoding="utf-8") as f:
        params = yaml.safe_load(f)
    params["data_dir"] = os.getenv("DATA_DIR", params["data_dir"])  # surcharge possible (Kaggle)
    return params


def git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
