"""Entraînement : python -m src.models.train  (hyperparamètres dans params.yaml)."""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import mlflow
from src.data.dataset import FaceDataset, get_transform
from src.models.architecture import build_model, describe_vit
from src.utils.config import get_device, git_sha, load_params
from src.utils.metrics import compute_metrics, predict


def main():
    params = load_params()
    t = params["train"]
    torch.manual_seed(params["seed"])
    np.random.seed(params["seed"])
    device = get_device()
    cuda = device.type == "cuda"

    def make_loader(split, train):
        ds = FaceDataset(
            f"data/processed/{split}.csv", params["data_dir"], get_transform(t["img_size"], train)
        )
        return DataLoader(
            ds,
            batch_size=t["batch_size"],
            shuffle=train,
            num_workers=t["num_workers"],
            pin_memory=cuda,
        )

    train_dl, val_dl = make_loader("train", True), make_loader("val", False)
    model = build_model(t["model"]).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t["epochs"] * len(train_dl))
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)
    loss_fn = nn.CrossEntropyLoss()
    Path("checkpoints").mkdir(exist_ok=True)
    Path("metrics").mkdir(exist_ok=True)

    mlflow.set_experiment("deepguard")
    with mlflow.start_run(run_name=t["model"]) as run:
        mlflow.set_tag("git_sha", git_sha())
        mlflow.log_params(
            {
                **t,
                "seed": params["seed"],
                "n_params_m": round(n_params, 2),
                **{f"prepare_{k}": v for k, v in params["prepare"].items()},
                **describe_vit(model),
            }
        )
        best, start = {"f1": -1.0}, time.time()

        for epoch in range(t["epochs"]):
            model.train()
            running = 0.0
            for x, y in train_dl:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device.type, enabled=cuda):
                    loss = loss_fn(model(x), y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
                running += loss.item() * len(y)

            m = compute_metrics(*predict(model, val_dl, device))
            mlflow.log_metrics(
                {"train_loss": running / len(train_dl.dataset), **{f"val_{k}": v for k, v in m.items()}},
                step=epoch,
            )
            print(f"epoch {epoch + 1}/{t['epochs']}  val_f1={m['f1']:.4f}  val_auc={m['roc_auc']:.4f}")
            if m["f1"] > best["f1"]:
                best = m
                torch.save(model.state_dict(), "checkpoints/best.pt")

        elapsed = time.time() - start
        mlflow.log_metric("train_time_s", elapsed)
        summary = {
            "model": t["model"],
            "run_id": run.info.run_id,
            "git_sha": git_sha(),
            "n_params_m": round(n_params, 2),
            "train_time_s": round(elapsed),
            **{f"val_{k}": round(float(v), 4) for k, v in best.items()},
        }
        Path("metrics/train.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(summary)


if __name__ == "__main__":
    main()
