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
from src.utils.metrics import binary_log_loss, compute_metrics, predict


def main():
    params = load_params()
    t = params["train"]
    torch.manual_seed(params["seed"])
    np.random.seed(params["seed"])
    device = get_device()
    cuda = device.type == "cuda"

    def make_loader(split, train):
        ds = FaceDataset(
            f"data/processed/{split}.csv",
            params["data_dir"],
            get_transform(t["img_size"], train, t.get("augment", "basic")),
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
    run_name = f"{t['model']}-{t.get('augment', 'basic')}-s{params['seed']}"
    with mlflow.start_run(run_name=run_name) as run:
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
        # Historique pour les courbes d'apprentissage (src/analysis/learning_curves.py)
        history = {"model": t["model"], "augment": t.get("augment", "basic"), "seed": params["seed"]}
        history.update({"steps": [], "epochs": []})
        log_every = max(1, len(train_dl) // 10)  # 10 points de perte d'entraînement par époque

        for epoch in range(t["epochs"]):
            model.train()
            running, window = 0.0, []
            for step, (x, y) in enumerate(train_dl, start=1):
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device.type, enabled=cuda):
                    loss = loss_fn(model(x), y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
                running += loss.item() * len(y)
                window.append(loss.item())
                if step % log_every == 0:
                    history["steps"].append(
                        {"epoch": epoch + step / len(train_dl), "train_loss": float(np.mean(window))}
                    )
                    window = []

            y_val, p_val = predict(model, val_dl, device)
            m = compute_metrics(y_val, p_val)
            train_loss, val_loss = running / len(train_dl.dataset), binary_log_loss(y_val, p_val)
            mlflow.log_metrics(
                {"train_loss": train_loss, "val_loss": val_loss, **{f"val_{k}": v for k, v in m.items()}},
                step=epoch,
            )
            history["epochs"].append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    **{f"val_{k}": float(v) for k, v in m.items()},
                }
            )
            Path("metrics/history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            print(f"epoch {epoch + 1}/{t['epochs']}  val_f1={m['f1']:.4f}  val_auc={m['roc_auc']:.4f}")
            if m["f1"] > best["f1"]:
                best = m
                torch.save(model.state_dict(), "checkpoints/best.pt")

        elapsed = time.time() - start
        mlflow.log_metric("train_time_s", elapsed)
        summary = {
            "model": t["model"],
            "augment": t.get("augment", "basic"),
            "seed": params["seed"],
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
