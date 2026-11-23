"""Robustesse quantitative : exactitude sous dégradations, sur N images du test (et non une seule).

1) Mesurer un modèle (GPU conseillé) :
   python -m src.analysis.robustness --tag vit --model vit --checkpoint /kaggle/working/vit.pt
   python -m src.analysis.robustness --tag vit_scale --model vit --checkpoint /kaggle/working/vit_scale.pt
   ->  reports/runs/<tag>_s42/robustness.json
2) Comparer tous les modèles mesurés :
   python -m src.analysis.robustness --report
   ->  reports/robustness.png, metrics/robustness.json, docs/rapport/generated/robustness.tex

Le recadrage (zoom avant) et le dézoom testent la dépendance au cadrage FFHQ, que l'augmentation
`augment: scale` (recadrages aléatoires) cherche à réduire.
"""

import argparse
import io
import json
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

from src.analysis import latex
from src.analysis.runs import MAIN_SEED, RUNS, list_runs, load_json, main_run

# ----------------------------------------------------------------- dégradations (PIL -> PIL)


def jpeg(img, quality):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def downscale(img, side):
    w, h = img.size
    return img.resize((side, side), Image.BICUBIC).resize((w, h), Image.BICUBIC)


def blur(img, radius):
    return img.filter(ImageFilter.GaussianBlur(radius))


def noise(img, sigma, seed=0):
    arr = np.asarray(img, dtype=np.float32)
    arr = arr + np.random.default_rng(seed).normal(0, sigma, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype("uint8"))


def center_crop(img, fraction):
    """Zoom avant : garde la partie centrale (le visage paraît plus grand)."""
    w, h = img.size
    cw, ch = round(w * fraction), round(h * fraction)
    left, top = (w - cw) // 2, (h - ch) // 2
    return img.crop((left, top, left + cw, top + ch))


def zoom_out(img, fraction):
    """Dézoom : image réduite et centrée sur un fond de sa couleur moyenne (le visage paraît plus petit)."""
    w, h = img.size
    small = img.resize((round(w * fraction), round(h * fraction)), Image.BICUBIC)
    mean = tuple(int(c) for c in np.asarray(img).reshape(-1, 3).mean(0))
    canvas = Image.new("RGB", (w, h), mean)
    canvas.paste(small, ((w - small.width) // 2, (h - small.height) // 2))
    return canvas


def shift(img, fraction):
    """Décalage : fenêtre de même taille déplacée vers le coin haut-gauche (bords complétés par réflexion)."""
    arr = np.asarray(img)
    h, w = arr.shape[:2]
    dy, dx = round(h * fraction), round(w * fraction)
    padded = np.pad(arr, ((dy, 0), (dx, 0), (0, 0)), mode="reflect")
    return Image.fromarray(padded[:h, :w])


def identity(img):
    return img


PERTURBATIONS = {
    "original": identity,
    "JPEG q50": partial(jpeg, quality=50),
    "JPEG q20": partial(jpeg, quality=20),
    "résolution 64 px": partial(downscale, side=64),
    "résolution 32 px": partial(downscale, side=32),
    "flou r=2": partial(blur, radius=2),
    "bruit σ=15": partial(noise, sigma=15),
    "recadrage 80 %": partial(center_crop, fraction=0.8),
    "recadrage 60 %": partial(center_crop, fraction=0.6),
    "dézoom 80 %": partial(zoom_out, fraction=0.8),
    "décalage 10 %": partial(shift, fraction=0.1),
}


def tex_label(n):
    return n.replace("%", "\\%").replace("σ", "$\\sigma$")


CADRAGE = ["recadrage 80 %", "recadrage 60 %", "dézoom 80 %", "décalage 10 %"]


# ----------------------------------------------------------------- mesure d'un modèle


def evaluate_model(tag, model_name, checkpoint, n_images=1000, batch_size=64, num_workers=2):
    import torch
    from torch.utils.data import DataLoader, Dataset

    from src.data.dataset import get_transform
    from src.models.architecture import build_model
    from src.utils.config import get_device, load_params
    from src.utils.metrics import compute_metrics, predict

    params = load_params()
    size = params["train"]["img_size"]
    tf = get_transform(size)
    df = pd.read_csv("data/processed/test.csv")
    df = pd.concat([df[df.label == 0].head(n_images // 2), df[df.label == 1].head(n_images // 2)])
    root = Path(params["data_dir"])

    class Perturbed(Dataset):
        def __init__(self, fn):
            self.fn = fn

        def __len__(self):
            return len(df)

        def __getitem__(self, i):
            row = df.iloc[i]
            img = Image.open(root / row["path"]).convert("RGB")
            return tf(self.fn(img)), int(row["label"])

    device = get_device()
    model = build_model(model_name, pretrained=False)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.to(device)

    results = {}
    for name, fn in PERTURBATIONS.items():
        loader = DataLoader(Perturbed(fn), batch_size=batch_size, num_workers=num_workers)
        y, p = predict(model, loader, device)
        m = compute_metrics(y, p)
        y_fake = y == 0
        m["fake_recall"] = float(((p >= 0.5) & y_fake).sum() / y_fake.sum())  # fakes bien détectés
        m["real_recall"] = float(((p < 0.5) & ~y_fake).sum() / (~y_fake).sum())
        results[name] = {k: round(float(v), 4) for k, v in m.items()}
        print(f"{tag:12s} {name:18s} acc={m['accuracy']:.3f} fake_recall={m['fake_recall']:.3f}")

    out = RUNS / f"{tag}_s{MAIN_SEED}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "robustness.json").write_text(
        json.dumps({"n_images": len(df), "results": results}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"{out / 'robustness.json'} écrit")


# ----------------------------------------------------------------- comparaison et rapport


def report():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = list_runs()
    data = {}
    for tag in runs:
        r = load_json(main_run(runs, tag), "robustness.json")
        if r:
            data[tag] = r
    if not data:
        raise SystemExit("Aucun robustness.json : lance d'abord --tag ... --checkpoint ...")
    Path("metrics").mkdir(exist_ok=True)
    Path("metrics/robustness.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    names = list(PERTURBATIONS)
    fig, ax = plt.subplots(figsize=(11, 4))
    width = 0.8 / len(data)
    for i, (tag, r) in enumerate(data.items()):
        acc = [r["results"][n]["accuracy"] for n in names]
        ax.bar(np.arange(len(names)) + (i - (len(data) - 1) / 2) * width, acc, width, label=latex.plain(tag))
    ax.axhline(0.5, color="grey", ls="--", lw=1)
    ax.text(len(names) - 0.5, 0.51, "hasard", color="grey", fontsize=8, ha="right")
    ax.set_xticks(np.arange(len(names)), names, rotation=30, ha="right")
    ax.set_ylabel("exactitude")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=len(data), frameon=False)
    ax.set_title(
        f"Exactitude sous dégradation ({next(iter(data.values()))['n_images']} images de test équilibrées)"
    )
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    latex.save_figure(fig, "robustness")
    plt.close(fig)

    tags = list(data)
    rows = [[tex_label(n), *(latex.fr(data[t]["results"][n]["accuracy"]) for t in tags)] for n in names]
    n_img = next(iter(data.values()))["n_images"]
    body = latex.table(
        ["Dégradation", *(latex.name(t) for t in tags)],
        rows,
        f"Exactitude sous dégradation sur {n_img} images de test équilibrées (hasard : 0,5).",
    )

    # Phrases factuelles sur le cadrage, et comparaison basic / scale si les deux existent
    sentences = []
    for t in tags:
        res = data[t]["results"]
        worst = min(CADRAGE, key=lambda n: res[n]["accuracy"])
        worst_tex = tex_label(worst)
        sentences.append(
            f"{latex.name(t)} : exactitude {latex.fr(res['original']['accuracy'])} sans dégradation, "
            f"{latex.fr(res[worst]['accuracy'])} au pire cas de cadrage ({worst_tex}, "
            f"rappel des faux {latex.fr(res[worst]['fake_recall'])})."
        )
    if "vit" in data and "vit_scale" in data:
        a, b = data["vit"]["results"], data["vit_scale"]["results"]
        mean_a = np.mean([a[n]["accuracy"] for n in CADRAGE])
        mean_b = np.mean([b[n]["accuracy"] for n in CADRAGE])
        verdict = (
            "améliore"
            if mean_b > mean_a + 0.01
            else ("dégrade" if mean_b < mean_a - 0.01 else "ne change pas")
        )
        sentences.append(
            f"En moyenne sur les quatre dégradations de cadrage, l'augmentation \\texttt{{scale}} {verdict} "
            "la "
            f"robustesse du ViT : {latex.fr(mean_a)} $\\to$ {latex.fr(mean_b)}, pour une exactitude sans "
            "dégradation "
            f"de {latex.fr(a['original']['accuracy'])} $\\to$ {latex.fr(b['original']['accuracy'])}."
        )
    body += "\\begin{itemize}\n" + "\n".join(f"\\item {s}" for s in sentences) + "\n\\end{itemize}\n"
    latex.write("robustness", body)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", help="nom de l'expérience (vit, vit_scale, efficientnet...)")
    ap.add_argument("--model", help="architecture : baseline | efficientnet | vit")
    ap.add_argument("--checkpoint", help="fichier .pt du modèle")
    ap.add_argument("--n-images", type=int, default=1000)
    ap.add_argument("--report", action="store_true", help="compare les modèles déjà mesurés")
    args = ap.parse_args()
    if args.report:
        report()
    else:
        if not (args.tag and args.model and args.checkpoint):
            ap.error("--tag, --model et --checkpoint sont requis (ou --report)")
        evaluate_model(args.tag, args.model, args.checkpoint, args.n_images)


if __name__ == "__main__":
    main()
