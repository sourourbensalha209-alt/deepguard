"""Ce que le ViT a appris à l'intérieur (Axe 4) : embeddings de position et distance d'attention.

python -m src.analysis.vit_internals --checkpoint /kaggle/working/vit.pt [--compare-pretrained]
->  reports/vit_pos_embed.png, reports/vit_attention_distance.png, metrics/vit_internals.json,
    docs/rapport/generated/vit_internals.tex

1. Similarité cosinus des embeddings de position (Dosovitskiy et al., 2021, fig. 7) : pour chacun des
   14 x 14 patchs, une petite carte montre à quels autres patchs son embedding ressemble. Une structure en
   lignes et colonnes prouve que le modèle a appris la géométrie 2D de l'image sans qu'on la lui donne.
2. Distance moyenne d'attention par couche et par tête : à quelle distance (en pixels) chaque tête regarde.
   Contrairement à un CNN, dont le champ réceptif est local au début, certaines têtes du ViT regardent loin
   dès la première couche : c'est l'auto-attention globale.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from src.analysis import latex  # noqa: E402
from src.data.dataset import get_transform  # noqa: E402
from src.models.architecture import build_model  # noqa: E402
from src.utils.config import get_device, load_params  # noqa: E402


def pos_embed_similarity(model):
    """Similarité cosinus entre embeddings de position des patchs, de forme (g, g, g, g)."""
    pe = model.pos_embed[0, model.num_prefix_tokens :].detach().float().cpu()  # (196, dim)
    pe = torch.nn.functional.normalize(pe, dim=-1)
    sim = pe @ pe.T
    g = int(round(sim.shape[0] ** 0.5))
    return sim.reshape(g, g, g, g).numpy()


def mosaic(sim):
    """Assemble les g x g cartes g x g en une seule image (avec une ligne de séparation)."""
    g = sim.shape[0]
    out = np.full((g * (g + 1), g * (g + 1)), np.nan)
    for i in range(g):
        for j in range(g):
            out[i * (g + 1) : i * (g + 1) + g, j * (g + 1) : j * (g + 1) + g] = sim[i, j]
    return out


def neighbour_score(sim):
    """Similarité moyenne avec les 4 voisins directs, et avec les patchs éloignés (distance > 4)."""
    g = sim.shape[0]
    near, far = [], []
    for i in range(g):
        for j in range(g):
            for a in range(g):
                for b in range(g):
                    d = abs(i - a) + abs(j - b)
                    if d == 1:
                        near.append(sim[i, j, a, b])
                    elif d > 4:
                        far.append(sim[i, j, a, b])
    return float(np.mean(near)), float(np.mean(far))


@torch.no_grad()
def attention_distances(model, images, device):
    """Distance moyenne d'attention (pixels) : tableau (couches, têtes), moyenné sur les images."""
    maps, hooks = [], []
    for blk in model.blocks:
        blk.attn.fused_attn = False  # nécessaire pour récupérer la matrice d'attention
        hooks.append(
            blk.attn.attn_drop.register_forward_hook(lambda m, i, o: maps.append(o.detach().float().cpu()))
        )
    patch = model.patch_embed.patch_size[0]
    g = model.patch_embed.grid_size[0]
    coords = np.stack(np.meshgrid(np.arange(g), np.arange(g), indexing="ij"), -1).reshape(-1, 2) * patch
    dist = torch.tensor(
        np.linalg.norm(coords[:, None] - coords[None], axis=-1), dtype=torch.float32
    )  # (N, N)
    npre = model.num_prefix_tokens
    total = None
    try:
        for x in images:
            maps.clear()
            model(x.to(device))
            per_layer = []
            for a in maps:  # (B, têtes, 1+N, 1+N)
                a = a[:, :, npre:, npre:]
                a = a / a.sum(-1, keepdim=True)  # renormalise l'attention patch -> patch
                per_layer.append((a * dist).sum(-1).mean(dim=(0, 2)))  # (têtes,)
            d = torch.stack(per_layer).numpy() * len(x)
            total = d if total is None else total + d
    finally:
        for h in hooks:
            h.remove()
    n = sum(len(x) for x in images)
    return total / n


def load_images(params, n=64, batch=16):
    size = params["train"]["img_size"]
    tf = get_transform(size)
    df = pd.read_csv("data/processed/test.csv").head(n)
    root = Path(params["data_dir"])
    xs = torch.stack([tf(Image.open(root / p).convert("RGB")) for p in df["path"]])
    return list(torch.split(xs, batch))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="poids du ViT fine-tuné (.pt)")
    ap.add_argument(
        "--compare-pretrained", action="store_true", help="compare aux poids ImageNet (Internet requis)"
    )
    ap.add_argument("--n-images", type=int, default=64)
    args = ap.parse_args()

    params = load_params()
    device = get_device()
    models = {}
    ft = build_model("vit", pretrained=False)
    ft.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    models["fine-tuné (deepfakes)"] = ft.eval().to(device)
    if args.compare_pretrained:
        models["pré-entraîné ImageNet"] = build_model("vit", pretrained=True).eval().to(device)
    images = load_images(params, args.n_images)

    result = {}
    # 1. Embeddings de position
    sim = pos_embed_similarity(ft)
    near, far = neighbour_score(sim)
    result["pos_embed"] = {"grid": sim.shape[0], "sim_neighbours": near, "sim_far": far}
    fig, ax = plt.subplots(figsize=(6.5, 6))
    im = ax.imshow(mosaic(sim), cmap="viridis", vmin=-0.2, vmax=1)
    ax.set_title(
        "Similarité cosinus des embeddings de position\n(chaque petite carte = un patch, ligne x colonne)",
        fontsize=10,
    )
    ax.set_xlabel("colonne du patch")
    ax.set_ylabel("ligne du patch")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046)
    latex.save_figure(fig, "vit_pos_embed")
    plt.close(fig)

    # 2. Distance moyenne d'attention
    fig, ax = plt.subplots(figsize=(7, 4))
    markers = ["o", "x"]
    for k, (label, model) in enumerate(models.items()):
        d = attention_distances(model, images, device)  # (couches, têtes)
        result[f"attention_distance_{k}"] = {
            "label": label,
            "per_layer_mean": d.mean(1).round(1).tolist(),
            "per_layer_min": d.min(1).round(1).tolist(),
            "per_layer_max": d.max(1).round(1).tolist(),
        }
        layers = np.repeat(np.arange(1, d.shape[0] + 1), d.shape[1])
        ax.scatter(layers + 0.15 * k, d.ravel(), s=14, marker=markers[k % 2], alpha=0.75, label=label)
    size = params["train"]["img_size"]
    # Référence : distance moyenne entre deux points tirés au hasard dans l'image (~ 0,52 x côté)
    ax.axhline(0.52 * size, color="grey", ls=":", lw=1, label="attention uniforme")
    ax.set(
        xlabel="couche (bloc Transformer)",
        ylabel="distance moyenne d'attention (pixels)",
        title="Distance d'attention par tête (un point = une tête)",
    )
    ax.set_xticks(range(1, d.shape[0] + 1))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    latex.save_figure(fig, "vit_attention_distance")
    plt.close(fig)

    Path("metrics").mkdir(exist_ok=True)
    Path("metrics/vit_internals.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    write_latex(result)


def write_latex(result):
    pe = result["pos_embed"]
    ad = result["attention_distance_0"]
    first_min, first_max = ad["per_layer_min"][0], ad["per_layer_max"][0]
    last_mean = ad["per_layer_mean"][-1]
    body = (
        "\\begin{itemize}\n"
        f"\\item \\textbf{{Embeddings de position.}} La similarité cosinus moyenne entre un patch et ses 4 "
        "voisins directs "
        f"est de {latex.fr(pe['sim_neighbours'], 2)}, contre {latex.fr(pe['sim_far'], 2)} avec les patchs "
        "éloignés "
        "(distance de Manhattan $>4$). Les embeddings de position sont de simples vecteurs appris "
        "(pré-entraînés sur "
        "ImageNet puis ajustés au fine-tuning), sans notion de 2D imposée : la structure en lignes et "
        "colonnes de la "
        "figure montre que le modèle a appris seul la géométrie de l'image.\n"
        f"\\item \\textbf{{Distance d'attention.}} Dès la première couche, les têtes regardent entre "
        f"{latex.fr(first_min, 0)} "
        f"et {latex.fr(first_max, 0)} pixels en moyenne : certaines sont locales (comme un filtre de "
        "convolution), d'autres "
        f"globales. Dans la dernière couche, la distance moyenne est de {latex.fr(last_mean, 0)} pixels. Un "
        "CNN ne peut "
        "pas relier deux régions éloignées avant ses dernières couches ; le ViT le peut dès le début, ce "
        "qui l'aide à "
        "repérer des incohérences globales (symétrie, arrière-plan, éclairage) typiques des images "
        "générées.\n"
    )
    if "attention_distance_1" in result:
        ad1 = result["attention_distance_1"]
        delta = np.array(ad["per_layer_mean"]) - np.array(ad1["per_layer_mean"])
        layer = int(np.abs(delta).argmax()) + 1
        body += (
            f"\\item \\textbf{{Effet du fine-tuning.}} Par rapport aux poids ImageNet, le changement le "
            "plus fort de distance "
            f"moyenne est à la couche {layer} ({latex.fr(float(delta[layer - 1]), 0)} pixels) : c'est là "
            "que l'adaptation "
            "à la détection de deepfakes modifie le plus la manière de regarder.\n"
        )
    body += "\\end{itemize}\n"
    latex.write("vit_internals", body)


if __name__ == "__main__":
    main()
