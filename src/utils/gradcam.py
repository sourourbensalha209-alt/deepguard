"""Explicabilité : CAM (équivalent Grad-CAM) pour les CNN, Attention Rollout pour le ViT.

python -m src.utils.gradcam  ->  reports/heatmaps.png (4 images fake + 4 images real du test)
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from src.data.dataset import get_transform  # noqa: E402
from src.models.architecture import build_model  # noqa: E402
from src.utils.config import load_params  # noqa: E402


def cnn_cam(model, x):
    """CAM de la classe « fake ».

    Pour une tête « pooling global + couche linéaire », Grad-CAM est équivalent à CAM :
    on l'obtient sans gradient (donc exportable en ONNX).
    """
    f = model.forward_features(x)
    logits = model.forward_head(f)
    cam = torch.einsum("bchw,c->bhw", f, model.classifier.weight[0])
    return logits, cam.clamp(min=0)


def vit_rollout(model, x):
    """Attention Rollout (Abnar & Zuidema, 2020) pour un ViT timm. Retourne (logits, masque 0..1)."""
    maps, hooks = [], []
    for blk in model.blocks:
        blk.attn.fused_attn = False  # nécessaire pour obtenir la matrice d'attention
        hooks.append(blk.attn.attn_drop.register_forward_hook(lambda m, i, o: maps.append(o.detach().cpu())))
    with torch.no_grad():
        logits = model(x)
    for h in hooks:
        h.remove()

    result = torch.eye(maps[0].shape[-1])
    for a in maps:
        a = a[0].mean(0) + torch.eye(a.shape[-1])  # moyenne des têtes + connexion résiduelle
        a = a / a.sum(-1, keepdim=True)
        result = a @ result
    mask = result[0, model.num_prefix_tokens :]  # token CLS -> patchs
    n = int(mask.numel() ** 0.5)
    mask = mask.reshape(n, n)
    return logits, (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)


def explain_image(model, kind, x):
    """Retourne (P(fake), carte 0..1) pour une image x de forme (1, 3, H, W)."""
    if kind == "vit":
        logits, mask = vit_rollout(model, x)
    else:
        with torch.no_grad():
            logits, mask = cnn_cam(model, x)
        mask = mask[0]
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    return torch.softmax(logits.float(), 1)[0, 0].item(), mask.cpu().numpy()


def main(n_per_class=4):
    params = load_params()
    t = params["train"]
    size = t["img_size"]
    model = build_model(t["model"], pretrained=False)
    model.load_state_dict(torch.load("checkpoints/best.pt", map_location="cpu"))
    model.eval()
    tf = get_transform(size)

    df = pd.read_csv("data/processed/test.csv")
    sample = pd.concat([df[df.label == 0].head(n_per_class), df[df.label == 1].head(n_per_class)])
    fig, axes = plt.subplots(2, len(sample), figsize=(2.6 * len(sample), 5.6))
    for i, (_, row) in enumerate(sample.iterrows()):
        img = Image.open(Path(params["data_dir"]) / row["path"]).convert("RGB").resize((size, size))
        p_fake, mask = explain_image(model, t["model"], tf(img).unsqueeze(0))
        truth = "fake" if row["label"] == 0 else "real"
        axes[0, i].imshow(img)
        axes[0, i].set_title(f"{truth} (P fake = {p_fake:.2f})", fontsize=9)
        axes[1, i].imshow(img)
        axes[1, i].imshow(mask, cmap="jet", alpha=0.45, extent=(0, size, size, 0), interpolation="bilinear")
        for ax in axes[:, i]:
            ax.axis("off")
    Path("reports").mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig("reports/heatmaps.png", dpi=150)
    plt.close()
    print("reports/heatmaps.png écrit")


if __name__ == "__main__":
    main()
