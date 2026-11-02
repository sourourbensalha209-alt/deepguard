"""Définition des trois architectures comparées : baseline CNN, EfficientNet-V2-S, ViT-B/16."""

import timm
import torch.nn as nn


class BaselineCNN(nn.Module):
    """Petit CNN de 4 blocs, entraîné from scratch. Même API que timm (forward_features / forward_head)."""

    def __init__(self, n_classes=2):
        super().__init__()

        def block(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True), nn.MaxPool2d(2)
            )

        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, 256))
        self.classifier = nn.Linear(256, n_classes)

    def forward_features(self, x):
        return self.features(x)

    def forward_head(self, f):
        return self.classifier(f.mean((2, 3)))

    def forward(self, x):
        return self.forward_head(self.forward_features(x))


def build_model(name, pretrained=True):
    if name == "baseline":
        return BaselineCNN()
    if name == "efficientnet":  # variante timm d'EfficientNet-V2-S (transfer learning ImageNet)
        return timm.create_model("efficientnetv2_rw_s", pretrained=pretrained, num_classes=2)
    if name == "vit":  # ViT-B/16 : patchs 16x16, embeddings de position appris, auto-attention globale
        return timm.create_model("vit_base_patch16_224", pretrained=pretrained, num_classes=2)
    raise ValueError(f"Modèle inconnu : {name} (baseline | efficientnet | vit)")


def describe_vit(model):
    """Stratégie de patchification et embeddings de position d'un ViT timm (pour MLflow et le rapport)."""
    if not hasattr(model, "patch_embed") or not hasattr(model, "pos_embed"):
        return {}
    pe = model.patch_embed
    return {
        "vit_patch_size": pe.patch_size[0],
        "vit_grid": f"{pe.grid_size[0]}x{pe.grid_size[1]}",
        "vit_num_patches": pe.num_patches,
        "vit_pos_embed_shape": str(tuple(model.pos_embed.shape)),  # (1, 1 + nb_patchs, dim)
    }
