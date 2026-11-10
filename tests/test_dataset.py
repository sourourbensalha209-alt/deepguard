"""Transformations d'image : l'augmentation « scale » ne touche que l'entraînement."""

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("torch")


def _img(w=300, h=260):
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8))


@pytest.mark.parametrize("augment", ["basic", "scale"])
def test_train_transform_output_shape(augment):
    from src.data.dataset import get_transform

    x = get_transform(224, train=True, augment=augment)(_img())
    assert tuple(x.shape) == (3, 224, 224)


def test_eval_transform_ignores_augmentation():
    import torch

    from src.data.dataset import get_transform

    img = _img()
    a = get_transform(224, train=False, augment="scale")(img)
    b = get_transform(224, train=False, augment="basic")(img)
    assert torch.equal(a, b)  # évaluation déterministe, identique quel que soit le réglage


def test_scale_augmentation_varies_the_view():
    import torch

    from src.data.dataset import get_transform

    tf = get_transform(224, train=True, augment="scale")
    views = [tf(_img()) for _ in range(6)]
    assert any(not torch.equal(views[0], v) for v in views[1:])
