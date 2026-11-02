import io

import numpy as np
import pytest
from app.main import app, get_predictor
from fastapi.testclient import TestClient
from PIL import Image


class FakePredictor:
    """Faux modèle : les tests de l'API n'ont besoin ni du vrai modèle ni de PyTorch."""

    def predict(self, img):
        return 0.93, np.random.default_rng(0).random((7, 7)).astype("float32")


@pytest.fixture()
def client():
    app.dependency_overrides[get_predictor] = lambda: FakePredictor()
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture()
def real_client():
    """Client SANS faux modèle : utilise le vrai chargement (modèle de démo ONNX)."""
    with TestClient(app) as c:
        yield c


def make_image(size=(256, 256), fmt="PNG", color="gray"):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, fmt)
    return buf.getvalue()
