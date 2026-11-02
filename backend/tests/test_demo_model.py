"""Test d'intégration : vrai Predictor ONNX (modèle de démo), sans faux modèle."""

from .conftest import make_image


def test_demo_model_is_loaded_and_flagged(real_client):
    h = real_client.get("/health").json()
    assert h["model_loaded"] is True
    assert h["demo_model"] is True
    meta = real_client.get("/model/metadata").json()
    assert meta["demo"] is True


def test_end_to_end_with_onnx(real_client):
    r = real_client.post("/predict/visualize", files={"file": ("a.png", make_image(), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] in {"REAL", "FAKE"}
    assert body["face_detected"] is False  # image grise : pas de visage, image entière analysée
    assert body["heatmap"].startswith("data:image/png;base64,")
