from .conftest import make_image


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert isinstance(body["gpu_available"], bool)
    assert "CPUExecutionProvider" in body["onnx_providers"]
    assert body["model_version"] == "demo"


def test_predict_ok(client):
    r = client.post("/predict", files={"file": ("a.png", make_image(), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == "FAKE"
    assert 0 <= body["prob_fake"] <= 1
    assert body["confidence"] == max(body["prob_fake"], 1 - body["prob_fake"])
    assert body["latency_ms"] >= 0


def test_predict_accepts_jpeg(client):
    r = client.post("/predict", files={"file": ("a.jpg", make_image(fmt="JPEG"), "image/jpeg")})
    assert r.status_code == 200


def test_visualize_returns_heatmap(client):
    r = client.post("/predict/visualize", files={"file": ("a.png", make_image(), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["heatmap"].startswith("data:image/png;base64,")
    assert body["face_box"] == [0.0, 0.0, 1.0, 1.0]  # pas de visage : image entière


def test_rejects_wrong_type(client):
    r = client.post("/predict", files={"file": ("a.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_rejects_corrupt_image(client):
    r = client.post("/predict", files={"file": ("a.png", b"not an image", "image/png")})
    assert r.status_code == 400


def test_rejects_too_large(client):
    big = b"0" * (5 * 1024 * 1024 + 1)
    r = client.post("/predict", files={"file": ("a.png", big, "image/png")})
    assert r.status_code == 413


def test_rejects_missing_file(client):
    assert client.post("/predict").status_code == 422


def test_metrics_endpoint_exposes_counters(client):
    client.post("/predict", files={"file": ("a.png", make_image(), "image/png")})
    body = client.get("/metrics").text
    assert "deepguard_predictions_total" in body
    assert "http_request_duration" in body


def test_heatmap_layer_is_transparent_png():
    import base64
    import io

    import numpy as np
    from PIL import Image

    from app.inference import heatmap_layer

    layer = heatmap_layer((1200, 800), np.random.default_rng(1).random((7, 7)).astype("float32"))
    assert layer.mode == "RGBA"
    assert max(layer.size) == 512  # réduit pour limiter le poids de la réponse
    buf = io.BytesIO()
    layer.save(buf, "PNG")
    assert Image.open(io.BytesIO(base64.b64decode(base64.b64encode(buf.getvalue())))).size == layer.size
