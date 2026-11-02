"""Test de charge : locust -f tests/load/locustfile.py --host http://localhost:8000

Génère une image synthétique et mesure la latence de /predict (utile pour comparer ONNX et PyTorch).
"""

import io

from locust import HttpUser, between, task
from PIL import Image


def _image_bytes():
    buf = io.BytesIO()
    Image.effect_noise((256, 256), 64).convert("RGB").save(buf, "JPEG")
    return buf.getvalue()


class DeepGuardUser(HttpUser):
    wait_time = between(0.2, 1.0)
    payload = _image_bytes()

    @task(5)
    def predict(self):
        self.client.post("/predict", files={"file": ("test.jpg", self.payload, "image/jpeg")})

    @task(1)
    def health(self):
        self.client.get("/health")
