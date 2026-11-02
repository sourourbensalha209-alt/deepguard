"""Prétraitement, inférence ONNX et génération de la carte de chaleur (sans PyTorch)."""

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SIZE = 224
_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def crop_face(img: Image.Image, margin: float = 0.35) -> tuple[Image.Image, tuple[int, int, int, int] | None]:
    """Recadre sur le plus grand visage détecté.

    Retourne (image analysée, boîte (x0, y0, x1, y1) en pixels). Boîte = None si aucun visage
    n'est détecté (ou si le détecteur est absent) : l'image entière est alors analysée.
    """
    if _CASCADE.empty():
        return img, None
    gray = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)
    faces = _CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(64, 64))
    if len(faces) == 0:
        return img, None
    x, y, w, h = (int(v) for v in max(faces, key=lambda f: f[2] * f[3]))
    dx, dy = int(w * margin), int(h * margin)
    box = (max(x - dx, 0), max(y - dy, 0), min(x + w + dx, img.width), min(y + h + dy, img.height))
    return img.crop(box), box


class Predictor:
    """Charge un modèle ONNX à deux sorties : logits (N, 2) et carte CAM (N, h, w). Classe 0 = fake."""

    def __init__(self, model_path: str):
        wanted = ("CUDAExecutionProvider", "CPUExecutionProvider")
        providers = [p for p in wanted if p in ort.get_available_providers()]
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    @property
    def providers(self) -> list[str]:
        return self.session.get_providers()

    @property
    def gpu_available(self) -> bool:
        return ort.get_device() == "GPU"

    @staticmethod
    def preprocess(img: Image.Image) -> np.ndarray:
        arr = np.asarray(img.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR), dtype=np.float32) / 255.0
        arr = (arr - MEAN) / STD
        return arr.transpose(2, 0, 1)[None].astype(np.float32)

    def predict(self, img: Image.Image) -> tuple[float, np.ndarray]:
        logits, cam = self.session.run(None, {self.input_name: self.preprocess(img)})
        z = logits[0] - logits[0].max()
        probs = np.exp(z) / np.exp(z).sum()
        return float(probs[0]), cam[0]


def heatmap_layer(size: tuple[int, int], cam: np.ndarray, max_side: int = 512) -> Image.Image:
    """Calque RGBA (palette « jet ») : transparent là où la décision a peu pesé.

    `size` = (largeur, hauteur) de la région analysée. Le calque est réduit à `max_side` pixels
    au plus : le navigateur l'étire sur la photo.
    """
    scale = min(1.0, max_side / max(size))
    w, h = max(1, round(size[0] * scale)), max(1, round(size[1] * scale))
    cam = np.maximum(cam, 0)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    mask = Image.fromarray((cam * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    m = np.asarray(mask, dtype=np.float32) / 255.0
    r, g, b = (np.clip(1.5 - np.abs(4 * m - k), 0, 1) for k in (3, 2, 1))
    alpha = np.clip((m - 0.3) / 0.7, 0, 1) * 0.8  # le quart le moins influent reste transparent
    rgba = np.stack([r, g, b, alpha], -1)
    return Image.fromarray((rgba * 255).astype(np.uint8), "RGBA")
