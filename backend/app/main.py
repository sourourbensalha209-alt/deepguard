"""API REST DeepGuard : détection de visages générés par IA."""

import base64
import io
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, UnidentifiedImageError
from prometheus_client import Counter
from prometheus_fastapi_instrumentator import Instrumentator

from .inference import Predictor, crop_face, heatmap_layer
from .schemas import HealthResponse, Prediction, VisualizeResponse

log = logging.getLogger("deepguard")

# backend/ en local, /app dans Docker
BASE = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE.parent / "models" / "deepguard.onnx"))
META_PATH = Path(os.getenv("META_PATH", BASE.parent / "models" / "metadata.json"))
DEMO_DIR = Path(os.getenv("DEMO_MODEL_DIR", BASE / "demo_model"))

MAX_BYTES = 5 * 1024 * 1024
MAX_PIXELS = 40_000_000
ALLOWED = {"image/jpeg", "image/png", "image/webp"}

PREDICTIONS = Counter("deepguard_predictions_total", "Prédictions rendues, par verdict", ["label"])
NO_FACE = Counter("deepguard_no_face_total", "Images sans visage détecté")

state: dict = {}


def load_model() -> None:
    """Charge le modèle entraîné s'il existe, sinon le modèle de démonstration (poids aléatoires)."""
    if MODEL_PATH.exists():
        model, meta_path = MODEL_PATH, META_PATH
        log.info("Modèle chargé : %s", model)
    elif (DEMO_DIR / "deepguard-demo.onnx").exists():
        model, meta_path = DEMO_DIR / "deepguard-demo.onnx", DEMO_DIR / "metadata.json"
        log.warning("Aucun modèle entraîné trouvé (%s) : MODÈLE DE DÉMONSTRATION chargé.", MODEL_PATH)
    else:
        log.error("Aucun modèle disponible : l'API répondra 503 sur /predict.")
        return
    state["predictor"] = Predictor(model)
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.setdefault("demo", model.name.endswith("-demo.onnx"))
    meta.setdefault("architecture", "inconnue")
    meta.setdefault("classes", ["fake", "real"])
    state["meta"] = meta


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield
    state.clear()


app = FastAPI(
    title="DeepGuard API",
    version="1.0.0",
    description="Détecte si un visage est réel ou généré par IA, et montre les zones qui ont pesé.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
Instrumentator().instrument(app).expose(app, include_in_schema=False)  # /metrics (Prometheus)


def get_predictor():
    if "predictor" not in state:
        raise HTTPException(503, "Aucun modèle n'est chargé sur le serveur.")
    return state["predictor"]


async def read_image(file: UploadFile) -> Image.Image:
    if file.content_type not in ALLOWED:
        raise HTTPException(400, "Format non supporté : envoie une image JPEG, PNG ou WebP.")
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Image trop lourde : 5 Mo maximum.")
    try:
        img = Image.open(io.BytesIO(data))
        if img.width * img.height > MAX_PIXELS:
            raise HTTPException(413, "Image trop grande : réduis sa résolution.")
        return ImageOps.exif_transpose(img).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "Fichier illisible : ce n'est pas une image valide.") from None


def run(img: Image.Image, predictor):
    t0 = time.perf_counter()
    face, box = crop_face(img)
    found = box is not None
    p_fake, cam = predictor.predict(face)
    ms = (time.perf_counter() - t0) * 1000
    label = "FAKE" if p_fake >= 0.5 else "REAL"
    if found:
        x0, y0, x1, y1 = box
        face_box = [
            round(x0 / img.width, 4),
            round(y0 / img.height, 4),
            round((x1 - x0) / img.width, 4),
            round((y1 - y0) / img.height, 4),
        ]
    else:
        face_box = [0.0, 0.0, 1.0, 1.0]
    PREDICTIONS.labels(label).inc()
    if not found:
        NO_FACE.inc()
    pred = Prediction(
        label=label,
        prob_fake=round(p_fake, 4),
        confidence=round(max(p_fake, 1 - p_fake), 4),
        face_detected=found,
        face_box=face_box,
        latency_ms=round(ms, 1),
    )
    return face, cam, pred


@app.get("/health", response_model=HealthResponse, tags=["système"])
def health():
    predictor, meta = state.get("predictor"), state.get("meta", {})
    return {
        "status": "ok",
        "model_loaded": predictor is not None,
        "demo_model": bool(meta.get("demo", False)),
        "gpu_available": bool(predictor and predictor.gpu_available),
        "onnx_providers": predictor.providers if predictor else [],
        "model_version": meta.get("version") or meta.get("git_sha"),
    }


@app.get("/model/metadata", tags=["système"])
def model_metadata():
    if "meta" not in state:
        raise HTTPException(503, "Aucun modèle n'est chargé sur le serveur.")
    return state["meta"]


@app.post("/predict", response_model=Prediction, tags=["prédiction"])
async def predict(file: UploadFile = File(...), predictor=Depends(get_predictor)):
    img = await read_image(file)
    _, _, pred = await run_in_threadpool(run, img, predictor)
    return pred


@app.post("/predict/visualize", response_model=VisualizeResponse, tags=["prédiction"])
async def visualize(file: UploadFile = File(...), predictor=Depends(get_predictor)):
    """Même verdict que /predict, plus un calque de chaleur transparent (PNG en base64)."""
    img = await read_image(file)
    face, cam, pred = await run_in_threadpool(run, img, predictor)
    buf = io.BytesIO()
    heatmap_layer(face.size, cam).save(buf, format="PNG")
    data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    return VisualizeResponse(**pred.model_dump(), heatmap=data_url)
