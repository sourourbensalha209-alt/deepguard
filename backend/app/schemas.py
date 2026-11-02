from typing import Literal

from pydantic import BaseModel, Field


class Prediction(BaseModel):
    label: Literal["REAL", "FAKE"]
    prob_fake: float = Field(ge=0, le=1, description="Probabilité que le visage soit généré par IA")
    confidence: float = Field(ge=0.5, le=1, description="Probabilité de la classe prédite")
    face_detected: bool = Field(description="Faux : l'image entière a été analysée, sans recadrage")
    face_box: list[float] = Field(
        description="Région analysée [x, y, largeur, hauteur], en fraction de l'image d'origine (0 à 1)"
    )
    latency_ms: float = Field(description="Temps de traitement côté serveur")


class VisualizeResponse(Prediction):
    heatmap: str = Field(
        description="Calque PNG transparent (data URL base64) à superposer à la région `face_box` : "
        "les zones colorées sont celles qui ont le plus pesé dans la décision"
    )


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    demo_model: bool = Field(description="Vrai : modèle de démonstration (poids aléatoires)")
    gpu_available: bool = Field(description="Un GPU est-il utilisable par ONNX Runtime ?")
    onnx_providers: list[str] = Field(description="Fournisseurs d'exécution ONNX Runtime actifs")
    model_version: str | None = Field(description="Version du modèle chargé (architecture + commit Git)")
