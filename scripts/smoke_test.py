"""Test de fumée de l'application lancée : python scripts/smoke_test.py [url_api] [url_frontend]"""

import io
import sys

import httpx
from PIL import Image

API = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
FRONT = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:3000"


def check(name, ok, detail=""):
    print(f"[{'OK' if ok else 'ÉCHEC'}] {name} {detail}")
    if not ok:
        sys.exit(1)


def main():
    health = httpx.get(f"{API}/health", timeout=10).json()
    check("API /health", health["status"] == "ok" and health["model_loaded"], str(health))
    meta = httpx.get(f"{API}/model/metadata", timeout=10).json()
    check("API /model/metadata", "architecture" in meta, meta.get("architecture", ""))

    buf = io.BytesIO()
    Image.effect_noise((256, 256), 64).convert("RGB").save(buf, "JPEG")
    files = {"file": ("smoke.jpg", buf.getvalue(), "image/jpeg")}
    r = httpx.post(f"{API}/predict/visualize", files=files, timeout=30)
    check("API /predict/visualize", r.status_code == 200 and r.json()["heatmap"].startswith("data:image/png"))
    check("API /metrics", "deepguard_predictions_total" in httpx.get(f"{API}/metrics", timeout=10).text)

    try:
        page = httpx.get(FRONT, timeout=10)
        check("Frontend", page.status_code == 200 and "DeepGuard" in page.text, FRONT)
    except httpx.HTTPError as exc:
        print(f"[info] Frontend injoignable ({exc}) : ignoré")
    if meta.get("demo"):
        print("\nATTENTION : modèle de démonstration (poids aléatoires). Entraîne un vrai modèle (README).")


if __name__ == "__main__":
    main()
