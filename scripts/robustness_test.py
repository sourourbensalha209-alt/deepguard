"""Test de robustesse : dégrade une image de plusieurs façons et mesure la réponse de l'API.

Usage (l'application doit tourner : docker-compose up -d) :
    python scripts/robustness_test.py visage.jpg
    python scripts/robustness_test.py visage.jpg --melange autre_visage.jpg
    python scripts/robustness_test.py visage.jpg --api http://localhost:8000

Pour chaque variante (compression JPEG, réduction de résolution, flou, bruit, recadrage, mélange avec une
deuxième image), il affiche le verdict et la probabilité « générée par IA » donnés par le modèle, et écrit un
fichier robustness_results.csv à joindre au rapport. Les images ne sont pas enregistrées.
"""

import argparse
import csv
import io
import sys

import httpx
import numpy as np
from PIL import Image, ImageFilter


def load(path):
    return Image.open(path).convert("RGB")


def jpeg(img, quality):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def downscale(img, side):
    w, h = img.size
    scale = side / max(w, h)
    small = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BICUBIC)
    return small.resize((w, h), Image.BICUBIC)


def noise(img, sigma):
    arr = np.asarray(img, dtype=np.float32)
    arr = arr + np.random.default_rng(0).normal(0, sigma, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype("uint8"))


def center_crop(img, fraction):
    w, h = img.size
    cw, ch = round(w * fraction), round(h * fraction)
    left, top = (w - cw) // 2, (h - ch) // 2
    return img.crop((left, top, left + cw, top + ch))


def variants(img, other=None):
    yield "original", img
    for q in (75, 50, 25, 10):
        yield f"JPEG qualité {q}", jpeg(img, q)
    for side in (128, 64, 32):
        yield f"résolution réduite à {side} px", downscale(img, side)
    for radius in (1, 2, 4):
        yield f"flou gaussien {radius}", img.filter(ImageFilter.GaussianBlur(radius))
    for sigma in (5, 15, 30):
        yield f"bruit sigma {sigma}", noise(img, sigma)
    for frac in (0.8, 0.6):
        yield f"recadrage central {int(frac * 100)} %", center_crop(img, frac)
    if other is not None:
        other = other.resize(img.size, Image.BICUBIC)
        for alpha in (0.25, 0.5, 0.75):
            yield f"mélange {int(alpha * 100)} % de l'autre image", Image.blend(img, other, alpha)


def predict(client, api, img, as_jpeg):
    buf = io.BytesIO()
    img.save(buf, "JPEG" if as_jpeg else "PNG", **({"quality": 95} if as_jpeg else {}))
    mime = "image/jpeg" if as_jpeg else "image/png"
    r = client.post(f"{api}/predict", files={"file": ("test", buf.getvalue(), mime)}, timeout=60)
    r.raise_for_status()
    return r.json()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("image")
    parser.add_argument("--melange", help="deuxième image pour les mélanges (ex. un vrai visage)")
    parser.add_argument("--api", default="http://localhost:8000")
    args = parser.parse_args()

    img = load(args.image)
    other = load(args.melange) if args.melange else None
    rows = []
    with httpx.Client() as client:
        try:
            health = client.get(f"{args.api}/health", timeout=10).json()
        except httpx.HTTPError as exc:
            sys.exit(f"API injoignable sur {args.api} : lance d'abord l'application ({exc})")
        if health.get("demo_model"):
            print("ATTENTION : modèle de démonstration (poids aléatoires), résultats sans valeur.\n")

        print(f"{'Variante':<38}{'Verdict':<8}{'P(généré)':>10}  Visage détecté")
        print("-" * 72)
        for name, variant in variants(img, other):
            res = predict(client, args.api, variant, as_jpeg=name.startswith("JPEG"))
            p = res["prob_fake"] * 100
            rows.append(
                {
                    "variante": name,
                    "verdict": res["label"],
                    "prob_generee_pct": round(p, 1),
                    "visage_detecte": res["face_detected"],
                }
            )
            print(f"{name:<38}{res['label']:<8}{p:>9.1f}%  {'oui' if res['face_detected'] else 'non'}")

    with open("robustness_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("\nRésultats enregistrés dans robustness_results.csv")


if __name__ == "__main__":
    main()
