"""Courbes d'apprentissage : perte d'entraînement, perte et F1 de validation par époque.

python -m src.analysis.learning_curves
->  reports/learning_curves.png, docs/rapport/generated/learning_curves.tex

Sert à justifier le nombre d'époques et à vérifier l'absence de sur-apprentissage
(une perte de validation qui remonte alors que la perte d'entraînement baisse).
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.analysis import latex  # noqa: E402
from src.analysis.runs import list_runs, load_json, main_run  # noqa: E402

COLORS = {"baseline": "#7f7f7f", "efficientnet": "#2ca02c", "vit": "#1f77b4", "vit_scale": "#9467bd"}


def overfit_diagnosis(epochs):
    """Retourne une phrase factuelle sur l'évolution des pertes à la dernière époque."""
    if len(epochs) < 2:
        return "une seule époque, pas de tendance mesurable"
    tl = [e["train_loss"] for e in epochs]
    vl = [e["val_loss"] for e in epochs]
    best = min(range(len(vl)), key=vl.__getitem__) + 1
    if vl[-1] > min(vl) * 1.10 and tl[-1] < tl[best - 1]:
        return (
            f"la perte de validation est minimale à l'époque {best} puis remonte alors que la perte "
            "d'entraînement continue de baisser : début de sur-apprentissage (le checkpoint retenu est le "
            "meilleur "
            "en validation)"
        )
    return (
        "la perte de validation baisse ou se stabilise jusqu'à la dernière époque "
        f"(minimum à l'époque {best}) : pas de sur-apprentissage visible"
    )


def main():
    runs = list_runs()
    histories = {}
    for tag in runs:
        h = load_json(main_run(runs, tag), "history.json")
        if h and h.get("epochs"):
            histories[tag] = h
    if not histories:
        raise SystemExit(
            "Aucun history.json dans reports/runs/ : relance les entraînements avec la version à jour."
        )

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 3.8))
    for tag, h in histories.items():
        c = COLORS.get(tag)
        lab = latex.plain(tag)
        if h["steps"]:
            ax1.plot(
                [s["epoch"] for s in h["steps"]],
                [s["train_loss"] for s in h["steps"]],
                color=c,
                alpha=0.8,
                label=lab,
            )
        ep = [e["epoch"] for e in h["epochs"]]
        ax2.plot(ep, [e["val_loss"] for e in h["epochs"]], marker="o", color=c, label=lab)
        ax2.plot(ep, [e["train_loss"] for e in h["epochs"]], marker="x", ls=":", color=c)
        ax3.plot(ep, [e["val_f1"] for e in h["epochs"]], marker="o", color=c, label=lab)
    ax1.set(
        title="Perte d'entraînement (moyenne glissante)",
        xlabel="époque",
        ylabel="entropie croisée",
        yscale="log",
    )
    ax2.set(title="Perte : validation (—) / entraînement (···)", xlabel="époque", yscale="log")
    ax3.set(title="F1 de validation", xlabel="époque", ylabel="F1")
    for ax in (ax1, ax2, ax3):
        ax.grid(alpha=0.3)
    ax3.legend(fontsize=8)
    plt.tight_layout()
    latex.save_figure(fig, "learning_curves")
    plt.close(fig)

    items = []
    for tag, h in histories.items():
        last = h["epochs"][-1]
        items.append(
            f"\\item \\textbf{{{latex.name(tag)}}} ({len(h['epochs'])} époques) : F1 de validation final "
            f"{latex.fr(last['val_f1'])}, perte de validation {latex.fr(last['val_loss'], 4)} ; "
            f"{overfit_diagnosis(h['epochs'])}."
        )
    latex.write("learning_curves", "\\begin{itemize}\n" + "\n".join(items) + "\n\\end{itemize}\n")


if __name__ == "__main__":
    main()
