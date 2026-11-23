"""Écriture des tableaux et phrases générés pour le rapport LaTeX (docs/rapport/generated/).

Le rapport les inclut avec la macro \\genere{nom}{...} : si le fichier n'existe pas encore, un encadré
« à compléter » s'affiche à la place. Aucun chiffre n'est donc écrit à la main.
"""

from pathlib import Path

GENERATED = Path("docs/rapport/generated")
FIGURES = Path("docs/rapport/figures")

NAMES = {
    "baseline": "Baseline CNN",
    "efficientnet": "EfficientNet-V2-S",
    "vit": "ViT-B/16",
    "vit_scale": "ViT-B/16 (\\texttt{scale})",
}


PLAIN = {
    "baseline": "Baseline CNN",
    "efficientnet": "EfficientNet-V2-S",
    "vit": "ViT-B/16",
    "vit_scale": "ViT-B/16 (scale)",
}


def plain(tag):
    """Nom lisible pour les figures matplotlib."""
    return PLAIN.get(tag, tag)


def name(tag):
    return NAMES.get(tag, tag.replace("_", "\\_"))


def fr(x, digits=3):
    """Nombre au format français : 0,996."""
    return f"{x:.{digits}f}".replace(".", "{,}")


def pct(x, digits=1):
    return f"{100 * x:.{digits}f}".replace(".", "{,}") + "\\,\\%"


def write(stem, body):
    GENERATED.mkdir(parents=True, exist_ok=True)
    path = GENERATED / f"{stem}.tex"
    path.write_text(
        "% Fichier généré automatiquement par src/analysis : ne pas modifier à la main.\n" + body, "utf-8"
    )
    print(f"{path} écrit")
    return path


def table(header, rows, caption, align=None):
    align = align or "@{}l" + "c" * (len(header) - 1) + "@{}"
    lines = [
        "\\begin{table}[H]",
        "\\centering\\small",
        f"\\begin{{tabular}}{{{align}}}",
        "\\toprule",
        " & ".join(f"\\textbf{{{h}}}" for h in header) + " \\\\",
        "\\midrule",
        *[" & ".join(str(c) for c in r) + " \\\\" for r in rows],
        "\\bottomrule",
        "\\end{tabular}",
        f"\\caption{{{caption}}}",
        "\\end{table}",
    ]
    return "\n".join(lines) + "\n"


def save_figure(fig, stem, dpi=150):
    """Enregistre une figure dans reports/ et dans docs/rapport/figures/ (utilisée par le rapport)."""
    Path("reports").mkdir(exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    for folder in (Path("reports"), FIGURES):
        fig.savefig(folder / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    print(f"reports/{stem}.png écrit")
