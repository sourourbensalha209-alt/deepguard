"""Calibration des probabilités : ECE, score de Brier, diagramme de fiabilité et temperature scaling.

python -m src.analysis.calibration   ->  metrics/calibration.json, reports/calibration.png,
                                         docs/rapport/generated/calibration.tex

Un modèle est calibré si, parmi les images auxquelles il donne 80 % de probabilité « fake », environ 80 % sont
réellement fausses. Le temperature scaling (Guo et al., 2017) divise la marge logit par une température T
apprise sur la VALIDATION (jamais sur le test), ce qui ne change aucune prédiction (même seuil, même AUC)
mais rend les probabilités affichées dans l'interface plus honnêtes.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import minimize_scalar  # noqa: E402

from src.analysis import latex  # noqa: E402
from src.analysis.runs import list_runs, load_preds, main_run  # noqa: E402


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def nll(y_fake, z, T=1.0):
    """Entropie croisée moyenne, calculée de façon stable à partir de la marge z."""
    s = z / T
    # -[y log σ(s) + (1-y) log(1-σ(s))] = log(1 + e^s) - y s
    return float(np.mean(np.logaddexp(0.0, s) - y_fake * s))


def ece(y_fake, p, n_bins=15):
    """Expected Calibration Error : écart moyen confiance / exactitude, pondéré par la taille des bacs."""
    conf = np.maximum(p, 1 - p)  # confiance dans la classe prédite
    correct = ((p >= 0.5).astype(int) == y_fake).astype(float)
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            total += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(total)


def brier(y_fake, p):
    return float(np.mean((p - y_fake) ** 2))


def fit_temperature(y_fake, z):
    """Température minimisant la NLL sur la validation."""
    res = minimize_scalar(lambda logT: nll(y_fake, z, np.exp(logT)), bounds=(-4, 5), method="bounded")
    return float(np.exp(res.x))


def reliability(y_fake, p, n_bins=10):
    """Points du diagramme de fiabilité : P(fake) prédite moyenne contre fréquence réelle de « fake »."""
    edges = np.linspace(0, 1, n_bins + 1)
    bins = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    xs, ys, ns = [], [], []
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            xs.append(p[mask].mean())
            ys.append(y_fake[mask].mean())
            ns.append(int(mask.sum()))
    return np.array(xs), np.array(ys), np.array(ns)


def calibrate(val_df, test_df):
    yv = (val_df["label"].to_numpy() == 0).astype(int)
    yt = (test_df["label"].to_numpy() == 0).astype(int)
    zv, zt = val_df["margin"].to_numpy(np.float64), test_df["margin"].to_numpy(np.float64)
    T = fit_temperature(yv, zv)
    p_before, p_after = _sigmoid(zt), _sigmoid(zt / T)
    return {
        "temperature": T,
        "before": {"ece": ece(yt, p_before), "nll": nll(yt, zt), "brier": brier(yt, p_before)},
        "after": {"ece": ece(yt, p_after), "nll": nll(yt, zt, T), "brier": brier(yt, p_after)},
        "accuracy_unchanged": bool(np.array_equal(p_before >= 0.5, p_after >= 0.5)),
    }, (yt, p_before, p_after)


def main():
    runs = list_runs()
    if not runs:
        raise SystemExit("Aucun résultat dans reports/runs/ : lance d'abord le notebook.")
    results, curves = {}, {}
    for tag in runs:
        folder = main_run(runs, tag)
        if not (folder / "preds_val.csv").exists():
            continue
        results[tag], curves[tag] = calibrate(load_preds(folder, "val"), load_preds(folder, "test"))

    Path("metrics").mkdir(exist_ok=True)
    Path("metrics/calibration.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))

    fig, axes = plt.subplots(1, len(curves), figsize=(3.6 * len(curves), 3.7), squeeze=False)
    for ax, (tag, (y, pb, pa)) in zip(axes[0], curves.items()):
        ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1, label="calibration parfaite")
        for p, lab, col in (
            (pb, "brut", "#d62728"),
            (pa, f"T = {results[tag]['temperature']:.2f}", "#1f77b4"),
        ):
            x, f, n = reliability(y, p)
            ax.plot(x, f, marker="o", color=col, label=lab)
        ax.set_title(latex.plain(tag), fontsize=10)
        ax.set_xlabel("P(fake) prédite")
        ax.set_ylabel("proportion réelle de fake")
        ax.legend(fontsize=7)
    plt.tight_layout()
    latex.save_figure(fig, "calibration")
    plt.close(fig)
    write_latex(results)


def write_latex(results):
    rows = []
    for tag, r in results.items():
        rows.append(
            [
                latex.name(tag),
                latex.fr(r["temperature"], 2),
                f"{latex.fr(r['before']['ece'], 4)} $\\to$ {latex.fr(r['after']['ece'], 4)}",
                f"{latex.fr(r['before']['nll'], 4)} $\\to$ {latex.fr(r['after']['nll'], 4)}",
                f"{latex.fr(r['before']['brier'], 4)} $\\to$ {latex.fr(r['after']['brier'], 4)}",
            ]
        )
    body = latex.table(
        ["Modèle", "T", "ECE", "NLL", "Brier"],
        rows,
        "Calibration sur le jeu de test, avant $\\to$ après temperature scaling (T apprise sur la "
        "validation). "
        "Plus c'est bas, mieux c'est. Les prédictions (et donc l'exactitude et l'AUC) sont inchangées.",
    )
    over = [t for t, r in results.items() if r["temperature"] > 1]
    if over:
        body += (
            "Une température supérieure à 1 signifie que le modèle est \\emph{trop sûr de lui} : ses "
            "probabilités "
            "brutes sont trop proches de 0 ou de 1. C'est le cas de "
            + ", ".join(latex.name(t) for t in over)
            + ". "
        )
    body += (
        "La température ne modifie que les probabilités affichées : elle peut être ajoutée à "
        "\\texttt{models/metadata.json} et appliquée par l'API avant le softmax.\n"
    )
    latex.write("calibration", body)


if __name__ == "__main__":
    main()
