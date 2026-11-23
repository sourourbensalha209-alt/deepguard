"""Rigueur statistique : intervalles de confiance (bootstrap), test de McNemar, variabilité entre graines.

python -m src.analysis.stats   ->  metrics/stats.json, docs/rapport/generated/stats.tex

Répond à la limite « un seul entraînement par modèle » :
- l'intervalle de confiance bootstrap mesure l'incertitude due à la taille du jeu de test (4 000 images) ;
- le test de McNemar vérifie que l'écart entre deux modèles, évalués sur les MÊMES images, n'est pas
  dû au hasard ;
- plusieurs graines (si entraînées) mesurent la variabilité due à l'entraînement lui-même.
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

from src.analysis import latex
from src.analysis.runs import MAIN_SEED, list_runs, load_json, load_preds, main_run


def _fast_metrics(y_fake, p, idx):
    """Exactitude, F1 et AUC pour chaque ré-échantillon (idx : n_boot x n)."""
    yt, pt = y_fake[idx], p[idx]
    pred = (pt >= 0.5).astype(int)
    acc = (pred == yt).mean(axis=1)
    tp = ((pred == 1) & (yt == 1)).sum(axis=1)
    fp = ((pred == 1) & (yt == 0)).sum(axis=1)
    fn = ((pred == 0) & (yt == 1)).sum(axis=1)
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    auc = np.array([roc_auc_score(a, b) if 0 < a.sum() < len(a) else np.nan for a, b in zip(yt, pt)])
    return {"accuracy": acc, "f1": f1, "roc_auc": auc}


def bootstrap_ci(y, p_fake, n_boot=2000, alpha=0.05, seed=0):
    """IC bootstrap percentile à 95 %. y : labels (0 = fake), p_fake : P(fake)."""
    y_fake = (np.asarray(y) == 0).astype(int)
    p = np.asarray(p_fake, dtype=np.float64)
    n = len(y_fake)
    point = _fast_metrics(y_fake, p, np.arange(n)[None, :])
    rng = np.random.default_rng(seed)
    out = {}
    for start in range(0, n_boot, 250):  # par paquets pour limiter la mémoire
        idx = rng.integers(0, n, size=(min(250, n_boot - start), n))
        for k, v in _fast_metrics(y_fake, p, idx).items():
            out.setdefault(k, []).append(v)
    res = {}
    for k, chunks in out.items():
        vals = np.concatenate(chunks)
        lo, hi = np.nanpercentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        res[k] = {"value": float(point[k][0]), "low": float(lo), "high": float(hi)}
    return res


def mcnemar(y, p_a, p_b):
    """Test exact de McNemar : les deux modèles se trompent-ils sur des images différentes par hasard ?

    b = images où seul A a raison, c = images où seul B a raison. Sous H0 (même taux d'erreur),
    b suit une loi binomiale(b + c, 1/2).
    """
    y_fake = (np.asarray(y) == 0).astype(int)
    ok_a = (np.asarray(p_a) >= 0.5).astype(int) == y_fake
    ok_b = (np.asarray(p_b) >= 0.5).astype(int) == y_fake
    b, c = int((ok_a & ~ok_b).sum()), int((~ok_a & ok_b).sum())
    p_value = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return {"only_a_correct": b, "only_b_correct": c, "p_value": float(p_value)}


def _fmt_p(p):
    if p < 1e-4:
        return "$<10^{-4}$"
    return latex.fr(p, 4)


def main(n_boot=2000):
    runs = list_runs()
    if not runs:
        raise SystemExit(
            "Aucun résultat dans reports/runs/ : lance d'abord les cellules d'entraînement du notebook."
        )
    result = {"bootstrap": {}, "mcnemar": {}, "seeds": {}}

    # 1. IC bootstrap (exécution principale de chaque modèle)
    preds = {}
    for tag in runs:
        df = load_preds(main_run(runs, tag)).sort_values("path").reset_index(drop=True)
        preds[tag] = df
        result["bootstrap"][tag] = bootstrap_ci(df["label"], df["p_fake"], n_boot=n_boot)

    # 2. McNemar : le meilleur modèle contre chacun des autres, sur les mêmes images
    best = max(preds, key=lambda t: result["bootstrap"][t]["accuracy"]["value"])
    for tag, df in preds.items():
        if tag == best:
            continue
        ref = preds[best]
        if not ref["path"].equals(df["path"]):
            print(f"[info] {tag} : jeu de test différent de {best}, McNemar ignoré.")
            continue
        result["mcnemar"][f"{best}_vs_{tag}"] = mcnemar(ref["label"], ref["p_fake"], df["p_fake"])

    # 3. Variabilité entre graines (si plusieurs graines ont été entraînées)
    for tag, seeds in runs.items():
        if len(seeds) < 2:
            continue
        rows = []
        for s, folder in sorted(seeds.items()):
            test = load_json(folder, "test.json")
            if test:
                rows.append({"seed": s, **{k: test[f"test_{k}"] for k in ("accuracy", "f1", "roc_auc")}})
        if len(rows) >= 2:
            arr = {k: np.array([r[k] for r in rows]) for k in ("accuracy", "f1", "roc_auc")}
            result["seeds"][tag] = {
                "seeds": [r["seed"] for r in rows],
                **{k: {"mean": float(v.mean()), "std": float(v.std(ddof=1))} for k, v in arr.items()},
            }

    Path("metrics").mkdir(exist_ok=True)
    Path("metrics/stats.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    write_latex(result, best, n_boot)


def write_latex(result, best, n_boot):
    def ci(m):
        return f"{latex.fr(m['value'])} [{latex.fr(m['low'])} ; {latex.fr(m['high'])}]"

    rows = [
        [latex.name(t), ci(b["accuracy"]), ci(b["f1"]), ci(b["roc_auc"])]
        for t, b in result["bootstrap"].items()
    ]
    body = latex.table(
        ["Modèle", "Exactitude [IC 95\\,\\%]", "F1 [IC 95\\,\\%]", "AUC [IC 95\\,\\%]"],
        rows,
        f"Métriques sur le jeu de test avec intervalles de confiance à 95\\,\\% (bootstrap percentile, "
        f"{n_boot} "
        f"ré-échantillonnages des 4\\,000 images, graine d'entraînement {MAIN_SEED}).",
    )
    if result["mcnemar"]:
        rows = []
        for key, m in result["mcnemar"].items():
            other = key.split("_vs_")[1]
            rows.append(
                [
                    f"{latex.name(best)} contre {latex.name(other)}",
                    m["only_a_correct"],
                    m["only_b_correct"],
                    _fmt_p(m["p_value"]),
                ]
            )
        body += latex.table(
            ["Comparaison (A contre B)", "Seul A a raison", "Seul B a raison", "p (McNemar exact)"],
            rows,
            "Test de McNemar sur les mêmes images de test : l'écart entre deux modèles est significatif si "
            "$p<0{,}05$.",
        )
        sig = [k.split("_vs_")[1] for k, m in result["mcnemar"].items() if m["p_value"] < 0.05]
        nonsig = [k.split("_vs_")[1] for k, m in result["mcnemar"].items() if m["p_value"] >= 0.05]
        if sig:
            body += (
                f"L'avantage de {latex.name(best)} sur "
                + ", ".join(latex.name(t) for t in sig)
                + " est statistiquement significatif (McNemar, $p<0{,}05$) : "
                "il ne s'explique pas par le hasard "
                "de l'échantillon de test. "
            )
        if nonsig:
            body += (
                "En revanche, l'écart avec "
                + ", ".join(latex.name(t) for t in nonsig)
                + " n'est pas significatif : les deux modèles sont équivalents sur ce jeu de test. "
            )
        body += "\n\n"
    if result["seeds"]:
        rows = []
        for tag, s in result["seeds"].items():
            rows.append(
                [
                    latex.name(tag),
                    ", ".join(map(str, s["seeds"])),
                    *(
                        f"{latex.fr(s[k]['mean'])} $\\pm$ {latex.fr(s[k]['std'])}"
                        for k in ("accuracy", "f1", "roc_auc")
                    ),
                ]
            )
        body += latex.table(
            ["Modèle", "Graines", "Exactitude", "F1", "AUC"],
            rows,
            "Variabilité entre entraînements : moyenne $\\pm$ écart-type sur le jeu de test pour plusieurs "
            "graines.",
        )
    latex.write("stats", body)


if __name__ == "__main__":
    main()
