# Rapport technique

`rapport_technique.tex` : rapport LaTeX (Partie A Deep Learning, Partie B MLOps). Compilation : `pdflatex rapport_technique.tex`
deux fois, ou Overleaf (importer tout le dossier `docs/rapport/`, `logo.png` et `figures/` compris).

## Déjà rempli
Résultats des trois modèles (chiffres du run Kaggle final), matrice de confusion, courbe ROC, cartes d'explication (ViT et
EfficientNet), optimisation ONNX / INT8, test de robustesse, analyse critique, limites, conclusion.

## Reste à compléter (éléments `[à compléter : ...]` en rouge)
| Figure à déposer dans `figures/` | Source |
|---|---|
| `dvc-dag.png` | capture de `dvc dag` |
| `mlflow-runs.png`, `mlflow-registry.png` | captures de l'interface MLflow (comparaison des runs, registre) |
| `github-actions.png` | capture de l'onglet Actions (workflows verts) |

Textes à ajuster : la répartition des tâches, le test de charge Locust, et la relecture de l'analyse critique.
Le PDF final se copie à la racine du dépôt sous le nom `rapport_technique.pdf`.
