# Chapitres 3 à 7 — LaTeX

## Compiler seul (pour relecture)
```
pdflatex main.tex
pdflatex main.tex   # deux passages pour la table des matières
```
`main.pdf` est déjà fourni, compilé et vérifié (18 pages, 0 erreur).

Si `pdflatex` râle sur `Unknown option 'french'` : le pack langue française de
TeX Live manque → `sudo apt install texlive-lang-french` (Linux) ou installer le
pack "French" via le gestionnaire de MiKTeX (Windows).

## Intégrer à ton rapport complet (avec couverture, Chapitres 1-2 déjà rédigés)

1. Copie le contenu de `preambule.tex` dans le préambule de ton document principal
   (fusionne avec tes packages existants s'il y a des doublons).
2. Colle le contenu de chaque `chapXX_*.tex` à la suite de ton Chapitre 2 actuel,
   dans l'ordre : 3, 4, 5, 6, 7.
3. Les compteurs de chapitre continuent automatiquement (`\chapter{...}` incrémente
   depuis ton Chapitre 2 existant) — pas besoin de forcer la numérotation.
4. Les environnements `definition` et `limite` sont nouveaux par rapport à tes
   Chapitres 1-2 (qui utilisent déjà "Définition X.X") : vérifie que le style
   visuel (boîte bleu marine) correspond bien à celui que tu as utilisé
   au Chapitre 2 — sinon ajuste les couleurs `bpnavy`/`bporange` dans
   `preambule.tex`.

## Ce que contient chaque chapitre

| Fichier | Contenu |
|---|---|
| `chap3_methodologie.tex` | Sources de données, nettoyage 638→339 variables, anonymisation, schéma BDD |
| `chap4_score_expert.tex` | Grille experte, pondérations, résultats par dossier |
| `chap5_score_ml.tex` | Seuil de défaut dynamique (K-means), RF+LogReg LOOCV, feature importance |
| `chap6_decision_hybride.tex` | Score hybride A-E, moteur de décision, confrontation MNS2, workflow |
| `chap7_discussion.tex` | Synthèse, limites hiérarchisées, recommandations, conclusion |

Tous les chiffres cités (scores, probabilités, feature importance) sont **les vrais
résultats** produits par `scoring/decision_engine.py` sur tes 9 dossiers réels —
à re-générer si tu relances le pipeline avec des données mises à jour.
