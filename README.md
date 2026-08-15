# Système de Décision Crédit — Détection de Défaut (PFA)

PFA — Banque Populaire, Centre d'Affaires Al Istiqlal (Rabat-Kénitra)
GMSI — École Mohammadia d'Ingénieurs (EMI)

## Ce que fait le système

Pas juste un score à interpréter : une **décision explicite** (Accepté / Refusé / À étudier)
avec justification, pour chaque dossier de crédit, à partir :
- d'une **grille experte** (ratios financiers pondérés)
- d'un **Random Forest** + une **régression logistique** entraînés à détecter le défaut,
  évalués en **LOOCV** (seul protocole viable avec un échantillon aussi restreint)
- d'une **règle de décision logique** qui combine les deux et arbitre les divergences

Le tout appuyé sur un **workflow** Analyste → Direction avec **anonymisation** : le moteur de
décision et l'explicabilité ne travaillent jamais sur le nom de la société, seulement sur
l'identifiant de dossier. Le nom n'est visible que dans la vue Direction (habilitée à lever
l'anonymat pour trancher).

## Installation (100% offline)

Sur ta machine Windows, **avec wifi, avant d'aller au stage** :
```powershell
pip install -r requirements.txt
```
Utilise un chemin court (ex: `C:\PFA`) pour éviter la limite Windows de 260 caractères.

## Reconstruction du pipeline (base + scores + décisions)

```powershell
python run_all.py
```

Étapes exécutées dans l'ordre :
1. `data/clean_data.py` — nettoyage/fusion des 638 variables brutes → 339 canoniques
2. `db/build_database.py` — construction de `db/credit_scoring.db` (SQLite, anonymisée)
3. `scoring/hybrid_score.py` — reconstruction de la notation A-E (grille + RF + LogReg)
4. `scoring/decision_engine.py` — détection défaut/non-défaut + décision finale + feature importance

## Comptes et connexion

L'application démarre sur un écran de **connexion**. Deux comptes de démonstration sont
créés automatiquement par `db/build_database.py` :

| Identifiant | Mot de passe | Rôle |
|---|---|---|
| `analyste1` | `changer123` | Analyste |
| `direction1` | `changer123` | Direction |

⚠️ **À changer avant tout usage réel** — édite `scoring/auth.py` (fonction `create_user`
dans `db/build_database.py`) pour créer de vrais comptes avec des mots de passe personnels.
Les mots de passe ne sont jamais stockés en clair (hash PBKDF2-HMAC-SHA256 + sel aléatoire,
stdlib Python uniquement — aucune dépendance externe).

Un bouton **Déconnexion** est disponible en haut de l'application à tout moment.

## Créer un nouveau dossier (espace Analyste)

Dans l'onglet « 🆕 Créer un nouveau dossier », trois modes de dépôt sont proposés :
- **📷 Scanner (caméra)** — capture directe via la caméra de l'ordinateur (`st.camera_input`)
- **📎 Uploader un PDF** — dépôt d'un PDF existant, avec aperçu du texte extrait (aide-mémoire,
  pas de saisie automatique fiable sur des documents hétérogènes)
- **⌨️ Saisie manuelle** — remplissage direct du formulaire

Dans tous les cas, un formulaire structuré (Bilan actif / Bilan passif / Résultat / Trésorerie)
permet de saisir les variables financières. À la validation, le système génère un
**identifiant de dossier unique et sécurisé** (aléatoire, cryptographiquement sûr via le
module `secrets` de Python — pas un compteur prévisible), affiché à l'écran pour être noté et
recherché plus tard (onglet « 💬 Commentaire sur un dossier »).

## Lancer le dashboard

```powershell
cd dashboard
streamlit run app.py
```
Ouvre `http://localhost:8501`. Trois vues dans la barre latérale :
- **Analyste** : dépose un dossier + un commentaire de défense du client
- **Direction** : voit la décision du système + l'argumentaire, valide/tranche, génère le PV
- **Explicabilité** : facteurs déterminants (feature importance), décisions anonymisées, limites

### Thème visuel
Fond crème pastel, accents orange Banque Populaire (`#EE7203`), boutons avec effet de
survol animé, badges de décision colorés (vert/rouge/orange). Pour afficher le vrai logo :
dépose un fichier `logo_bp.png` dans `dashboard/assets/` (sinon un badge "BP" générique
s'affiche à la place). Tout le CSS est local — aucune police/CDN externe, donc ça fonctionne
100% hors ligne.

## Rapport — Chapitres 3 à 7 (LaTeX)

Dans `rapport_latex/` : code source LaTeX des Chapitres 3 à 7 (méthodologie, score expert,
score ML, décision hybride, discussion), dans le même style que tes Chapitres 1-2 existants
(boîtes "Définition" bleu marine), plus une nouvelle boîte "Limite méthodologique" en orange
pastel. **Tous les chiffres cités sont les vrais résultats du pipeline** sur tes 9 dossiers.
Un PDF déjà compilé (`main.pdf`) est fourni pour relecture immédiate. Voir
`rapport_latex/README.md` pour l'intégration à ton rapport complet.

## Comment est calculée la cible "défaut"

Il n'y a pas d'historique réel d'incidents de paiement dans les données disponibles. Le système
utilise donc un **proxy calculé automatiquement** : un clustering (KMeans à 2 groupes) sur
`score_final` détermine la coupure naturelle sain/à risque dans la distribution réelle des
scores. **Ce seuil n'est pas figé** — il se recalcule à chaque exécution du pipeline, donc il
s'adapte automatiquement à mesure que de nouveaux dossiers sont ajoutés via le workflow.

⚠️ **Ceci est une hypothèse de travail à valider avec ta maître de stage.** Si la banque peut
fournir un historique réel d'incidents de paiement/défauts, il faut remplacer ce proxy par la
vraie variable (`scoring/default_detection.py`, fonction `build_target`).

## Règle de décision (`scoring/decision_engine.py`)

```
proba_hybride = moyenne(proba_RF, proba_LogReg)

si proba_hybride >= 60%                          -> REFUSÉ
si proba_hybride <= 25%                          -> ACCEPTÉ
   sauf si grille experte < 30/100 (divergence)   -> À ÉTUDIER
sinon (zone grise)                                -> À ÉTUDIER
```
Seuils à valider/ajuster avec le maître de stage — ce sont des hypothèses de travail explicites,
pas la politique de risque réelle de la banque.

## Structure du projet

```
PFA_Credit_Scoring/
├── data/
│   ├── base_donnees_fusionnee.xlsx   # données sources (9 dossiers, 638 variables brutes)
│   └── clean_data.py                 # nettoyage → clean_df.pkl + mapping.json
├── db/
│   ├── build_database.py             # construit credit_scoring.db (schéma anonymisé + comptes)
│   └── credit_scoring.db             # base SQLite (générée)
├── scoring/
│   ├── auth.py                       # authentification (hash+sel) + ID de dossier sécurisé
│   ├── expert_grille.py              # grille experte (ratios pondérés)
│   ├── ml_pipeline.py                # notation A-E : RF + LogReg, LOOCV
│   ├── hybrid_score.py               # combine grille+RF+LogReg → notation A-E
│   ├── default_detection.py          # cible défaut (seuil dynamique KMeans) + RF/LogReg + feature importance
│   └── decision_engine.py            # règle de décision finale (Accepté/Refusé/À étudier)
├── dashboard/
│   ├── app.py                        # dashboard Streamlit (Connexion / Accueil / Analyste / Direction / Explicabilité)
│   ├── assets/                       # logo_bp.png à déposer ici
│   └── uploads/                      # PDF/scans déposés par les analystes (généré à l'usage)
├── run_all.py                        # reconstruit tout le pipeline en une commande
└── requirements.txt
```

## Schéma de la base de données

- **dossiers** : 1 ligne par dossier (id, segment) — **anonyme**, pas de nom de société
- **identites** : dossier_id → nom de société — **table isolée**, jointe uniquement côté Direction
- **variables_dictionnaire** / **variables_valeurs** : 339 variables en format EAV (forte sparsité)
- **mapping_nettoyage** : traçabilité variable brute → canonique
- **notations** : notation MNS2 réelle reconstituée
- **scores_calcules** : sorties du pipeline hybride (notation A-E)
- **decision_ml** : probabilités de défaut (RF, LogReg, hybride), décision finale, justification
- **feature_importance** : facteurs déterminants (MDI + permutation)
- **parametres_systeme** : seuil de défaut calculé dynamiquement (traçable, horodaté)
- **soumissions** : dossiers déposés par les analystes + commentaire de défense
- **decisions_direction** : décisions finales de la direction (source du PV)
- **dossiers_prioritaires** : dossiers flagués pour investigation qualitative
- **utilisateurs** : comptes locaux (identifiant, rôle, mot de passe hashé+salé)
- **documents_dossier** : pièces jointes (scan caméra / PDF uploadé) liées à chaque dossier

## ⚠️ Points à valider avec le maître de stage avant la soutenance

1. **Cible "défaut"** : proxy statistique (KMeans sur score_final), pas un historique réel — à documenter comme limite explicite dans le rapport (cf. Chapitre 1.4 "Limites assumées").
2. **n=7 dossiers labellisés** : le LOOCV donne une estimation très bruitée — à présenter comme indicatif, jamais comme validation statistique robuste (cf. Chapitre 2.3 du rapport).
3. **Segments et noms de société** (`segment_a_verifier` dans `dossiers`, table `identites`) sont devinés automatiquement — à corriger dossier par dossier.
4. **Pondérations de la grille experte** et **seuils de décision** (60%/25%) sont des hypothèses de travail, pas la politique de risque réelle de la banque.
5. **Dossier Feuil4** n'a pas de score_final identifié dans les données fusionnées.
6. Certaines variables ont plusieurs valeurs sous la même étiquette brute (`__dup2`/`__dup3`, probablement N-2/N-1/N mal étiquetés à la source) — seule la première valeur non-nulle est actuellement retenue.
#   P F A _ c r e d i t _ s c o r i n g  
 