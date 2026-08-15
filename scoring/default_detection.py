"""
Systeme de decision credit — detection defaut / non-defaut.

Cible binaire : le seuil de defaut n'est PAS fige en dur. Il est recalcule automatiquement
a partir des donnees disponibles (clustering non supervise sur score_final) a chaque execution
du pipeline. Ainsi, quand de nouveaux dossiers seront ajoutes via le site (workflow analyste ->
direction), le seuil s'adapte automatiquement a la distribution reelle des scores, au lieu de
rester bloque sur une valeur choisie une fois pour toutes sur un tout petit echantillon.

Methode : KMeans a 2 clusters sur score_final -> le cluster de score le plus bas = "defaut".
Le seuil affiche est le point median entre les deux centres de cluster. Avec un echantillon
tres restreint (n<4 valeurs distinctes), on retombe sur un partage median (fallback).

Modeles de classification : Random Forest + Regression Logistique, evalues en LOOCV (seul
protocole viable avec un tres petit n). Sortie : probabilite de defaut par dossier + importance
des variables (MDI moyenne + importance par permutation sur le modele final).

Toutes les fonctions travaillent uniquement sur dossier_id (jamais nom_societe) : le module
ne touche jamais a la table identites.
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import LeaveOneOut
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

FEATURES_CANDIDATES = [
    'autre_charges_ext', 'capacite_e_remboursement', 'capital_propres', 'actif_circulant',
    'fond_de_roulement', 'actif_net_compta', 'consomm_exercice', 'immob_corporelles',
    'resulat_avt_impot', 'concentration_fournisseur', 'bque_caisse_ccp',
    'tresorerie_actif', 'charge_personnel', 'rentabilite_finaciere', 'total_passif', 'clients',
    'divers', 'materiel_agence', 'resutats_exploit', 'finct_permanent', 'capital_soc_perso',
    'valeur_ajoutee', 'dotaion_exploit', 'experience_pdg', 'rentabilite_globale',
    'actif_immobilise_net', 'dettes_nettes_fonds_propres', 'tresorerie_nette',
    'ammort_provision', 'ratios_liquidite_reduite', 'endettement_a_court_terme', 'total_actif',
    'reserves', 'autonom_financiere',
]


def load_dataset():
    return pd.read_pickle(os.path.join(DATA_DIR, 'clean_df.pkl'))


def compute_dynamic_threshold(scores: pd.Series):
    """Trouve automatiquement la coupure sain/defaut dans score_final via KMeans (k=2).
    Retourne (seuil, labels_binaires_alignes_sur_scores.index)."""
    valid = scores.dropna()
    if valid.nunique() < 4:
        seuil = valid.median()
        return seuil, (scores < seuil)

    X = valid.values.reshape(-1, 1)
    km = KMeans(n_clusters=2, n_init=10, random_state=42).fit(X)
    centers = sorted(km.cluster_centers_.flatten())
    seuil = (centers[0] + centers[1]) / 2
    return seuil, (scores < seuil)


def build_target(df):
    y = pd.to_numeric(df['score_final'], errors='coerce')
    seuil, is_defaut = compute_dynamic_threshold(y)
    label = is_defaut.astype('Int64')
    label[y.isna()] = pd.NA
    return label, seuil


def run_default_detection():
    df = load_dataset()
    df = df.copy()
    df['label_defaut'], seuil_defaut = build_target(df)

    feats = [f for f in FEATURES_CANDIDATES if f in df.columns]
    X_all = df[feats].apply(pd.to_numeric, errors='coerce')
    feats = [f for f in feats if X_all[f].notna().sum() > 0]
    X_all = X_all[feats]

    has_label = df['label_defaut'].notna()
    dossier_ids = df['dossier_id'].tolist()

    proba_out = {d: {'rf': None, 'logreg': None, 'label': None} for d in dossier_ids}
    for d, lab in zip(dossier_ids, df['label_defaut']):
        proba_out[d]['label'] = None if pd.isna(lab) else int(lab)

    X_lab = X_all[has_label.values].reset_index(drop=True)
    y_lab = df.loc[has_label, 'label_defaut'].astype(int).reset_index(drop=True)
    ids_lab = df.loc[has_label, 'dossier_id'].reset_index(drop=True)

    rf_importances = []

    if y_lab.nunique() < 2:
        # pas assez de variabilite (ex: tous sains ou tous en defaut) -> pas de LOOCV possible
        return proba_out, feats, [], len(y_lab), seuil_defaut

    loo = LeaveOneOut()
    for train_idx, test_idx in loo.split(X_lab):
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()
        X_train = scaler.fit_transform(imputer.fit_transform(X_lab.iloc[train_idx]))
        X_test = scaler.transform(imputer.transform(X_lab.iloc[test_idx]))
        y_train = y_lab.iloc[train_idx]

        rf = RandomForestClassifier(n_estimators=300, max_depth=4, random_state=42, class_weight='balanced')
        rf.fit(X_train, y_train)
        rf_importances.append(rf.feature_importances_)
        rf_proba = rf.predict_proba(X_test)[0]
        classes = list(rf.classes_)
        p_defaut_rf = rf_proba[classes.index(1)] if 1 in classes else 0.0

        try:
            lr = LogisticRegression(max_iter=2000)
            lr.fit(X_train, y_train)
            lr_proba = lr.predict_proba(X_test)[0]
            lr_classes = list(lr.classes_)
            p_defaut_lr = lr_proba[lr_classes.index(1)] if 1 in lr_classes else 0.0
        except Exception:
            p_defaut_lr = None

        d = ids_lab.iloc[test_idx[0]]
        proba_out[d]['rf'] = round(float(p_defaut_rf), 3)
        proba_out[d]['logreg'] = round(float(p_defaut_lr), 3) if p_defaut_lr is not None else None

    # Feature importance globale : modele final entraine sur tout l'echantillon labellise
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    X_full = scaler.fit_transform(imputer.fit_transform(X_lab))
    rf_full = RandomForestClassifier(n_estimators=300, max_depth=4, random_state=42, class_weight='balanced')
    rf_full.fit(X_full, y_lab)

    mdi = rf_full.feature_importances_
    try:
        perm = permutation_importance(rf_full, X_full, y_lab, n_repeats=30, random_state=42)
        perm_imp = perm.importances_mean
    except Exception:
        perm_imp = np.zeros_like(mdi)

    importance_table = sorted(
        zip(feats, mdi, perm_imp), key=lambda x: x[1], reverse=True
    )

    return proba_out, feats, importance_table, len(y_lab), seuil_defaut


if __name__ == '__main__':
    proba_out, feats, importance_table, n, seuil = run_default_detection()
    print(f"Seuil de defaut calcule dynamiquement (KMeans sur score_final) = {seuil:.1f}")
    print(f"n labellise = {n}, {len(feats)} variables")
    for d, v in proba_out.items():
        print(d, v)
    print("\nTop facteurs (importance RF) :")
    for var, mdi, perm in importance_table[:15]:
        print(f"  {var:35s} MDI={mdi:.3f}  permutation={perm:.3f}")
