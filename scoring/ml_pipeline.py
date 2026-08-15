"""
Pipeline ML hybride : Random Forest + Regression Logistique, evalues en LOOCV
(Leave-One-Out) car seulement 9 dossiers -> impossible de faire un vrai split train/test.

Cible : notation_lettre (A-E) recodee en score ordinal 0-100 (A=100 ... E=20)
pour permettre a la fois classification (lettre predite) et un score continu
utilisable dans le score hybride.

IMPORTANT (limite methodologique a mentionner dans le rapport) : avec n=9, le LOOCV donne
une estimation tres bruitee de la performance reelle. Les resultats doivent etre lus comme
indicatifs, pas comme une validation statistique robuste.
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

LETTRE_TO_SCORE = {'A': 100, 'B': 80, 'C': 60, 'D': 40, 'E': 20}

# variables retenues : celles presentes dans >= 6/9 dossiers ET numeriques (cf. clean_data.py)
FEATURES_CANDIDATES = [
    'autre_charges_ext', 'capacite_e_remboursement', 'capital_propres', 'actif_circulant',
    'fond_de_roulement', 'actif_net_compta', 'consomm_exercice', 'immob_corporelles',
    'taux_defaut_secteur', 'resulat_avt_impot', 'concentration_fournisseur', 'bque_caisse_ccp',
    'tresorerie_actif', 'charge_personnel', 'rentabilite_finaciere', 'total_passif', 'clients',
    'divers', 'materiel_agence', 'resutats_exploit', 'finct_permanent', 'capital_soc_perso',
    'valeur_ajoutee', 'dotaion_exploit', 'experience_pdg', 'rentabilite_globale',
    'actif_immobilise_net', 'dettes_nettes_fonds_propres', 'tresorerie_nette',
    'ammort_provision', 'ratios_liquidite_reduite', 'endettement_a_court_terme', 'total_actif',
    'reserves', 'autonom_financiere',
]


def load_dataset():
    clean_df = pd.read_pickle(os.path.join(DATA_DIR, 'clean_df.pkl'))
    notation_candidates = ['note_finale', 'note_calculee', 'notation', 'notation_2024_affaire', 'notation_projet']
    lettre = []
    for _, r in clean_df.iterrows():
        val = None
        for c in notation_candidates:
            if c in clean_df.columns and pd.notna(r[c]):
                val = str(r[c]).strip().upper()
                break
        lettre.append(val)
    clean_df = clean_df.copy()
    clean_df['notation_lettre'] = lettre
    return clean_df


def run_loocv():
    df = load_dataset()
    df = df[df['notation_lettre'].notna()].reset_index(drop=True)  # Feuil4 exclu (pas de notation connue)
    feats = [f for f in FEATURES_CANDIDATES if f in df.columns]
    X_raw = df[feats].apply(pd.to_numeric, errors='coerce')
    feats = [f for f in feats if X_raw[f].notna().sum() > 0]
    X_raw = X_raw[feats]
    y_lettre = df['notation_lettre']
    y_score = y_lettre.map(LETTRE_TO_SCORE)

    dossier_ids = df['dossier_id'].tolist()
    n = len(df)

    results = {d: {} for d in dossier_ids}
    loo = LeaveOneOut()
    for train_idx, test_idx in loo.split(X_raw):
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()

        X_train = imputer.fit_transform(X_raw.iloc[train_idx])
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(imputer.transform(X_raw.iloc[test_idx]))

        y_train_lettre = y_lettre.iloc[train_idx]

        rf = RandomForestClassifier(n_estimators=200, max_depth=4, random_state=42, class_weight='balanced')
        rf.fit(X_train, y_train_lettre)
        rf_pred = rf.predict(X_test)[0]
        rf_classes = list(rf.classes_)
        rf_proba = rf.predict_proba(X_test)[0]
        rf_score = float(np.dot(rf_proba, [LETTRE_TO_SCORE.get(c, 50) for c in rf_classes]))

        try:
            logreg = LogisticRegression(max_iter=2000)
            logreg.fit(X_train, y_train_lettre)
            lr_pred = logreg.predict(X_test)[0]
            lr_classes = list(logreg.classes_)
            lr_proba = logreg.predict_proba(X_test)[0]
            lr_score = float(np.dot(lr_proba, [LETTRE_TO_SCORE.get(c, 50) for c in lr_classes]))
        except Exception:
            lr_pred, lr_score = None, None

        d = dossier_ids[test_idx[0]]
        results[d] = {
            'rf_pred_lettre': rf_pred, 'rf_score_0_100': round(rf_score, 1),
            'logreg_pred_lettre': lr_pred, 'logreg_score_0_100': round(lr_score, 1) if lr_score is not None else None,
            'vraie_lettre': y_lettre.iloc[test_idx[0]],
        }

    return results, feats, n


if __name__ == '__main__':
    results, feats, n = run_loocv()
    print(f"LOOCV sur n={n} dossiers, {len(feats)} variables")
    for d, r in results.items():
        print(d, r)
