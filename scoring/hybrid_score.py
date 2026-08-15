"""
Calcule le score hybride final = combinaison ponderee de :
  - grille experte (regles metier / ratios)
  - RandomForest (LOOCV)
  - Regression logistique (LOOCV)
puis compare a la notation MNS2 reelle (notation_lettre) pour flaguer les dossiers divergents.
Ecrit le resultat dans la table scores_calcules de credit_scoring.db.
"""
import os
import sqlite3
import numpy as np
import pandas as pd

from expert_grille import score_grille_experte
from ml_pipeline import run_loocv, LETTRE_TO_SCORE

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, 'db', 'credit_scoring.db')

W_GRILLE, W_RF, W_LOGREG = 0.30, 0.35, 0.35


def score_to_lettre(score):
    if score is None or np.isnan(score):
        return None
    bornes = [(90, 'A'), (70, 'B'), (50, 'C'), (30, 'D'), (0, 'E')]
    for seuil, lettre in bornes:
        if score >= seuil:
            return lettre
    return 'E'


def main():
    conn = sqlite3.connect(DB_PATH)

    # variables en format large pour la grille experte
    vv = pd.read_sql("SELECT dossier_id, variable, valeur_num FROM variables_valeurs", conn)
    wide = vv.pivot(index='dossier_id', columns='variable', values='valeur_num')

    ml_results, feats, n = run_loocv()

    notations = pd.read_sql("SELECT dossier_id, notation_lettre FROM notations", conn).set_index('dossier_id')

    rows = []
    for dossier_id in wide.index:
        row_dict = wide.loc[dossier_id].to_dict()
        grille = score_grille_experte(row_dict)

        ml = ml_results.get(dossier_id, {})
        rf_score = ml.get('rf_score_0_100')
        lr_score = ml.get('logreg_score_0_100')

        parts, poids = [], []
        if grille is not None:
            parts.append(grille); poids.append(W_GRILLE)
        if rf_score is not None:
            parts.append(rf_score); poids.append(W_RF)
        if lr_score is not None:
            parts.append(lr_score); poids.append(W_LOGREG)

        hybride = round(sum(p * w for p, w in zip(parts, poids)) / sum(poids), 1) if poids else None
        notation_predite = score_to_lettre(hybride)

        vraie_lettre = notations.loc[dossier_id, 'notation_lettre'] if dossier_id in notations.index else None
        divergence = 0
        if vraie_lettre and notation_predite:
            ecart = abs(LETTRE_TO_SCORE.get(vraie_lettre, 50) - LETTRE_TO_SCORE.get(notation_predite, 50))
            divergence = 1 if ecart >= 40 else 0  # >= 2 crans de notation -> a investiguer

        rows.append((dossier_id, grille, rf_score, lr_score, hybride, notation_predite, divergence))

    conn.execute("DELETE FROM scores_calcules")
    conn.executemany(
        "INSERT INTO scores_calcules (dossier_id, score_grille_expert, proba_rf_defaut, "
        "proba_logreg_defaut, score_hybride, notation_predite, divergence_avec_mns2) "
        "VALUES (?,?,?,?,?,?,?)", rows)
    conn.commit()

    df_out = pd.read_sql("""
        SELECT s.dossier_id, i.nom_societe, n.notation_lettre AS mns2_reel,
               s.score_grille_expert, s.proba_rf_defaut AS score_rf, s.proba_logreg_defaut AS score_logreg,
               s.score_hybride, s.notation_predite, s.divergence_avec_mns2
        FROM scores_calcules s
        JOIN dossiers d ON d.dossier_id = s.dossier_id
        LEFT JOIN identites i ON i.dossier_id = s.dossier_id
        LEFT JOIN notations n ON n.dossier_id = s.dossier_id
        ORDER BY s.divergence_avec_mns2 DESC, s.dossier_id
    """, conn)
    conn.close()
    print(df_out.to_string(index=False))
    return df_out


if __name__ == '__main__':
    main()
