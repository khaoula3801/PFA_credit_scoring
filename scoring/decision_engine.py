"""
Moteur de decision : transforme les scores (grille experte + RF + LogReg) en UNE decision
explicite et justifiee, pas juste un chiffre a interpreter. C'est la difference entre un
systeme d'aide a la decision (la banque aujourd'hui) et un systeme de decision (ce projet).

Regle de decision (hypothese de travail, a valider avec le maitre de stage) :
  - proba_defaut_hybride = moyenne(proba_rf, proba_logreg) si les deux dispo, sinon celle dispo
  - Si proba_defaut_hybride >= 0.60           -> REFUSE
  - Si proba_defaut_hybride <= 0.25           -> ACCEPTE
  - Sinon                                     -> A ETUDIER (zone grise, arbitrage direction)
  - Le score de la grille experte sert de garde-fou : si grille < 30/100 (tres faible) alors
    que le ML dit ACCEPTE, la decision est forcee a "A ETUDIER" (divergence expert/ML a
    examiner avant toute decision automatique).
"""
import os
import sqlite3
import pandas as pd

from expert_grille import score_grille_experte
from default_detection import run_default_detection

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, 'db', 'credit_scoring.db')

SEUIL_REFUS = 0.60
SEUIL_ACCEPT = 0.25
SEUIL_GRILLE_ALERTE = 30


def decide(proba_hybride, grille_score):
    if proba_hybride is None:
        return "A ETUDIER", "Probabilite de defaut non calculable (donnees insuffisantes pour ce dossier)."

    if proba_hybride >= SEUIL_REFUS:
        return "REFUSE", f"Probabilite de defaut estimee a {proba_hybride:.0%} (>= seuil de refus {SEUIL_REFUS:.0%})."

    if proba_hybride <= SEUIL_ACCEPT:
        if grille_score is not None and grille_score < SEUIL_GRILLE_ALERTE:
            return ("A ETUDIER",
                    f"ML favorable (defaut {proba_hybride:.0%}) mais grille experte tres faible "
                    f"({grille_score:.0f}/100) -> divergence a examiner avant decision automatique.")
        return "ACCEPTE", f"Probabilite de defaut estimee a {proba_hybride:.0%} (<= seuil d'acceptation {SEUIL_ACCEPT:.0%})."

    return "A ETUDIER", f"Probabilite de defaut estimee a {proba_hybride:.0%} (zone grise, arbitrage direction requis)."


def main():
    conn = sqlite3.connect(DB_PATH)

    vv = pd.read_sql("SELECT dossier_id, variable, valeur_num FROM variables_valeurs", conn)
    wide = vv.pivot(index='dossier_id', columns='variable', values='valeur_num')

    proba_out, feats, importance_table, n_labels, seuil_defaut = run_default_detection()

    rows = []
    for dossier_id in wide.index:
        row_dict = wide.loc[dossier_id].to_dict()
        grille = score_grille_experte(row_dict)

        ml = proba_out.get(dossier_id, {})
        p_rf, p_lr, label = ml.get('rf'), ml.get('logreg'), ml.get('label')

        candidates = [p for p in (p_rf, p_lr) if p is not None]
        p_hybride = round(sum(candidates) / len(candidates), 3) if candidates else None

        decision, justification = decide(p_hybride, grille)
        rows.append((dossier_id, p_rf, p_lr, p_hybride, label, decision, justification))

    conn.execute("DELETE FROM decision_ml")
    conn.executemany(
        "INSERT INTO decision_ml (dossier_id, proba_defaut_rf, proba_defaut_logreg, "
        "proba_defaut_hybride, label_reel_defaut, decision_systeme, justification) "
        "VALUES (?,?,?,?,?,?,?)", rows)

    conn.execute("DELETE FROM feature_importance")
    conn.executemany(
        "INSERT INTO feature_importance (variable, importance_rf, importance_permutation, rang) VALUES (?,?,?,?)",
        [(var, float(mdi), float(perm), i + 1) for i, (var, mdi, perm) in enumerate(importance_table)]
    )
    conn.execute("INSERT OR REPLACE INTO parametres_systeme (cle, valeur) VALUES ('seuil_defaut_score_final', ?)",
                 (str(round(seuil_defaut, 1)),))
    conn.commit()

    out = pd.read_sql("""
        SELECT dm.dossier_id, dm.proba_defaut_rf, dm.proba_defaut_logreg, dm.proba_defaut_hybride,
               dm.label_reel_defaut, dm.decision_systeme, dm.justification
        FROM decision_ml dm ORDER BY dm.proba_defaut_hybride DESC
    """, conn)
    conn.close()

    print(f"Cible : defaut = score_final < seuil dynamique ({seuil_defaut:.1f}, calcule par KMeans) | n labellise = {n_labels}")
    print(out.to_string(index=False))
    print("\nTop 10 facteurs determinants (importance RF) :")
    for var, mdi, perm in importance_table[:10]:
        print(f"  {var:35s} MDI={mdi:.3f}  permutation={perm:.3f}")
    return out


if __name__ == '__main__':
    main()
