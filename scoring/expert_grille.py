"""
Grille experte : reconstruit un score 0-100 a partir de ratios financiers cles,
inspiree de la logique MNS2 (quantitatif = structure financiere / rentabilite / liquidite,
qualitatif = deja capture par score_qualitatif quand disponible).

ATTENTION : les ponderations ci-dessous sont une premiere hypothese de travail a valider
avec le maitre de stage (Centre d'Affaires Al Istiqlal) - elles ne reproduisent pas
la grille interne exacte de la Banque Populaire, qui n'est pas documentee dans les
dossiers sources.
"""
import numpy as np

# ratio -> (poids, sens: 'plus_est_mieux' ou 'moins_est_mieux', bornes de normalisation [min,max])
RATIOS_PONDERATION = {
    'rentabilite_finaciere':      (0.20, 'plus_est_mieux', (-20, 40)),
    'autonom_financiere':         (0.20, 'plus_est_mieux', (0, 100)),
    'endettement_a_court_terme':  (0.15, 'moins_est_mieux', (0, 100)),
    'ratios_liquidite_reduite':   (0.15, 'plus_est_mieux', (0, 200)),
    'capacite_e_remboursement':   (0.15, 'plus_est_mieux', (0, 10)),
    'dettes_nettes_fonds_propres':(0.15, 'moins_est_mieux', (0, 5)),
}


def _normalize(value, borne_min, borne_max, sens):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    v = max(min(value, borne_max), borne_min)
    score = (v - borne_min) / (borne_max - borne_min) * 100
    if sens == 'moins_est_mieux':
        score = 100 - score
    return score


def score_grille_experte(row: dict) -> float | None:
    """row: dict variable_canonique -> valeur (issu de variables_valeurs pour un dossier).
    Retourne un score 0-100, ou None si aucune variable dispo (pas de reponse forcee)."""
    total_poids = 0.0
    total_score = 0.0
    for var, (poids, sens, (bmin, bmax)) in RATIOS_PONDERATION.items():
        val = row.get(var)
        s = _normalize(val, bmin, bmax, sens)
        if s is None:
            continue
        total_score += poids * s
        total_poids += poids
    if total_poids == 0:
        return None
    return round(total_score / total_poids, 2)
