"""
Nettoyage de base_donnees_fusionnee.xlsx -> base canonique propre
- Fusionne variables typo/doublons (Regroupement_propose) sauf cas ACTIF/PASSIF melanges (split manuel)
- Coalesce __dup2/__dup3 vers la variable primaire (garde 1re valeur non-nulle)
- Corrige l'echelle KDH/DH sur cash_flow (Feuil8 en DH -> /1000 pour ramener en KDH)
- Segment et nom_societe repris de "(a verifier)" -> a corriger manuellement avec le maitre de stage
"""
import pandas as pd, json, re, openpyxl

import os
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'base_donnees_fusionnee.xlsx')
wb = openpyxl.load_workbook(SRC, data_only=True)

# --- 1. Charger Base_fusionnee ---
ws = wb['Base_fusionnee']
rows = list(ws.iter_rows(values_only=True))
header = list(rows[0])
df = pd.DataFrame(rows[1:], columns=header)

DOSSIER_COL = 'source_feuille'
SOCIETE_COL = 'nom_societe (a verifier)'
SEGMENT_VERIF_COL = 'segment (a verifier)'
meta_cols = [DOSSIER_COL, SOCIETE_COL, SEGMENT_VERIF_COL]
raw_cols = [c for c in df.columns if c not in meta_cols]

# --- 2. Charger Regroupement_propose ---
ws2 = wb['Regroupement_propose']
rows2 = list(ws2.iter_rows(values_only=True))[1:]
groups = []
for r in rows2:
    if not r or not r[0]:
        continue
    canon, raws, warn = r
    groups.append({'canon': canon, 'raws': [x.strip() for x in raws.split(',')], 'warn': warn})

# --- 3. Construire mapping raw -> canon (gestion explicite des 3 groupes actif/passif melanges) ---
explicit_fix = {
    'actif_circulant': 'actif_circulant', 'l_actif_circulant': 'actif_circulant',
    'passif_circulant': 'passif_circulant', 'actif_circulant__dup2': 'actif_circulant',
    'tresorerie_actif': 'tresorerie_actif', 'tresorerie_passif': 'tresorerie_passif',
    'tresorerie_actif__dup2': 'tresorerie_actif',
    'total_passif__dup2': 'total_passif', 'total_actif__dup2': 'total_actif',
}
mapping = {}
for g in groups:
    if g['warn']:
        continue  # gere via explicit_fix
    for r in g['raws']:
        if r in df.columns:
            mapping[r] = g['canon']
mapping.update({k: v for k, v in explicit_fix.items() if k in df.columns})
for c in raw_cols:
    mapping.setdefault(c, c)

canon_to_raws = {}
for raw, canon in mapping.items():
    canon_to_raws.setdefault(canon, []).append(raw)

# --- 4. Coalesce (primaire d'abord, puis __dup2/__dup3) ---
clean_data = {}
for canon, raws in canon_to_raws.items():
    raws_sorted = sorted(raws, key=lambda x: ('__dup' in x, x))
    sub = df[raws_sorted]
    clean_data[canon] = sub.bfill(axis=1).iloc[:, 0]

clean_df = pd.DataFrame(clean_data)
clean_df.insert(0, 'segment_a_verifier', df[SEGMENT_VERIF_COL])
clean_df.insert(0, 'nom_societe_a_verifier', df[SOCIETE_COL])
clean_df.insert(0, 'dossier_id', df[DOSSIER_COL])

# --- 5. Correction d'echelle cash_flow (Feuil8 en DH -> KDH, coherent avec les autres) ---
if 'cash_flow' in clean_df.columns:
    mask = clean_df['dossier_id'] == 'Feuil8'
    clean_df.loc[mask, 'cash_flow'] = pd.to_numeric(clean_df.loc[mask, 'cash_flow'], errors='coerce') / 1000.0

# --- 6. Conversion numerique des colonnes majoritairement numeriques ---
def try_numeric(s):
    conv = pd.to_numeric(s, errors='coerce')
    n_ok = conv.notna().sum()
    n_nonnull_orig = s.notna().sum()
    if n_nonnull_orig > 0 and n_ok / n_nonnull_orig >= 0.7:
        return conv
    return s

for c in clean_df.columns:
    if c in ('dossier_id', 'nom_societe_a_verifier', 'segment_a_verifier'):
        continue
    clean_df[c] = try_numeric(clean_df[c])

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
clean_df.to_pickle(os.path.join(OUT_DIR, 'clean_df.pkl'))
json.dump(mapping, open(os.path.join(OUT_DIR, 'mapping.json'), 'w'), ensure_ascii=False, indent=1)

print("Colonnes brutes :", len(raw_cols))
print("Variables canoniques :", clean_df.shape[1] - 3)
print("Dossiers :", clean_df.shape[0])
cov = clean_df.drop(columns=['dossier_id', 'nom_societe_a_verifier', 'segment_a_verifier']).notna().sum()
print("Variables presentes dans >=6/9 dossiers :", (cov >= 6).sum())
print(clean_df[['dossier_id', 'nom_societe_a_verifier', 'segment_a_verifier']])
