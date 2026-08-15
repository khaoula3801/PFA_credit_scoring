"""
Construit credit_scoring.db (SQLite) a partir de data/clean_df.pkl + mapping.json
Schema normalise :
  dossiers            : 1 ligne par dossier client (id, nom, segment, notation cible)
  variables_valeurs   : format EAV (dossier_id, variable, valeur_num, valeur_texte) -> gere les 339
                         variables tres creuses sans avoir 339 colonnes vides dans une table plate
  variables_dictionnaire : catalogue des variables (couverture, type, exemple)
  mapping_nettoyage   : tracabilite variable_brute -> variable_canonique (audit)
  notations           : notation MNS2 cible + composantes (score_quanti/quali/final, lettre)
  scores_calcules     : sorties du pipeline hybride (grille experte, RF, LogReg, score hybride)
"""
import sqlite3, json, os
import pandas as pd
import numpy as np
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scoring'))
from auth import create_user

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'credit_scoring.db')

clean_df = pd.read_pickle(os.path.join(DATA_DIR, 'clean_df.pkl'))
mapping = json.load(open(os.path.join(DATA_DIR, 'mapping.json'), encoding='utf-8'))

if os.path.exists(DB_PATH):
    os.remove(DB_PATH)
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.executescript("""
CREATE TABLE dossiers (
    dossier_id TEXT PRIMARY KEY,
    segment_brut TEXT,
    segment_valide TEXT,          -- a completer/corriger avec le maitre de stage
    statut_validation TEXT DEFAULT 'a_verifier'
);

-- Table separee et isolee : seule table contenant une donnee identifiante (nom societe).
-- Le reste du systeme (ML, feature importance, decision) ne travaille QUE sur dossier_id.
-- Acces reserve a la vue "Direction" du dashboard (cf. README section securite).
CREATE TABLE identites (
    dossier_id TEXT PRIMARY KEY,
    nom_societe TEXT,
    FOREIGN KEY (dossier_id) REFERENCES dossiers(dossier_id)
);

CREATE TABLE variables_dictionnaire (
    variable TEXT PRIMARY KEY,
    nb_dossiers_renseignes INTEGER,
    type_detecte TEXT,
    exemple_valeur TEXT
);

CREATE TABLE variables_valeurs (
    dossier_id TEXT,
    variable TEXT,
    valeur_num REAL,
    valeur_texte TEXT,
    PRIMARY KEY (dossier_id, variable),
    FOREIGN KEY (dossier_id) REFERENCES dossiers(dossier_id),
    FOREIGN KEY (variable) REFERENCES variables_dictionnaire(variable)
);

CREATE TABLE mapping_nettoyage (
    variable_brute TEXT PRIMARY KEY,
    variable_canonique TEXT
);

CREATE TABLE notations (
    dossier_id TEXT PRIMARY KEY,
    score_qualitatif REAL,
    score_final REAL,
    notation_lettre TEXT,          -- notation MNS2 cible (A-E), source expert
    FOREIGN KEY (dossier_id) REFERENCES dossiers(dossier_id)
);

CREATE TABLE scores_calcules (
    dossier_id TEXT PRIMARY KEY,
    score_grille_expert REAL,
    proba_rf_defaut REAL,
    proba_logreg_defaut REAL,
    score_hybride REAL,
    notation_predite TEXT,
    divergence_avec_mns2 INTEGER,   -- 1 si ecart notable avec notation_lettre -> a investiguer
    FOREIGN KEY (dossier_id) REFERENCES dossiers(dossier_id)
);

CREATE TABLE dossiers_prioritaires (
    dossier_id TEXT PRIMARY KEY,
    motif TEXT
);

-- Workflow analyste -> direction
CREATE TABLE soumissions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dossier_id TEXT,
    analyste TEXT,
    commentaire_analyste TEXT,     -- argumentaire de defense du client
    date_soumission TEXT DEFAULT CURRENT_TIMESTAMP,
    statut TEXT DEFAULT 'en_attente',   -- en_attente / valide / refuse
    FOREIGN KEY (dossier_id) REFERENCES dossiers(dossier_id)
);

CREATE TABLE decisions_direction (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dossier_id TEXT,
    decideur TEXT,
    decision_finale TEXT,          -- Accepte / Refuse / A completer
    commentaire_direction TEXT,
    date_decision TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (dossier_id) REFERENCES dossiers(dossier_id)
);

CREATE TABLE decision_ml (
    dossier_id TEXT PRIMARY KEY,
    proba_defaut_rf REAL,
    proba_defaut_logreg REAL,
    proba_defaut_hybride REAL,
    label_reel_defaut INTEGER,     -- 1 = defaut (score_final<40), 0 = sain, NULL = inconnu
    decision_systeme TEXT,         -- Accepte / Refuse / A etudier (regle logique explicite)
    justification TEXT,
    FOREIGN KEY (dossier_id) REFERENCES dossiers(dossier_id)
);

CREATE TABLE feature_importance (
    variable TEXT,
    importance_rf REAL,
    importance_permutation REAL,
    rang INTEGER
);

CREATE TABLE parametres_systeme (
    cle TEXT PRIMARY KEY,
    valeur TEXT,
    date_calcul TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Comptes utilisateurs (authentification locale, hash+sel, aucune donnee ne sort du poste)
CREATE TABLE utilisateurs (
    identifiant TEXT PRIMARY KEY,
    nom_affiche TEXT,
    role TEXT CHECK(role IN ('Analyste','Direction')),
    mdp_hash TEXT,
    mdp_sel TEXT,
    actif INTEGER DEFAULT 1
);

-- Documents attaches a un dossier (PDF uploade ou photo scannee)
CREATE TABLE documents_dossier (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dossier_id TEXT,
    type_source TEXT,          -- 'pdf_upload' / 'scan_camera' / 'saisie_manuelle'
    nom_fichier TEXT,
    chemin_fichier TEXT,
    date_ajout TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (dossier_id) REFERENCES dossiers(dossier_id)
);
""")

# --- dossiers (anonyme) + identites (isole) ---
for _, r in clean_df.iterrows():
    cur.execute("INSERT INTO dossiers VALUES (?,?,?,?)",
                (r['dossier_id'], r['segment_a_verifier'], None, 'a_verifier'))
    cur.execute("INSERT INTO identites VALUES (?,?)",
                (r['dossier_id'], r['nom_societe_a_verifier']))

# --- variables_dictionnaire + variables_valeurs (EAV) ---
value_cols = [c for c in clean_df.columns if c not in ('dossier_id', 'nom_societe_a_verifier', 'segment_a_verifier')]
for var in value_cols:
    col = clean_df[var]
    nb = col.notna().sum()
    is_num = pd.api.types.is_numeric_dtype(col)
    example = col.dropna().iloc[0] if nb > 0 else None
    cur.execute("INSERT INTO variables_dictionnaire VALUES (?,?,?,?)",
                (var, int(nb), 'numerique' if is_num else 'texte', str(example) if example is not None else None))

for _, r in clean_df.iterrows():
    for var in value_cols:
        val = r[var]
        if pd.isna(val):
            continue
        if isinstance(val, (int, float, np.integer, np.floating)):
            cur.execute("INSERT INTO variables_valeurs VALUES (?,?,?,?)", (r['dossier_id'], var, float(val), None))
        else:
            cur.execute("INSERT INTO variables_valeurs VALUES (?,?,?,?)", (r['dossier_id'], var, None, str(val)))

# --- mapping_nettoyage ---
for raw, canon in mapping.items():
    cur.execute("INSERT OR REPLACE INTO mapping_nettoyage VALUES (?,?)", (raw, canon))

# --- notations (notation cible reconstruite par coalescence des colonnes de notation dispo) ---
notation_candidates = ['note_finale', 'note_calculee', 'notation', 'notation_2024_affaire', 'notation_projet']
for _, r in clean_df.iterrows():
    lettre = None
    for c in notation_candidates:
        if c in clean_df.columns and pd.notna(r[c]):
            lettre = str(r[c]).strip().upper()
            break
    sq = r['score_qualitatif'] if 'score_qualitatif' in clean_df.columns and pd.notna(r['score_qualitatif']) else None
    sf = r['score_final'] if 'score_final' in clean_df.columns and pd.notna(r['score_final']) else None
    cur.execute("INSERT INTO notations VALUES (?,?,?,?)", (r['dossier_id'], sq, sf, lettre))

# --- dossiers prioritaires (flagges precedemment comme divergents, a investiguer avec le maitre de stage) ---
for d, motif in [('Feuil1', 'Score/notation divergent releve lors de la premiere analyse - a investiguer'),
                  ('Feuil2', 'Score/notation divergent releve lors de la premiere analyse - a investiguer'),
                  ('Feuil8', 'Score/notation divergent releve lors de la premiere analyse - a investiguer'),
                  ('Feuil13', 'Score/notation divergent releve lors de la premiere analyse - a investiguer')]:
    cur.execute("INSERT INTO dossiers_prioritaires VALUES (?,?)", (d, motif))

# --- comptes de demonstration (A CHANGER avant tout usage reel) ---
create_user(conn, 'analyste1', 'Analyste Démo', 'Analyste', 'changer123')
create_user(conn, 'direction1', 'Direction Démo', 'Direction', 'changer123')

conn.commit()
conn.close()
print("Base construite :", DB_PATH)
print("Dossiers:", len(clean_df), "| Variables:", len(value_cols))
