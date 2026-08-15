"""
Systeme de Decision Credit — Banque Populaire, Centre d'Affaires Al Istiqlal
PFA GMSI - EMI

Workflow :
  - CONNEXION : chaque utilisateur (Analyste ou Direction) se connecte avec un compte
    local (mot de passe hashe + sele, jamais stocke en clair) ; deconnexion explicite.
  - ANALYSTE  : cree un nouveau dossier (scanner via camera, upload PDF, ou saisie
    manuelle), obtient un identifiant unique genere de facon cryptographiquement sure,
    puis soumet un commentaire pour defendre son client.
  - SYSTEME   : calcule automatiquement une decision (Accepte / Refuse / A etudier) avec
    justification et facteurs determinants — combinaison d'une logique experte (regles
    explicites) et d'une logique statistique (apprentissage automatique).
  - DIRECTION : consulte la decision + l'argumentaire, valide ou tranche, genere le PV.

Securite/anonymat : le nom de la societe (table `identites`) n'est jamais utilise par le
moteur de decision ni affiche dans les vues Analyste/Explicabilite. Il n'apparait que dans
la vue Direction, seule habilitee a lever l'anonymat pour la decision finale.

100% offline (SQLite local, aucune donnee ne sort du poste).
"""
import os
import sys
import base64
import sqlite3
import subprocess
from datetime import datetime

import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, 'db', 'credit_scoring.db')
UPLOADS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(UPLOADS_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(BASE_DIR, 'scoring'))
from auth import authenticate, generate_dossier_id  # noqa: E402
try:
    from default_detection import FEATURES_CANDIDATES  # noqa: E402
except Exception:
    FEATURES_CANDIDATES = []

st.set_page_config(page_title="Système de Décision Crédit - PFA", layout="wide", page_icon="🐎")
LOGO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "logo_bp.png")


def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def q(sql, params=None):
    return pd.read_sql(sql, get_conn(), params=params)


def execute(sql, params=None):
    conn = get_conn()
    conn.execute(sql, params or ())
    conn.commit()
    conn.close()


def get_performance_df():
    """Merge scores_calcules (grille experte / RF / LogReg / hybride) avec la notation
    MNS2 reelle (notations.score_final, notations.notation_lettre) pour la page de
    validation. Uniquement disponible pour les dossiers historiques deja notes par la
    banque -- un nouveau dossier cree via le workflow n'a pas de notation_lettre tant
    que la Direction ne l'a pas tranche, donc mns2_reel/score_final valent NaN pour lui."""
    df = q("""
        SELECT s.dossier_id, s.score_grille_expert,
               s.proba_rf_defaut AS score_rf, s.proba_logreg_defaut AS score_logreg,
               s.score_hybride, s.notation_predite, s.divergence_avec_mns2,
               n.notation_lettre AS mns2_lettre, n.score_final AS mns2_score
        FROM scores_calcules s
        LEFT JOIN notations n ON n.dossier_id = s.dossier_id
        ORDER BY s.dossier_id
    """)
    df['score_ml_moyen'] = df[['score_rf', 'score_logreg']].mean(axis=1)
    return df


# ============================================================================
# THEME — pastel chaleureux derive de l'orange Banque Populaire (#EE7203),
# fond creme doux, cartes blanches arrondies, badges de decision colores,
# boutons avec micro-interaction. 100% CSS local -> fonctionne hors ligne.
# ============================================================================
st.markdown("""
<style>
    :root {
        --bp-orange: #EE7203;
        --bp-orange-soft: #FFE3C7;
        --bp-navy: #1B2B45;
        --bp-cream: #FFF8F1;
        --pastel-green: #DCF2E3; --pastel-green-text: #1F7A4D;
        --pastel-red: #FBE1DF; --pastel-red-text: #B23B32;
        --pastel-amber: #FDECC3; --pastel-amber-text: #92650F;
    }
    .stApp { background: linear-gradient(180deg, var(--bp-cream) 0%, #FFFFFF 55%); }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #FFF1E2 0%, #FFF8F1 100%);
        border-right: 1px solid var(--bp-orange-soft);
    }
    section[data-testid="stSidebar"] label { font-weight: 600; color: var(--bp-navy); }
    .bp-header {
        display: flex; align-items: center; gap: 18px; padding: 14px 22px;
        background: #FFFFFF; border-radius: 18px; border: 1px solid var(--bp-orange-soft);
        box-shadow: 0 4px 18px rgba(238,114,3,0.08); margin-bottom: 22px;
    }
    .bp-header img { height: 52px; }
    .bp-header .bp-title { font-size: 1.5rem; font-weight: 800; color: var(--bp-navy); margin: 0; }
    .bp-header .bp-subtitle { font-size: 0.92rem; color: #7a7f8a; margin: 0; }
    .bp-badge {
        margin-left: auto; background: var(--bp-orange-soft); color: #8a4700; font-weight: 700;
        font-size: 0.78rem; padding: 6px 14px; border-radius: 999px; letter-spacing: .03em;
    }
    h1, h2, h3 { color: var(--bp-navy) !important; }
    .stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {
        background: var(--bp-orange) !important; color: white !important; border: none !important;
        border-radius: 12px !important; padding: 0.55em 1.4em !important; font-weight: 700 !important;
        box-shadow: 0 3px 10px rgba(238,114,3,0.28);
        transition: transform 0.15s ease, box-shadow 0.15s ease, background 0.15s ease;
    }
    .stButton > button:hover, .stFormSubmitButton > button:hover, .stDownloadButton > button:hover {
        transform: translateY(-3px) scale(1.02); box-shadow: 0 8px 20px rgba(238,114,3,0.38);
        background: #FF8A1F !important;
    }
    .stButton > button:active, .stFormSubmitButton > button:active { transform: translateY(0px) scale(0.99); }
    div[role="radiogroup"] label {
        border-radius: 999px !important; padding: 6px 14px !important;
        transition: background 0.15s ease, transform 0.15s ease;
    }
    div[role="radiogroup"] label:hover { background: var(--bp-orange-soft) !important; transform: translateX(2px); }
    div[data-testid="stMetric"] {
        background: #FFFFFF; border: 1px solid #F1E4D6; border-radius: 16px; padding: 14px 16px;
        box-shadow: 0 2px 10px rgba(27,43,69,0.05); transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    div[data-testid="stMetric"]:hover { transform: translateY(-2px); box-shadow: 0 6px 16px rgba(238,114,3,0.14); }
    div[data-testid="stExpander"] { border-radius: 14px !important; border: 1px solid #F1E4D6 !important; overflow: hidden; }
    .stTabs [data-baseweb="tab"] { border-radius: 10px 10px 0 0; padding: 8px 18px; font-weight: 600; }
    .stTabs [aria-selected="true"] { background: var(--bp-orange-soft) !important; color: var(--bp-navy) !important; }
    .badge-accepte { background: var(--pastel-green); color: var(--pastel-green-text);
        padding: 4px 14px; border-radius: 999px; font-weight: 700; display:inline-block; }
    .badge-refuse { background: var(--pastel-red); color: var(--pastel-red-text);
        padding: 4px 14px; border-radius: 999px; font-weight: 700; display:inline-block; }
    .badge-etudier { background: var(--pastel-amber); color: var(--pastel-amber-text);
        padding: 4px 14px; border-radius: 999px; font-weight: 700; display:inline-block; }
    div[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; border: 1px solid #F1E4D6; }
    .dossier-id-box {
        background: var(--bp-orange-soft); color: #8a4700; font-weight: 800; font-size: 1.1rem;
        padding: 10px 18px; border-radius: 12px; display: inline-block; letter-spacing: .04em;
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)

logo_html = ""
if os.path.exists(LOGO_PATH):
    with open(LOGO_PATH, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    logo_html = f'<img src="data:image/png;base64,{b64}" />'
else:
    logo_html = ('<div style="height:52px;width:52px;border-radius:12px;background:var(--bp-orange-soft);'
                 'display:flex;align-items:center;justify-content:center;color:#8a4700;font-weight:800;">BP</div>')

st.markdown(f"""
<div class="bp-header">
    {logo_html}
    <div>
        <p class="bp-title">Système de Décision Crédit</p>
        <p class="bp-subtitle">Centre d'Affaires Al Istiqlal · PFA GMSI-EMI</p>
    </div>
    <span class="bp-badge">100% OFFLINE</span>
</div>
""", unsafe_allow_html=True)

DECISION_BADGE_CLASS = {"ACCEPTE": "badge-accepte", "REFUSE": "badge-refuse", "A ETUDIER": "badge-etudier"}


def decision_badge(decision):
    cls = DECISION_BADGE_CLASS.get(decision, "badge-etudier")
    return f'<span class="{cls}">{decision}</span>'


# ============================================================================
# CONNEXION — obligatoire avant tout accès au reste de l'application.
# Comptes de démonstration créés par db/build_database.py :
#   analyste1 / changer123   (rôle Analyste)
#   direction1 / changer123  (rôle Direction)
# À changer avant tout usage réel (cf. README).
# ============================================================================
if 'user' not in st.session_state:
    st.session_state.user = None

if st.session_state.user is None:
    st.markdown("## 🔐 Connexion")
    st.caption("Accès réservé au personnel du centre d'affaires. Comptes locaux, aucune donnée transmise en ligne.")
    with st.form("login_form"):
        identifiant = st.text_input("Identifiant")
        mdp = st.text_input("Mot de passe", type="password")
        submit = st.form_submit_button("Se connecter")
        if submit:
            conn = get_conn()
            result = authenticate(conn, identifiant.strip(), mdp)
            conn.close()
            if result is None:
                st.error("Identifiant ou mot de passe incorrect.")
            else:
                nom_affiche, role = result
                st.session_state.user = {"identifiant": identifiant.strip(), "nom": nom_affiche, "role": role}
                st.rerun()
    with st.expander("ℹ️ Comptes de démonstration (à changer avant usage réel)"):
        st.code("analyste1 / changer123   → rôle Analyste\ndirection1 / changer123  → rôle Direction")
    st.stop()

# ---- Barre utilisateur + déconnexion ----
u = st.session_state.user
c1, c2 = st.columns([5, 1])
c1.markdown(f"👋 Connecté en tant que **{u['nom']}** — rôle *{u['role']}*")
if c2.button("🚪 Déconnexion"):
    st.session_state.user = None
    st.rerun()
st.markdown("---")

# ============================================================================
# NAVIGATION — les vues proposées dépendent du rôle du compte connecté.
# ============================================================================
views_by_role = {
    "Analyste": ["Accueil", "Analyste", "Explicabilité (facteurs)"],
    "Direction": ["Accueil", "Direction", "Explicabilité (facteurs)"],
}
available_views = views_by_role.get(u["role"], ["Accueil"])
st.sidebar.markdown("### 🧭 Navigation")
role_view = st.sidebar.radio("Navigation", available_views, label_visibility="collapsed")

# ================= ACCUEIL =================
if role_view == "Accueil":
    st.header(f"🏠 Bienvenue, {u['nom']}")
    n_dossiers = q("SELECT COUNT(*) n FROM dossiers")['n'][0]
    n_attente = q("SELECT COUNT(*) n FROM soumissions WHERE statut='en_attente'")['n'][0]
    n_traites = q("SELECT COUNT(*) n FROM decisions_direction")['n'][0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Dossiers dans la base", n_dossiers)
    c2.metric("En attente de décision", n_attente)
    c3.metric("Décisions rendues", n_traites)
    st.info(
        "Le caractère hybride de ce système ne réside pas dans l'empilement de plusieurs "
        "techniques statistiques, mais dans la combinaison de deux logiques indépendantes : "
        "une **logique experte** (règles explicites et transparentes) et une **logique "
        "statistique** (apprentissage automatique de régularités dans les données). Les deux "
        "sont construites séparément, puis confrontées entre elles et à la notation bancaire "
        "réelle — permettant d'analyser les zones de convergence et de divergence entre "
        "expertise humaine et apprentissage statistique."
    )

    dec_all = q("SELECT decision_systeme FROM decision_ml")
    perf_df = get_performance_df()

    g1, g2 = st.columns(2)
    with g1:
        st.markdown("##### 📊 Répartition des décisions du système")
        if dec_all.empty:
            st.caption("Aucune décision calculée pour le moment.")
        else:
            repartition = dec_all['decision_systeme'].value_counts()
            st.bar_chart(repartition)
    with g2:
        st.markdown("##### ⚖️ Score Expert vs ML vs Banque (dossiers notés)")
        comp = perf_df.dropna(subset=['mns2_score']).set_index('dossier_id')[
            ['score_grille_expert', 'score_ml_moyen', 'mns2_score']
        ]
        comp.columns = ['Expert', 'ML', 'Banque (MNS2)']
        if comp.empty:
            st.caption("Aucun dossier avec notation MNS2 de référence pour l'instant.")
        else:
            st.bar_chart(comp)

    fi_home = q("SELECT * FROM feature_importance ORDER BY rang LIMIT 8")
    if not fi_home.empty:
        st.markdown("##### 📈 Variables les plus déterminantes (Random Forest)")
        st.bar_chart(fi_home.set_index('variable')['importance_rf'])

# ================= ANALYSTE =================
elif role_view == "Analyste":
    st.header("👤 Espace Analyste")
    st.caption("Vous voyez uniquement les identifiants de dossier (anonymisé) — jamais le nom de la société.")

    tab_new, tab_comment, tab_mine = st.tabs(
        ["🆕 Créer un nouveau dossier", "💬 Commentaire sur un dossier", "📋 Mes soumissions"]
    )

    # ---- Création de dossier : scanner / upload PDF / saisie manuelle ----
    with tab_new:
        st.markdown("Choisissez comment déposer les pièces du dossier :")
        mode = st.radio(
            "Mode de dépôt", ["📷 Scanner (caméra)", "📎 Uploader un PDF", "⌨️ Saisie manuelle uniquement"],
            horizontal=True
        )

        scanned_image = None
        uploaded_pdf = None
        pdf_text_preview = ""

        if mode == "📷 Scanner (caméra)":
            scanned_image = st.camera_input("Scanner le document (liasse fiscale, demande de crédit...)")
            if scanned_image is not None:
                st.success("Document scanné. Reportez les valeurs visibles dans le formulaire ci-dessous.")

        elif mode == "📎 Uploader un PDF":
            uploaded_pdf = st.file_uploader("Uploader le document (PDF)", type=["pdf"])
            if uploaded_pdf is not None:
                try:
                    import pdfplumber
                    with pdfplumber.open(uploaded_pdf) as pdf:
                        pdf_text_preview = "\n".join((p.extract_text() or "") for p in pdf.pages[:5])
                    st.success("PDF chargé. Texte extrait ci-dessous à titre d'aide-mémoire (pas de saisie automatique).")
                    with st.expander("📄 Texte extrait du PDF (aperçu)"):
                        st.text_area("Contenu détecté", pdf_text_preview[:4000], height=200, disabled=True)
                except Exception:
                    st.warning("PDF chargé, mais l'extraction de texte a échoué (document scanné/image ?). "
                               "Saisissez les valeurs manuellement ci-dessous.")
        else:
            st.caption("Aucune pièce jointe — remplissez directement le formulaire ci-dessous.")

        st.markdown("---")
        st.subheader("Informations du dossier")

        with st.form("nouveau_dossier_form"):
            nom_societe = st.text_input("Raison sociale du client *")
            segment = st.selectbox("Segment", ["Crédit de Fonctionnement / CT", "Promotion Immobilière",
                                                "Moyen/Long Terme", "À vérifier"])

            groupes = {
                "Bilan — Actif": [f for f in FEATURES_CANDIDATES if 'actif' in f or 'immob' in f or 'clients' in f],
                "Bilan — Passif / Capitaux": [f for f in FEATURES_CANDIDATES if any(
                    k in f for k in ['passif', 'capital', 'dette', 'reserve', 'fonds', 'propres'])],
                "Résultat & charges": [f for f in FEATURES_CANDIDATES if any(
                    k in f for k in ['resultat', 'resutats', 'charge', 'consomm', 'valeur_ajoutee', 'dotaion', 'rentabilite'])],
                "Trésorerie & ratios": [f for f in FEATURES_CANDIDATES if any(
                    k in f for k in ['tresorerie', 'liquidite', 'endettement', 'capacite', 'autonom'])],
            }
            deja_places = {v for lst in groupes.values() for v in lst}
            groupes["Autres variables"] = [f for f in FEATURES_CANDIDATES if f not in deja_places]

            saisies = {}
            for titre, variables in groupes.items():
                if not variables:
                    continue
                with st.expander(titre):
                    cols = st.columns(2)
                    for i, var in enumerate(variables):
                        label = var.replace('_', ' ').capitalize()
                        val = cols[i % 2].text_input(label, key=f"new_{var}", placeholder="laisser vide si inconnu")
                        saisies[var] = val

            commentaire_creation = st.text_area(
                "Commentaire initial (optionnel)",
                placeholder="Contexte du dossier, éléments qualitatifs non capturés par les chiffres..."
            )

            creer = st.form_submit_button("✅ Créer le dossier")

            if creer:
                if not nom_societe.strip():
                    st.error("La raison sociale est obligatoire.")
                else:
                    conn = get_conn()
                    new_id = generate_dossier_id(conn)
                    conn.execute("INSERT INTO dossiers (dossier_id, segment_brut, segment_valide, statut_validation) "
                                 "VALUES (?,?,?, 'nouveau')", (new_id, segment, segment))
                    conn.execute("INSERT INTO identites (dossier_id, nom_societe) VALUES (?,?)",
                                 (new_id, nom_societe.strip()))

                    for var, val in saisies.items():
                        val = (val or "").strip().replace(',', '.')
                        if val == "":
                            continue
                        try:
                            fval = float(val)
                            conn.execute(
                                "INSERT OR REPLACE INTO variables_valeurs (dossier_id, variable, valeur_num, valeur_texte) "
                                "VALUES (?,?,?,NULL)", (new_id, var, fval))
                        except ValueError:
                            conn.execute(
                                "INSERT OR REPLACE INTO variables_valeurs (dossier_id, variable, valeur_num, valeur_texte) "
                                "VALUES (?,?,NULL,?)", (new_id, var, val))

                    # sauvegarde de la piece jointe eventuelle
                    if scanned_image is not None:
                        path = os.path.join(UPLOADS_DIR, f"{new_id}_scan.jpg")
                        with open(path, "wb") as f:
                            f.write(scanned_image.getbuffer())
                        conn.execute("INSERT INTO documents_dossier (dossier_id, type_source, nom_fichier, chemin_fichier) "
                                     "VALUES (?, 'scan_camera', ?, ?)", (new_id, os.path.basename(path), path))
                    elif uploaded_pdf is not None:
                        path = os.path.join(UPLOADS_DIR, f"{new_id}_{uploaded_pdf.name}")
                        with open(path, "wb") as f:
                            f.write(uploaded_pdf.getbuffer())
                        conn.execute("INSERT INTO documents_dossier (dossier_id, type_source, nom_fichier, chemin_fichier) "
                                     "VALUES (?, 'pdf_upload', ?, ?)", (new_id, uploaded_pdf.name, path))
                    else:
                        conn.execute("INSERT INTO documents_dossier (dossier_id, type_source, nom_fichier, chemin_fichier) "
                                     "VALUES (?, 'saisie_manuelle', NULL, NULL)", (new_id,))

                    if commentaire_creation.strip():
                        conn.execute(
                            "INSERT INTO soumissions (dossier_id, analyste, commentaire_analyste, date_soumission, statut) "
                            "VALUES (?,?,?,?, 'en_attente')",
                            (new_id, u['nom'], commentaire_creation.strip(), datetime.now().isoformat())
                        )
                    conn.commit()
                    conn.close()

                    st.success("Dossier créé avec succès.")
                    st.markdown("**Identifiant unique généré :**")
                    st.markdown(f'<span class="dossier-id-box">{new_id}</span>', unsafe_allow_html=True)
                    st.caption(
                        "Cet identifiant est généré aléatoirement (aucun lien avec un compteur "
                        "prévisible) et garanti unique dans la base. Notez-le pour retrouver ce "
                        "dossier plus tard."
                    )

    # ---- Commentaire sur un dossier existant (recherche par ID) ----
    with tab_comment:
        st.markdown("Recherchez un dossier par son identifiant pour y ajouter un commentaire de défense.")
        recherche = st.text_input("🔎 Identifiant du dossier (ex: DOS-A1B2C3D4E5)")
        dossiers_disponibles = q("SELECT dossier_id FROM dossiers ORDER BY dossier_id")['dossier_id'].tolist()

        if recherche:
            correspondances = [d for d in dossiers_disponibles if recherche.strip().upper() in d.upper()]
        else:
            correspondances = dossiers_disponibles

        if not correspondances:
            st.warning("Aucun dossier ne correspond à cette recherche.")
        else:
            with st.form("soumission_form"):
                dossier_id = st.selectbox("Dossier concerné", correspondances)
                commentaire = st.text_area(
                    "Commentaire — argumentaire pour défendre ce client",
                    placeholder="Ex: retard de paiement ponctuel lié à un différend client, "
                                "trésorerie structurellement saine, garanties complémentaires proposées..."
                )
                submitted = st.form_submit_button("Soumettre à la Direction")
                if submitted:
                    if not commentaire.strip():
                        st.error("Merci de renseigner un commentaire.")
                    else:
                        execute(
                            "INSERT INTO soumissions (dossier_id, analyste, commentaire_analyste, date_soumission, statut) "
                            "VALUES (?,?,?,?, 'en_attente')",
                            (dossier_id, u['nom'], commentaire.strip(), datetime.now().isoformat())
                        )
                        st.success(f"Dossier {dossier_id} soumis à la Direction.")

    with tab_mine:
        subs = q("SELECT dossier_id, analyste, commentaire_analyste, date_soumission, statut FROM soumissions "
                 "WHERE analyste = ? ORDER BY date_soumission DESC", params=(u['nom'],))
        if subs.empty:
            st.info("Aucune soumission pour l'instant.")
        else:
            st.dataframe(subs, use_container_width=True)

# ================= DIRECTION =================
elif role_view == "Direction":
    st.header("🧑‍💼 Espace Direction")
    st.warning("Vue habilitée à lever l'anonymat (nom de société visible) pour la décision finale.")

    with st.expander("⚙️ Administration du pipeline"):
        st.caption(
            "Après ajout de nouveaux dossiers, relancez le calcul pour mettre à jour les scores, "
            "le seuil de défaut (recalculé dynamiquement) et les décisions."
        )
        if st.button("🔄 Recalculer scores & décisions"):
            scoring_dir = os.path.join(BASE_DIR, 'scoring')
            with st.spinner("Recalcul en cours (nettoyage, grille experte, RF/LogReg, décision)..."):
                r1 = subprocess.run([sys.executable, 'hybrid_score.py'], cwd=scoring_dir, capture_output=True, text=True)
                r2 = subprocess.run([sys.executable, 'decision_engine.py'], cwd=scoring_dir, capture_output=True, text=True)
            if r1.returncode == 0 and r2.returncode == 0:
                st.success("Recalcul terminé.")
            else:
                st.error("Erreur pendant le recalcul — voir détails ci-dessous.")
                st.code((r1.stderr or "") + "\n" + (r2.stderr or ""))

    en_attente = q("""
        SELECT s.id, s.dossier_id, i.nom_societe, s.analyste, s.commentaire_analyste, s.date_soumission
        FROM soumissions s
        LEFT JOIN identites i ON i.dossier_id = s.dossier_id
        WHERE s.statut = 'en_attente'
        ORDER BY s.date_soumission
    """)

    if en_attente.empty:
        st.info("Aucun dossier en attente de décision.")
    else:
        for _, sub in en_attente.iterrows():
            with st.expander(f"📁 {sub['dossier_id']} — {sub['nom_societe']} (soumis par {sub['analyste']})"):
                st.write("**Commentaire de l'analyste :**")
                st.info(sub['commentaire_analyste'])

                perf = q("""
                    SELECT s.score_grille_expert, s.proba_rf_defaut AS score_rf,
                           s.proba_logreg_defaut AS score_logreg, s.score_hybride,
                           n.notation_lettre AS mns2_lettre, n.score_final AS mns2_score
                    FROM scores_calcules s LEFT JOIN notations n ON n.dossier_id = s.dossier_id
                    WHERE s.dossier_id = ?
                """, params=(sub['dossier_id'],))
                if not perf.empty:
                    p = perf.iloc[0]
                    score_ml = pd.Series([p['score_rf'], p['score_logreg']]).mean()
                    sc1, sc2, sc3 = st.columns(3)
                    sc1.metric("🧭 Score Expert", f"{p['score_grille_expert']:.0f}/100" if pd.notna(p['score_grille_expert']) else "N/A")
                    sc2.metric("🤖 Score ML", f"{score_ml:.0f}/100" if pd.notna(score_ml) else "N/A")
                    sc3.metric("🏦 Score Banque (MNS2)",
                               f"{p['mns2_score']:.0f}/100 ({p['mns2_lettre']})" if pd.notna(p['mns2_score']) else "Non noté")

                dec = q("SELECT * FROM decision_ml WHERE dossier_id = ?", params=(sub['dossier_id'],))
                if not dec.empty:
                    d = dec.iloc[0]
                    st.markdown("**Décision automatique du système (risque de défaut)**")
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown(decision_badge(d['decision_systeme']), unsafe_allow_html=True)
                    c2.metric("Proba défaut (RF)", f"{d['proba_defaut_rf']:.0%}" if pd.notna(d['proba_defaut_rf']) else "N/A")
                    c3.metric("Proba défaut (LogReg)", f"{d['proba_defaut_logreg']:.0%}" if pd.notna(d['proba_defaut_logreg']) else "N/A")
                    st.write("**Justification du système :**", d['justification'])
                else:
                    st.warning("Décision système non disponible — relancez le recalcul ci-dessus si ce dossier est nouveau.")

                docs = q("SELECT type_source, nom_fichier, chemin_fichier FROM documents_dossier WHERE dossier_id = ?",
                         params=(sub['dossier_id'],))
                if not docs.empty:
                    for _, doc in docs.iterrows():
                        if doc['type_source'] == 'scan_camera' and doc['chemin_fichier'] and os.path.exists(doc['chemin_fichier']):
                            st.image(doc['chemin_fichier'], caption="Document scanné", width=250)
                        elif doc['type_source'] == 'pdf_upload' and doc['nom_fichier']:
                            st.caption(f"📎 Pièce jointe : {doc['nom_fichier']}")

                with st.form(f"decision_form_{sub['id']}"):
                    decision_finale = st.radio(
                        "Décision finale", ["Accepté", "Refusé", "À compléter"], horizontal=True, key=f"dec_radio_{sub['id']}"
                    )
                    commentaire_direction = st.text_area("Commentaire / motif de la décision", key=f"dec_comment_{sub['id']}")
                    valider = st.form_submit_button("Valider et générer le PV")
                    if valider:
                        execute(
                            "INSERT INTO decisions_direction (dossier_id, decideur, decision_finale, "
                            "commentaire_direction, date_decision) VALUES (?,?,?,?,?)",
                            (sub['dossier_id'], u['nom'], decision_finale, commentaire_direction, datetime.now().isoformat())
                        )
                        execute("UPDATE soumissions SET statut = 'traite' WHERE id = ?", (int(sub['id']),))
                        st.success("Décision enregistrée. PV disponible dans l'onglet 'Historique / PV'.")
                        st.rerun()

    st.markdown("---")
    st.subheader("📜 Historique / PV des décisions")
    hist = q("""
        SELECT dd.dossier_id, i.nom_societe, dd.decideur, dd.decision_finale,
               dd.commentaire_direction, dd.date_decision
        FROM decisions_direction dd
        LEFT JOIN identites i ON i.dossier_id = dd.dossier_id
        ORDER BY dd.date_decision DESC
    """)
    st.dataframe(hist, use_container_width=True)

    if not hist.empty:
        libelles = (hist['dossier_id'] + " — " + hist['nom_societe'].fillna("")).tolist()
        choix = st.selectbox("Générer le PV texte pour :", libelles)
        idx = libelles.index(choix)
        r = hist.iloc[idx]
        pv_text = f"""PROCÈS-VERBAL DE DÉCISION DE CRÉDIT
=====================================
Dossier          : {r['dossier_id']}
Société          : {r['nom_societe']}
Décision finale  : {r['decision_finale']}
Décideur         : {r['decideur']}
Date             : {r['date_decision']}

Commentaire / motif :
{r['commentaire_direction']}
"""
        st.text_area("PV généré", pv_text, height=220)
        st.download_button("Télécharger le PV (.txt)", pv_text, file_name=f"PV_{r['dossier_id']}.txt")

# ================= EXPLICABILITE =================
else:
    st.header("📊 Explicabilité du système de décision")
    st.caption("Vue anonymisée — aucune identité de société n'apparaît ici.")

    param = q("SELECT valeur FROM parametres_systeme WHERE cle = 'seuil_defaut_score_final'")
    seuil = param['valeur'][0] if not param.empty else "N/A"
    st.info(
        f"**Seuil de défaut calculé automatiquement (KMeans sur score_final) : {seuil}.** "
        "Ce seuil n'est pas figé — il est recalculé à chaque exécution du pipeline à mesure "
        "que de nouveaux dossiers sont ajoutés via le workflow analyste → direction."
    )

    st.markdown(
        "Le caractère hybride ne réside pas dans l'empilement de techniques statistiques sur "
        "une même donnée, mais dans la combinaison de **deux logiques de décision indépendantes**, "
        "construites séparément puis confrontées entre elles et à la notation bancaire réelle (MNS2)."
    )

    tab_expert, tab_ml, tab_perf = st.tabs(
        ["🧭 Système Expert", "🤖 Modèle ML", "⚖️ Performance vs Banque (MNS2)"]
    )

    # ---- Onglet Système Expert ----
    with tab_expert:
        st.caption("Grille pondérée, règles explicites et transparentes, fondées sur des ratios financiers.")
        grille = pd.DataFrame([
            {"Ratio": "Rentabilité financière", "Poids": "20%", "Sens": "plus est mieux"},
            {"Ratio": "Autonomie financière", "Poids": "20%", "Sens": "plus est mieux"},
            {"Ratio": "Endettement à court terme", "Poids": "15%", "Sens": "moins est mieux"},
            {"Ratio": "Liquidité réduite", "Poids": "15%", "Sens": "plus est mieux"},
            {"Ratio": "Capacité de remboursement", "Poids": "15%", "Sens": "plus est mieux"},
            {"Ratio": "Dettes nettes / fonds propres", "Poids": "15%", "Sens": "moins est mieux"},
        ])
        st.dataframe(grille, use_container_width=True, hide_index=True)
        st.caption(
            "Ces pondérations sont une hypothèse de travail à valider avec le maître de stage — "
            "elles ne reproduisent pas la grille interne exacte de la Banque Populaire."
        )
        st.markdown("**Score expert par dossier (anonymisé)**")
        perf_df = get_performance_df()
        st.bar_chart(perf_df.set_index('dossier_id')['score_grille_expert'])

    # ---- Onglet Modèle ML ----
    with tab_ml:
        st.caption("Random Forest + régression logistique, évalués en LOOCV, apprentissage de régularités dans les données réelles.")
        st.subheader("📋 Décisions du système, par dossier (anonymisé)")
        dec_all = q("SELECT * FROM decision_ml ORDER BY proba_defaut_hybride DESC")
        for _, d in dec_all.iterrows():
            c1, c2, c3, c4 = st.columns([1, 1, 1, 3])
            c1.markdown(f"**{d['dossier_id']}**")
            c2.markdown(decision_badge(d['decision_systeme']), unsafe_allow_html=True)
            c3.markdown(f"{d['proba_defaut_hybride']:.0%}" if pd.notna(d['proba_defaut_hybride']) else "N/A")
            c4.caption(d['justification'])
        st.markdown("---")

        st.subheader("🎯 Facteurs les plus déterminants dans la décision (Random Forest)")
        fi = q("SELECT * FROM feature_importance ORDER BY rang LIMIT 15")
        if fi.empty:
            st.warning("Importance des variables non calculée — lancez `python scoring/decision_engine.py`.")
        else:
            st.bar_chart(fi.set_index('variable')['importance_rf'])
            st.dataframe(fi, use_container_width=True)

    # ---- Onglet Performance vs Banque ----
    with tab_perf:
        st.caption(
            "Comparaison des scores reconstruits (grille experte, ML, hybride) à la notation "
            "MNS2 réelle — uniquement disponible pour les dossiers historiques déjà notés par la banque."
        )
        perf_df = get_performance_df()
        notes = perf_df.dropna(subset=['mns2_score'])

        if notes.empty:
            st.info("Aucun dossier avec notation MNS2 de référence pour l'instant.")
        else:
            corr_expert = notes['score_grille_expert'].corr(notes['mns2_score'])
            corr_ml = notes['score_ml_moyen'].corr(notes['mns2_score'])
            corr_hybride = notes['score_hybride'].corr(notes['mns2_score'])
            mae_expert = (notes['score_grille_expert'] - notes['mns2_score']).abs().mean()
            mae_ml = (notes['score_ml_moyen'] - notes['mns2_score']).abs().mean()
            mae_hybride = (notes['score_hybride'] - notes['mns2_score']).abs().mean()

            st.markdown("##### Corrélation avec la notation bancaire")
            cc1, cc2, cc3 = st.columns(3)
            cc1.metric("Score Expert", f"{corr_expert:.2f}" if pd.notna(corr_expert) else "N/A")
            cc2.metric("Score ML", f"{corr_ml:.2f}" if pd.notna(corr_ml) else "N/A")
            cc3.metric("Score Hybride", f"{corr_hybride:.2f}" if pd.notna(corr_hybride) else "N/A")

            st.markdown("##### Erreur moyenne absolue (points, échelle 0-100)")
            ce1, ce2, ce3 = st.columns(3)
            ce1.metric("Score Expert", f"{mae_expert:.1f} pts" if pd.notna(mae_expert) else "N/A")
            ce2.metric("Score ML", f"{mae_ml:.1f} pts" if pd.notna(mae_ml) else "N/A")
            ce3.metric("Score Hybride", f"{mae_hybride:.1f} pts" if pd.notna(mae_hybride) else "N/A")

            meilleur = min(
                [("le score expert", mae_expert), ("le modèle ML", mae_ml), ("le score hybride", mae_hybride)],
                key=lambda x: x[1] if pd.notna(x[1]) else float('inf')
            )[0]
            st.success(
                f"Sur les {len(notes)} dossiers notés, **{meilleur}** présente l'écart moyen le plus "
                f"faible avec la notation MNS2 réelle. À interpréter avec prudence (échantillon très "
                f"restreint, cf. limites méthodologiques ci-dessous)."
            )

            table = notes[[
                'dossier_id', 'score_grille_expert', 'score_ml_moyen', 'score_hybride',
                'notation_predite', 'mns2_score', 'mns2_lettre', 'divergence_avec_mns2'
            ]].rename(columns={
                'score_grille_expert': 'Expert', 'score_ml_moyen': 'ML', 'score_hybride': 'Hybride',
                'notation_predite': 'Lettre prédite', 'mns2_score': 'Score MNS2 réel',
                'mns2_lettre': 'Lettre MNS2 réelle', 'divergence_avec_mns2': 'Divergence ≥2 crans'
            })
            st.dataframe(table, use_container_width=True, hide_index=True)

            divergents = notes[notes['divergence_avec_mns2'] == 1]
            if not divergents.empty:
                st.warning(
                    f"⚠️ {len(divergents)} dossier(s) présentent un écart de notation ≥ 2 crans avec "
                    f"la notation MNS2 réelle : {', '.join(divergents['dossier_id'].tolist())}. "
                    "À investiguer en priorité avec le maître de stage."
                )

    st.subheader("⚠️ Limites méthodologiques à rappeler dans le rapport")
    st.markdown("""
- **n très restreint** (9 dossiers, 7 labellisés) : le LOOCV donne une estimation *bruitée* de
  la performance réelle — à présenter comme indicatif, jamais comme une validation robuste.
- La cible "défaut" est un **proxy** (coupure automatique sur score_final), pas un historique
  réel d'incidents de paiement — hypothèse à valider avec le maître de stage.
- Le seuil de décision (60% / 25%) et les poids de la grille experte sont des **hypothèses de
  travail**, pas la politique de risque réelle de la banque.
- Les corrélations/erreurs moyennes ci-dessus portent sur un très petit nombre de dossiers notés
  — à présenter comme indicatif, jamais comme une validation statistique robuste.
    """)

st.sidebar.markdown("---")
st.sidebar.caption("Base : credit_scoring.db (SQLite locale) · 100% offline · Anonymisation par séparation des identités")
