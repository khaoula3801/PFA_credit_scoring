"""
Reconstruit tout le pipeline en une commande :
  1) nettoyage des donnees (data/clean_data.py)
  2) construction de la base SQLite (db/build_database.py)
  3) calcul des scores hybrides (scoring/hybrid_score.py)
Puis affiche comment lancer le dashboard.

Usage : python run_all.py
"""
import subprocess, sys, os

BASE = os.path.dirname(os.path.abspath(__file__))

steps = [
    ("Nettoyage des données", [sys.executable, os.path.join(BASE, "data", "clean_data.py")]),
    ("Construction de la base SQLite", [sys.executable, os.path.join(BASE, "db", "build_database.py")]),
    ("Calcul des scores hybrides (notation A-E)", [sys.executable, os.path.join(BASE, "scoring", "hybrid_score.py")]),
    ("Système de décision (défaut/non-défaut + feature importance)", [sys.executable, os.path.join(BASE, "scoring", "decision_engine.py")]),
]

for label, cmd in steps:
    print(f"\n=== {label} ===")
    result = subprocess.run(cmd, cwd=os.path.dirname(cmd[1]))
    if result.returncode != 0:
        print(f"ÉCHEC à l'étape : {label}")
        sys.exit(1)

print("\nPipeline reconstruit avec succès.")
print("Pour lancer le dashboard :")
print(f"  cd {os.path.join(BASE, 'dashboard')}")
print("  streamlit run app.py")
