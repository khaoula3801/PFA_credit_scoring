"""
Authentification locale (aucun serveur externe) + generation d'identifiants
uniques et securises pour les nouveaux dossiers.

Mots de passe : jamais stockes en clair. PBKDF2-HMAC-SHA256 (stdlib `hashlib`,
100 000 iterations) avec un sel aleatoire par utilisateur (stdlib `secrets`).
Identifiants de dossier : `secrets.token_hex` (aleatoire cryptographique,
pas un compteur predictible), verifie contre la base pour garantir l'unicite.
"""
import hashlib
import secrets
import sqlite3


def hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac('sha256', password.encode(), bytes.fromhex(salt), 100_000)
    return digest.hex(), salt


def verify_password(password: str, stored_hash: str, salt: str) -> bool:
    computed, _ = hash_password(password, salt)
    return secrets.compare_digest(computed, stored_hash)


def create_user(conn: sqlite3.Connection, identifiant: str, nom_affiche: str, role: str, password: str):
    h, salt = hash_password(password)
    conn.execute(
        "INSERT OR REPLACE INTO utilisateurs (identifiant, nom_affiche, role, mdp_hash, mdp_sel, actif) "
        "VALUES (?,?,?,?,?,1)",
        (identifiant, nom_affiche, role, h, salt)
    )


def authenticate(conn: sqlite3.Connection, identifiant: str, password: str):
    """Retourne (nom_affiche, role) si les identifiants sont valides, sinon None."""
    row = conn.execute(
        "SELECT nom_affiche, role, mdp_hash, mdp_sel FROM utilisateurs WHERE identifiant = ? AND actif = 1",
        (identifiant,)
    ).fetchone()
    if row is None:
        return None
    nom_affiche, role, mdp_hash, mdp_sel = row
    if verify_password(password, mdp_hash, mdp_sel):
        return nom_affiche, role
    return None


def generate_dossier_id(conn: sqlite3.Connection, prefix: str = "DOS") -> str:
    """Genere un identifiant de dossier aleatoire, cryptographiquement sur, et
    garanti unique dans la base (nouvelle tentative en cas de collision, improbable)."""
    while True:
        candidate = f"{prefix}-{secrets.token_hex(5).upper()}"
        exists = conn.execute(
            "SELECT 1 FROM dossiers WHERE dossier_id = ?", (candidate,)
        ).fetchone()
        if not exists:
            return candidate
