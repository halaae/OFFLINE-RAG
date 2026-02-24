"""
encryption.py — Optional AES-256-GCM document encryption.

When enabled, chunk content stored in metadata.db is base-64 encoded
ciphertext.  FAISS vectors and BM25 tokens are NOT encrypted (they are
mathematical representations, not human-readable text).

Usage
-----
Set the environment variable before running build_index.py:
    export RAG_PASSPHRASE="your-secret-passphrase"
    export ENCRYPT_DOCUMENTS=1
    python build_index.py

And before running main.py:
    export RAG_PASSPHRASE="your-secret-passphrase"
    python main.py

Security notes
--------------
• AES-256-GCM provides both confidentiality AND integrity (detects tampering).
• A random 16-byte salt is generated once and stored in vector_store/enc_salt.bin.
  BACK IT UP — without it the data cannot be decrypted.
• The passphrase is never stored; set it each session via the env var.
• Each chunk uses a fresh 12-byte random nonce (prepended to ciphertext).
• The GCM authentication tag (16 bytes) is appended — do not truncate.
"""

import base64
import logging
import os
from pathlib import Path

log = logging.getLogger("encryption")

SALT_PATH = Path("vector_store/enc_salt.bin")
SALT_SIZE  = 16   # bytes
KEY_SIZE   = 32   # bytes → AES-256
NONCE_SIZE = 12   # bytes → GCM standard

ENCRYPT_ENABLED = os.environ.get("ENCRYPT_DOCUMENTS", "0").strip() not in {"0", "", "false", "no"}


# ---------------------------------------------------------------------------
# Key derivation
# ---------------------------------------------------------------------------

def _get_or_create_salt() -> bytes:
    SALT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if SALT_PATH.exists():
        return SALT_PATH.read_bytes()
    salt = os.urandom(SALT_SIZE)
    SALT_PATH.write_bytes(salt)
    log.info("New encryption salt written to %s — back it up!", SALT_PATH)
    return salt


def _derive_key(passphrase: str) -> bytes:
    """PBKDF2-HMAC-SHA256 key derivation (600 000 iterations per NIST 2023)."""
    import hashlib
    salt = _get_or_create_salt()
    return hashlib.pbkdf2_hmac(
        "sha256",
        passphrase.encode("utf-8"),
        salt,
        iterations=600_000,
        dklen=KEY_SIZE,
    )


# ---------------------------------------------------------------------------
# Module-level key (derived once per process)
# ---------------------------------------------------------------------------

_KEY: bytes | None = None


def _ensure_key():
    global _KEY
    if _KEY is not None:
        return
    passphrase = os.environ.get("RAG_PASSPHRASE", "")
    if not passphrase:
        raise EnvironmentError(
            "Encryption is enabled but RAG_PASSPHRASE is not set.\n"
            "Set it with: export RAG_PASSPHRASE='your-passphrase'"
        )
    _KEY = _derive_key(passphrase)
    log.info("Encryption key derived (AES-256-GCM).")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def encrypt(plaintext: str) -> str:
    """
    Encrypt *plaintext* with AES-256-GCM.
    Returns a base-64 encoded string: nonce(12) + ciphertext + tag(16).
    """
    if not ENCRYPT_ENABLED:
        return plaintext

    _ensure_key()

    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError:
        raise ImportError(
            "cryptography library not installed. "
            "Run: pip install cryptography"
        )

    nonce      = os.urandom(NONCE_SIZE)
    aesgcm     = AESGCM(_KEY)
    ciphertext = aesgcm.encrypt(nonce, plaintext.encode("utf-8"), associated_data=None)
    blob       = nonce + ciphertext   # tag is already appended by AESGCM
    return base64.b64encode(blob).decode("ascii")


def decrypt(ciphertext_b64: str) -> str:
    """
    Decrypt a base-64 blob produced by encrypt().
    Returns the original plaintext string.
    """
    if not ENCRYPT_ENABLED:
        return ciphertext_b64

    _ensure_key()

    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError:
        raise ImportError(
            "cryptography library not installed. "
            "Run: pip install cryptography"
        )

    blob       = base64.b64decode(ciphertext_b64)
    nonce      = blob[:NONCE_SIZE]
    ciphertext = blob[NONCE_SIZE:]
    aesgcm     = AESGCM(_KEY)
    plaintext  = aesgcm.decrypt(nonce, ciphertext, associated_data=None)
    return plaintext.decode("utf-8")