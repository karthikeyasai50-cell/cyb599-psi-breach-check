"""Item encoding shared by every protocol (Part 2, Section 2).

x = SHA-256("TADS-PSI-v1" || type || normalized)
"""

import hashlib
import unicodedata

DOMAIN_TAG = b"TADS-PSI-v1"
KINDS = ("email", "password")


def normalize(value: str, kind: str) -> str:
    """Emails are trimmed, lowercased and NFKC-normalized; passwords are kept as given."""
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}")
    if kind == "email":
        return unicodedata.normalize("NFKC", value).strip().lower()
    return value


def encode_item(value: str, kind: str) -> bytes:
    """Return the 32-byte domain-separated encoding of one credential."""
    norm = normalize(value, kind).encode("utf-8")
    # A separator byte between type and value keeps the encoding unambiguous.
    return hashlib.sha256(DOMAIN_TAG + kind.encode() + b"\x00" + norm).digest()
