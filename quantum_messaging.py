"""Small HPKE envelope helper for text-messaging integrations.

This module uses RFC 9180 HPKE with the FIPS 203 ML-KEM-768 KEM,
HKDF-SHA256, and AES-256-GCM. It is an encryption building block, not a
complete messaging protocol: sender authentication and replay prevention
must be provided by the calling application.
"""

from __future__ import annotations

import base64
import binascii
import json
import re
from dataclasses import dataclass
from typing import Any

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.mlkem import MLKEM768PrivateKey
from cryptography.hazmat.primitives.serialization import BestAvailableEncryption
from cryptography.hazmat.primitives.hpke import AEAD, KDF, KEM, Suite


PROTOCOL_VERSION = 1
SUITE_NAME = "MLKEM768-HKDF-SHA256-AES256GCM"
_INFO_PREFIX = b"text-messaging/hpke/v1/"
_MAX_ENVELOPE_BYTES = 2_000_000
_ID_RE = re.compile(r"^[A-Za-z0-9._:@-]{1,200}$")
_SUITE = Suite(KEM.MLKEM768, KDF.HKDF_SHA256, AEAD.AES_256_GCM)


class MessageDecryptionError(Exception):
    """Generic failure; callers should return the same external error for all causes."""


@dataclass(frozen=True)
class Envelope:
    version: int
    suite: str
    recipient_key_id: str
    conversation_id: str
    message_id: str
    ciphertext: str

    def to_bytes(self) -> bytes:
        return json.dumps(
            {
                "v": self.version,
                "suite": self.suite,
                "recipient_key_id": self.recipient_key_id,
                "conversation_id": self.conversation_id,
                "message_id": self.message_id,
                "ciphertext": self.ciphertext,
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")


def generate_keypair() -> tuple[MLKEM768PrivateKey, bytes]:
    """Generate a recipient keypair; publish the returned raw public key securely."""
    private_key = MLKEM768PrivateKey.generate()
    return private_key, private_key.public_key().public_bytes_raw()


# Plain-text encryption (no envelope / no metadata). Returns the raw
# base64 ciphertext so the caller can store/forward it as-is.


def encrypt(plaintext: str, recipient_public_key: bytes) -> str:
    """Encrypt UTF-8 text and return the raw base64 ciphertext.

    No envelope, no metadata: this is the smallest possible encrypt() for
    storing or forwarding an encrypted blob directly.
    """
    if not isinstance(plaintext, str):
        raise TypeError("plaintext must be str")
    from cryptography.hazmat.primitives.asymmetric.mlkem import MLKEM768PublicKey

    public_key = MLKEM768PublicKey.from_public_bytes(recipient_public_key)
    ciphertext = _SUITE.encrypt(plaintext.encode("utf-8"), public_key)
    return base64.b64encode(ciphertext).decode("ascii")



# Plain-text encryption (no envelope / no metadata). Returns the raw
# base64 ciphertext so the caller can store/forward it as-is.


def encrypt(plaintext: str, recipient_public_key: bytes) -> str:
    """Encrypt UTF-8 text and return the raw base64 ciphertext.

    No envelope, no metadata: this is the smallest possible encrypt() for
    storing or forwarding an encrypted blob directly.
    """
    if not isinstance(plaintext, str):
        raise TypeError("plaintext must be str")
    from cryptography.hazmat.primitives.asymmetric.mlkem import MLKEM768PublicKey

    public_key = MLKEM768PublicKey.from_public_bytes(recipient_public_key)
    ciphertext = _SUITE.encrypt(plaintext.encode("utf-8"), public_key)
    return base64.b64encode(ciphertext).decode("ascii")





def export_encrypted_private_key(
    private_key: MLKEM768PrivateKey, passphrase: bytes
) -> bytes:
    """Export PKCS8 encrypted with a passphrase. Prefer a KMS/HSM in production."""
    if not isinstance(passphrase, bytes) or len(passphrase) < 16:
        raise ValueError("passphrase must contain at least 16 bytes")
    return private_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        BestAvailableEncryption(passphrase),
    )


def load_encrypted_private_key(
    pem: bytes, passphrase: bytes
) -> MLKEM768PrivateKey:
    """Load an encrypted PKCS8 ML-KEM-768 key without exposing parsing details."""
    try:
        key = serialization.load_pem_private_key(pem, password=passphrase)
        if not isinstance(key, MLKEM768PrivateKey):
            raise ValueError("wrong key type")
        return key
    except Exception:
        raise ValueError("invalid encrypted ML-KEM private key") from None


def validate_identifier(value: str, field: str = "identifier") -> str:
    """Check an id against the envelope's safe alphabet; returns it unchanged."""
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise ValueError(f"invalid {field}")
    return value


_validate_id = validate_identifier


def _info(recipient_key_id: str, conversation_id: str, message_id: str) -> bytes:
    # Identifiers are restricted to an ASCII-safe alphabet, avoiding ambiguous encoding.
    return (
        _INFO_PREFIX
        + recipient_key_id.encode("ascii")
        + b"/"
        + conversation_id.encode("ascii")
        + b"/"
        + message_id.encode("ascii")
    )


def encrypt_message(
    recipient_public_key: bytes,
    *,
    recipient_key_id: str,
    conversation_id: str,
    message_id: str,
    plaintext: str,
) -> bytes:
    """Encrypt UTF-8 text into a versioned HPKE envelope.

    The caller MUST verify/bind recipient_public_key to the recipient identity
    (for example via an authenticated account key directory or key pinning).
    Generate a unique message_id and enforce replay rejection when receiving.
    """
    recipient_key_id = _validate_id(recipient_key_id, "recipient_key_id")
    conversation_id = _validate_id(conversation_id, "conversation_id")
    message_id = _validate_id(message_id, "message_id")
    if not isinstance(plaintext, str):
        raise TypeError("plaintext must be str")
    from cryptography.hazmat.primitives.asymmetric.mlkem import MLKEM768PublicKey

    public_key = MLKEM768PublicKey.from_public_bytes(recipient_public_key)
    context = _info(recipient_key_id, conversation_id, message_id)
    ciphertext = _SUITE.encrypt(plaintext.encode("utf-8"), public_key, info=context)
    return Envelope(
        PROTOCOL_VERSION,
        SUITE_NAME,
        recipient_key_id,
        conversation_id,
        message_id,
        base64.b64encode(ciphertext).decode("ascii"),
    ).to_bytes()


def decrypt(
    encrypted_b64: str,
    private_key: MLKEM768PrivateKey,
) -> str:
    """Decrypt a base64-encoded ciphertext and return the plaintext.

    Lowest-level decrypt() for the plain-text API: takes the raw base64
    ciphertext (as returned by :func:`encrypt`) and the private key,
    returns the plaintext string.
    """
    ciphertext = base64.b64decode(encrypted_b64, validate=True)
    # Direct raw-KEM decrypt: no envelope context binds. The HPKE suite
    # already binds the recipient public key and message context inside
    # the AEAD authenticator, so tampering is detected by decryption failure.
    plaintext = _SUITE.decrypt(ciphertext, private_key)
    return plaintext.decode("utf-8")


def decrypt_message(
    private_key: MLKEM768PrivateKey,
    envelope_bytes: bytes,
    *,
    expected_recipient_key_id: str,
    expected_conversation_id: str,
) -> tuple[str, str]:
    """Return (message_id, plaintext); use a generic error for any failure.

    The caller must atomically reject/store message_id to prevent replay. Keep
    old private keys available only as long as required to decrypt queued/history
    messages, then securely retire them through the key-management system.
    """
    try:
        if len(envelope_bytes) > _MAX_ENVELOPE_BYTES:
            raise ValueError
        obj: Any = json.loads(envelope_bytes)
        if not isinstance(obj, dict) or set(obj) != {
            "v", "suite", "recipient_key_id", "conversation_id", "message_id", "ciphertext"
        }:
            raise ValueError
        if obj["v"] != PROTOCOL_VERSION or obj["suite"] != SUITE_NAME:
            raise ValueError
        recipient_key_id = _validate_id(obj["recipient_key_id"], "recipient_key_id")
        conversation_id = _validate_id(obj["conversation_id"], "conversation_id")
        message_id = _validate_id(obj["message_id"], "message_id")
        if recipient_key_id != _validate_id(expected_recipient_key_id, "expected key id"):
            raise ValueError
        if conversation_id != _validate_id(expected_conversation_id, "expected conversation id"):
            raise ValueError
        if not isinstance(obj["ciphertext"], str):
            raise ValueError
        ciphertext = base64.b64decode(obj["ciphertext"], validate=True)
        plaintext = _SUITE.decrypt(
            ciphertext,
            private_key,
            info=_info(recipient_key_id, conversation_id, message_id),
        )
        return message_id, plaintext.decode("utf-8")
    except Exception:
        # One application-level failure surface; never log the underlying exception.
        raise MessageDecryptionError("message could not be decrypted") from None

__all__ = [
    "Envelope",
    "MessageDecryptionError",
    "encrypt",
    "decrypt",
    "decrypt",
    "encrypt_message",
    "decrypt_message",
    "export_encrypted_private_key",
    "generate_keypair",
    "load_encrypted_private_key",
    "validate_identifier",
]
