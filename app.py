"""HTTP backend for the post-quantum messaging envelope.

The service is a reference relay for ``quantum_messaging``: it keeps a key
directory, encrypts outgoing messages with ML-KEM-768 HPKE envelopes, stores
them per conversation, and serves one-time decryption.

Endpoints
---------
GET  /health                                            liveness probe
POST /keys                                              provision a recipient keypair
GET  /keys/{key_id}                                     publish a recipient public key
POST /conversations/{conversation_id}/messages          encrypt + store a message
GET  /conversations/{conversation_id}/messages          list stored envelopes
POST /conversations/{conversation_id}/messages/{message_id}/decrypt
                                                        decrypt once (replay-protected)

Operational notes
-----------------
* With ``MESSAGING_KEY_PASSPHRASE`` set (>= 16 bytes) private keys are held
  encrypted at rest with PKCS8 and only decrypted per request. Without it, key
  creation and decryption return 503; health and delivery keep working.
* With ``MESSAGING_API_TOKEN`` set, every endpoint except ``/health`` requires
  ``Authorization: Bearer <token>``. Without it the API is open, so bind it to
  localhost in development.
* Messages and keys live in process memory; restarting the service drops them,
  so run a single worker until a durable store is wired in.
* Because the server holds recipient private keys, this is server-side
  encryption, not end-to-end. For end-to-end delivery, clients should fetch the
  public key and hand the opaque envelope to the server instead of plaintext.
"""

from __future__ import annotations

import base64
import json
import os
import secrets
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from quantum_messaging import (
    MessageDecryptionError,
    decrypt_message,
    decrypt,
    encrypt,
    encrypt_message,
    export_encrypted_private_key,
    generate_keypair,
    load_encrypted_private_key,
    validate_identifier,
)

# Local-development secret: replace this with your own private value.
# Keep it unchanged between restarts or existing encrypted private keys
# will no longer be decryptable. Never commit a real production secret.
MESSAGING_KEY_PASSPHRASE = "MvFoiIXX2FNtW073k5JUl8h5W2bodA8VygTJZiL1wm8"

TOKEN_ENV = "MESSAGING_API_TOKEN"
MIN_PASSPHRASE_BYTES = 16
MAX_PLAINTEXT_CHARS = 16_000
GENERIC_DECRYPTION_ERROR = "message could not be decrypted"


# --------------------------------------------------------------------------- #
# Storage
# --------------------------------------------------------------------------- #


@dataclass
class StoredKey:
    key_id: str
    public_key: bytes
    encrypted_private_key: bytes
    created_at: float


@dataclass
class StoredMessage:
    message_id: str
    conversation_id: str
    recipient_key_id: str
    envelope: bytes
    created_at: float
    delivered_at: float | None = None


class MessageStore:
    """Thread-safe in-memory key directory and message log."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._keys: dict[str, StoredKey] = {}
        self._messages: dict[tuple[str, str], StoredMessage] = {}
        self._by_conversation: dict[str, list[str]] = {}

    def add_key(self, key: StoredKey) -> None:
        with self._lock:
            self._keys[key.key_id] = key

    def get_key(self, key_id: str) -> StoredKey | None:
        with self._lock:
            return self._keys.get(key_id)

    def add_message(self, message: StoredMessage) -> bool:
        """Store a message; returns False when the message_id was already used."""
        with self._lock:
            slot = (message.conversation_id, message.message_id)
            if slot in self._messages:
                return False
            self._messages[slot] = message
            self._by_conversation.setdefault(message.conversation_id, []).append(
                message.message_id
            )
            return True

    def get_message(self, conversation_id: str, message_id: str) -> StoredMessage | None:
        with self._lock:
            return self._messages.get((conversation_id, message_id))

    def list_messages(self, conversation_id: str) -> list[StoredMessage]:
        with self._lock:
            ids = list(self._by_conversation.get(conversation_id, []))
            return [self._messages[(conversation_id, mid)] for mid in ids]

    def mark_delivered(self, conversation_id: str, message_id: str) -> bool:
        """Claim a message for delivery; False when it was already delivered."""
        with self._lock:
            message = self._messages.get((conversation_id, message_id))
            if message is None or message.delivered_at is not None:
                return False
            message.delivered_at = time.time()
            return True


store = MessageStore()

app = FastAPI(
    title="Post-Quantum Messaging Backend",
    version="1.0.0",
)

# CORS: allow the separately hosted frontend to call this API during development.
# For production, replace ["*"] with the exact frontend origin(s).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Request/response models
# --------------------------------------------------------------------------- #


class KeyResponse(BaseModel):
    key_id: str
    algorithm: str
    public_key: str
    created_at: str


class SendMessageRequest(BaseModel):
    recipient_key_id: str = Field(min_length=1, max_length=200)
    plaintext: str = Field(max_length=MAX_PLAINTEXT_CHARS)
    message_id: str | None = Field(default=None, min_length=1, max_length=200)


class MessageResponse(BaseModel):
    message_id: str
    conversation_id: str
    recipient_key_id: str
    envelope: dict[str, Any]
    created_at: str
    delivered: bool


class DecryptResponse(BaseModel):
    message_id: str
    conversation_id: str
    plaintext: str


# --------------------------------------------------------------------------- #
# Dependencies
# --------------------------------------------------------------------------- #

_bearer = HTTPBearer(auto_error=False)


def require_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> None:
    """Enforce the shared bearer token when one is configured."""
    expected = os.environ.get(TOKEN_ENV)
    if not expected:
        return
    supplied = credentials.credentials if credentials else ""
    if not supplied or not secrets.compare_digest(supplied, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid or missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )


def custody_passphrase() -> bytes:
    passphrase = MESSAGING_KEY_PASSPHRASE.encode("utf-8")

    if len(passphrase) < MIN_PASSPHRASE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "MESSAGING_KEY_PASSPHRASE in app.py must contain at least "
                f"{MIN_PASSPHRASE_BYTES} bytes"
            ),
        )

    return passphrase


def _iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def _checked_id(value: str, field: str) -> str:
    try:
        return validate_identifier(value, field)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"invalid {field}") from None


def _to_response(message: StoredMessage) -> MessageResponse:
    return MessageResponse(
        message_id=message.message_id,
        conversation_id=message.conversation_id,
        recipient_key_id=message.recipient_key_id,
        envelope=json.loads(message.envelope),
        created_at=_iso(message.created_at),
        delivered=message.delivered_at is not None,
    )


def _not_found(what: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"{what} not found")


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/keys",
    status_code=status.HTTP_201_CREATED,
    response_model=KeyResponse,
    dependencies=[Depends(require_token)],
)
def create_key() -> KeyResponse:
    """Generate a recipient keypair; the private key stays only in encrypted form."""
    passphrase = custody_passphrase()
    private_key, public_key = generate_keypair()
    key_id = f"key-{uuid.uuid4().hex}"
    created_at = time.time()
    store.add_key(
        StoredKey(
            key_id=key_id,
            public_key=public_key,
            encrypted_private_key=export_encrypted_private_key(private_key, passphrase),
            created_at=created_at,
        )
    )
    return KeyResponse(
        key_id=key_id,
        algorithm="MLKEM768-HKDF-SHA256-AES256GCM",
        public_key=base64.b64encode(public_key).decode("ascii"),
        created_at=_iso(created_at),
    )


class EncryptRequest(BaseModel):
    recipient_public_key: str = Field(..., description="recipient ML-KEM-768 public key, base64")
    plaintext: str = Field(..., max_length=MAX_PLAINTEXT_CHARS)


class EncryptResponse(BaseModel):
    ciphertext: str


@app.post(
    "/encrypt",
    status_code=status.HTTP_200_OK,
    response_model=EncryptResponse,
    dependencies=[Depends(require_token)],
)

def encrypt_plaintext(request: EncryptRequest) -> EncryptResponse:
    """Encrypt the plaintext and return only the ciphertext. Nothing else."""
    public_key = base64.b64decode(request.recipient_public_key, validate=True)
    ciphertext = encrypt(request.plaintext, public_key)
    return EncryptResponse(ciphertext=ciphertext)


@app.get(
    "/keys/{key_id}",
    response_model=KeyResponse,
    dependencies=[Depends(require_token)],
)
def get_public_key(key_id: str) -> KeyResponse:
    """Publish a recipient public key so senders can encrypt to it."""
    key = store.get_key(_checked_id(key_id, "key_id"))
    if key is None:
        raise _not_found("key")
    return KeyResponse(
        key_id=key.key_id,
        algorithm="MLKEM768-HKDF-SHA256-AES256GCM",
        public_key=base64.b64encode(key.public_key).decode("ascii"),
        created_at=_iso(key.created_at),
    )


@app.post(
    "/conversations/{conversation_id}/messages",
    status_code=status.HTTP_201_CREATED,
    response_model=MessageResponse,
    dependencies=[Depends(require_token)],
)
def send_message(conversation_id: str, request: SendMessageRequest) -> MessageResponse:
    """Encrypt plaintext into an HPKE envelope and queue it for the recipient."""
    conversation_id = _checked_id(conversation_id, "conversation_id")
    recipient_key_id = _checked_id(request.recipient_key_id, "recipient_key_id")
    message_id = _checked_id(request.message_id or f"msg-{uuid.uuid4().hex}", "message_id")

    key = store.get_key(recipient_key_id)
    if key is None:
        raise _not_found("recipient key")

    try:
        envelope = encrypt_message(
            key.public_key,
            recipient_key_id=recipient_key_id,
            conversation_id=conversation_id,
            message_id=message_id,
            plaintext=request.plaintext,
        )
    except ValueError:
        raise HTTPException(status_code=422, detail="invalid message") from None

    message = StoredMessage(
        message_id=message_id,
        conversation_id=conversation_id,
        recipient_key_id=recipient_key_id,
        envelope=envelope,
        created_at=time.time(),
    )
    if not store.add_message(message):
        # Replaying a message_id would let an attacker force a duplicate delivery.
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="message_id already used in this conversation",
        )
    return _to_response(message)


@app.get(
    "/conversations/{conversation_id}/messages",
    response_model=list[MessageResponse],
    dependencies=[Depends(require_token)],
)
def list_messages(conversation_id: str) -> list[MessageResponse]:
    """Return stored envelopes newest-first; plaintext is never included."""
    conversation_id = _checked_id(conversation_id, "conversation_id")
    return [_to_response(message) for message in reversed(store.list_messages(conversation_id))]


@app.post(
    "/conversations/{conversation_id}/messages/{message_id}/decrypt",
    response_model=DecryptResponse,
    dependencies=[Depends(require_token)],
)
def decrypt(conversation_id: str, message_id: str) -> DecryptResponse:
    """Decrypt a message exactly once; replays and tampering fail generically."""
    conversation_id = _checked_id(conversation_id, "conversation_id")
    message_id = _checked_id(message_id, "message_id")
    passphrase = custody_passphrase()

    message = store.get_message(conversation_id, message_id)
    if message is None:
        raise _not_found("message")
    if message.delivered_at is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="message was already delivered",
        )

    key = store.get_key(message.recipient_key_id)
    if key is None:
        raise _not_found("recipient key")

    try:
        private_key = load_encrypted_private_key(key.encrypted_private_key, passphrase)
        decrypted_id, plaintext = decrypt_message(
            private_key,
            message.envelope,
            expected_recipient_key_id=message.recipient_key_id,
            expected_conversation_id=message.conversation_id,
        )
    except (ValueError, MessageDecryptionError):
        # Same response for bad keys, tampering and malformed envelopes.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=GENERIC_DECRYPTION_ERROR
        ) from None

    if decrypted_id != message.message_id or not store.mark_delivered(
        conversation_id, message_id
    ):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="message was already delivered",
        )

    return DecryptResponse(
        message_id=message_id, conversation_id=conversation_id, plaintext=plaintext
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8000")),
    )
