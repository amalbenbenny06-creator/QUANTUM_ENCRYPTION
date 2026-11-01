import base64
import json
import os
import unittest

from cryptography.hazmat.primitives.asymmetric.mlkem import MLKEM768PublicKey
from fastapi.testclient import TestClient

import app as backend

PASSPHRASE = "test-passphrase-0123456789"
TOKEN = "test-api-token"


class MessagingApiTests(unittest.TestCase):
    def setUp(self):
        os.environ[backend.PASSPHRASE_ENV] = PASSPHRASE
        os.environ.pop(backend.TOKEN_ENV, None)
        backend.store = backend.MessageStore()
        self.client = TestClient(backend.app)
        self.key = self.create_key()

    def tearDown(self):
        os.environ.pop(backend.PASSPHRASE_ENV, None)
        os.environ.pop(backend.TOKEN_ENV, None)

    def create_key(self):
        response = self.client.post("/keys")
        self.assertEqual(response.status_code, 201, response.text)
        body = response.json()
        self.assertNotIn("private", json.dumps(body).lower())
        return body

    def send(self, text="hello 🌍", message_id="message-1", conversation_id="conversation-1"):
        return self.client.post(
            f"/conversations/{conversation_id}/messages",
            json={
                "recipient_key_id": self.key["key_id"],
                "plaintext": text,
                "message_id": message_id,
            },
        )

    def decrypt(self, message_id="message-1", conversation_id="conversation-1"):
        return self.client.post(
            f"/conversations/{conversation_id}/messages/{message_id}/decrypt"
        )

    def test_health_and_published_public_key(self):
        self.assertEqual(self.client.get("/health").json(), {"status": "ok"})
        published = self.client.get(f"/keys/{self.key['key_id']}")
        self.assertEqual(published.status_code, 200)
        body = published.json()
        self.assertEqual(body["public_key"], self.key["public_key"])
        self.assertEqual(body["algorithm"], "MLKEM768-HKDF-SHA256-AES256GCM")
        raw = base64.b64decode(body["public_key"], validate=True)
        self.assertIsInstance(MLKEM768PublicKey.from_public_bytes(raw), MLKEM768PublicKey)

    def test_send_then_decrypt_exactly_once(self):
        sent = self.send()
        self.assertEqual(sent.status_code, 201, sent.text)
        body = sent.json()
        self.assertEqual(body["message_id"], "message-1")
        self.assertFalse(body["delivered"])
        self.assertNotIn("hello", json.dumps(body["envelope"]))
        self.assertEqual(set(body["envelope"]), {
            "v", "suite", "recipient_key_id", "conversation_id", "message_id", "ciphertext",
        })

        first = self.decrypt()
        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(first.json()["plaintext"], "hello 🌍")

        replay = self.decrypt()
        self.assertEqual(replay.status_code, 409)
        self.assertNotIn("hello", replay.text)

        listed = self.client.get("/conversations/conversation-1/messages").json()
        self.assertEqual(len(listed), 1)
        self.assertTrue(listed[0]["delivered"])
        self.assertNotIn("hello", json.dumps(listed))

    def test_duplicate_message_id_rejected(self):
        self.assertEqual(self.send().status_code, 201)
        duplicate = self.send(text="second")
        self.assertEqual(duplicate.status_code, 409)
        self.assertEqual(duplicate.json()["detail"], "message_id already used in this conversation")

    def test_unknown_recipient_and_bad_conversation_id(self):
        missing = self.client.post(
            "/conversations/conversation-1/messages",
            json={"recipient_key_id": "key-does-not-exist", "plaintext": "hi"},
        )
        self.assertEqual(missing.status_code, 404)
        bad_id = self.send(conversation_id="bad id!")
        self.assertEqual(bad_id.status_code, 422)
        self.assertEqual(self.client.get("/keys/bad%20id").status_code, 422)

    def test_tampered_envelope_fails_generically_and_stays_undelivered(self):
        message_id = self.send().json()["message_id"]
        stored = backend.store.get_message("conversation-1", message_id)
        envelope = json.loads(stored.envelope)
        ciphertext = envelope["ciphertext"]
        envelope["ciphertext"] = ciphertext[:5] + ("A" if ciphertext[5] != "A" else "B") + ciphertext[6:]
        stored.envelope = json.dumps(envelope).encode()

        response = self.decrypt()
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "message could not be decrypted")
        self.assertIsNone(stored.delivered_at)

    def test_passphrase_required_for_key_custody(self):
        os.environ.pop(backend.PASSPHRASE_ENV, None)
        self.assertEqual(self.client.post("/keys").status_code, 503)
        message_id = self.send().json()["message_id"]
        self.assertEqual(self.decrypt(message_id).status_code, 503)

    def test_bearer_token_enforced_on_non_health_routes(self):
        os.environ[backend.TOKEN_ENV] = TOKEN
        self.assertEqual(self.client.get("/health").status_code, 200)
        self.assertEqual(self.client.get(f"/keys/{self.key['key_id']}").status_code, 401)

        headers = {"Authorization": f"Bearer {TOKEN}"}
        created = self.client.post("/keys", headers=headers)
        self.assertEqual(created.status_code, 201, created.text)
        wrong = self.client.post("/keys", headers={"Authorization": "Bearer nope"})
        self.assertEqual(wrong.status_code, 401)

        sent = self.client.post(
            "/conversations/conversation-1/messages",
            json={"recipient_key_id": self.key["key_id"], "plaintext": "hi"},
            headers=headers,
        )
        self.assertEqual(sent.status_code, 201, sent.text)
        self.assertEqual(self.client.get("/conversations/conversation-1/messages").status_code, 401)


if __name__ == "__main__":
    unittest.main()
