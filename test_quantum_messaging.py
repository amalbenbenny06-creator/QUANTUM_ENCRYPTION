import json
import unittest

from cryptography.hazmat.primitives.asymmetric.mlkem import (
    MLKEM768PrivateKey,
    MLKEM768PublicKey,
)

from quantum_messaging import (
    MessageDecryptionError,
    decrypt_message,
    encrypt_message,
    export_encrypted_private_key,
    generate_keypair,
    load_encrypted_private_key,
)


class MessagingCryptoTests(unittest.TestCase):
    def setUp(self):
        self.private, self.public = generate_keypair()

    def encrypt(self, text="hello"):
        return encrypt_message(
            self.public,
            recipient_key_id="user-1-key-1",
            conversation_id="conversation-1",
            message_id="message-1",
            plaintext=text,
        )

    def decrypt(self, envelope):
        return decrypt_message(
            self.private,
            envelope,
            expected_recipient_key_id="user-1-key-1",
            expected_conversation_id="conversation-1",
        )

    def test_round_trip_and_randomized_ciphertexts(self):
        first = self.encrypt("hello 🌍")
        second = self.encrypt("hello 🌍")
        self.assertNotEqual(first, second)
        self.assertEqual(self.decrypt(first), ("message-1", "hello 🌍"))

    def test_raw_mlkem_interoperates_with_library_kem_api(self):
        # Independently exercise the library's FIPS 203 encapsulate/decapsulate
        # interface against the key material exported by this component.
        public = MLKEM768PublicKey.from_public_bytes(self.public)
        shared_sender, kem_ciphertext = public.encapsulate()
        shared_recipient = self.private.decapsulate(kem_ciphertext)
        self.assertEqual(shared_sender, shared_recipient)

    def test_wrong_private_key_fails_generically(self):
        envelope = self.encrypt()
        other_private, _ = generate_keypair()
        with self.assertRaises(MessageDecryptionError) as caught:
            decrypt_message(
                other_private,
                envelope,
                expected_recipient_key_id="user-1-key-1",
                expected_conversation_id="conversation-1",
            )
        self.assertEqual(str(caught.exception), "message could not be decrypted")

    def test_tampering_and_context_mismatch_fail(self):
        envelope = json.loads(self.encrypt())
        envelope["message_id"] = "message-2"
        with self.assertRaises(MessageDecryptionError):
            self.decrypt(json.dumps(envelope).encode())
        with self.assertRaises(MessageDecryptionError):
            decrypt_message(
                self.private,
                self.encrypt(),
                expected_recipient_key_id="other-key",
                expected_conversation_id="conversation-1",
            )

    def test_malformed_and_downgrade_envelopes_fail(self):
        for bad in (b"{", b"{}", b"\xff", b"x" * 2_000_001):
            with self.subTest(bad=bad[:8]), self.assertRaises(MessageDecryptionError):
                self.decrypt(bad)
        obj = json.loads(self.encrypt())
        obj["v"] = 0
        with self.assertRaises(MessageDecryptionError):
            self.decrypt(json.dumps(obj).encode())

    def test_encrypted_private_key_round_trip(self):
        blob = export_encrypted_private_key(self.private, b"long-test-passphrase")
        restored = load_encrypted_private_key(blob, b"long-test-passphrase")
        self.assertIsInstance(restored, MLKEM768PrivateKey)
        self.assertEqual(
            decrypt_message(
                restored,
                self.encrypt(),
                expected_recipient_key_id="user-1-key-1",
                expected_conversation_id="conversation-1",
            ),
            ("message-1", "hello"),
        )
        with self.assertRaises(ValueError):
            load_encrypted_private_key(blob, b"incorrect-passphrase")

    def test_invalid_ids_rejected(self):
        with self.assertRaises(ValueError):
            encrypt_message(
                self.public,
                recipient_key_id="user/one",
                conversation_id="conversation-1",
                message_id="message-1",
                plaintext="hello",
            )


if __name__ == "__main__":
    unittest.main()
