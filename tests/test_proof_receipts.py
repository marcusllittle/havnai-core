"""Tests for tamper-evident Proof of Creation receipts."""

from __future__ import annotations

import sqlite3
import base64
import tempfile
import unittest
from pathlib import Path

from server import proof_receipts


class ProofReceiptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_get_db = getattr(proof_receipts, "get_db", None)
        self.original_signing_key = proof_receipts.SIGNING_KEY
        self.original_ed25519_key = proof_receipts.ED25519_PRIVATE_KEY
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        proof_receipts.get_db = lambda: self.conn
        proof_receipts.SIGNING_KEY = "test-receipt-secret"
        proof_receipts.ED25519_PRIVATE_KEY = ""
        proof_receipts.init_receipt_tables(self.conn)
        self.tmpdir = tempfile.TemporaryDirectory()
        self.artifact = Path(self.tmpdir.name) / "job-1.png"
        self.artifact.write_bytes(b"havnai-proof-artifact")

    def tearDown(self) -> None:
        if self.original_get_db is None:
            delattr(proof_receipts, "get_db")
        else:
            proof_receipts.get_db = self.original_get_db
        proof_receipts.SIGNING_KEY = self.original_signing_key
        proof_receipts.ED25519_PRIVATE_KEY = self.original_ed25519_key
        self.conn.close()
        self.tmpdir.cleanup()

    def _payload(self) -> dict:
        return {
            "creator": {"wallet": "0x1111111111111111111111111111111111111111"},
            "model": {"name": "network-model", "pipeline": "sdxl"},
            "timeline": {"sha256": "abc123", "through_sequence": 8},
            "settlement": {"outcome": "spent"},
            "artifact": {"filename": "job-1.png", "media_type": "image/png"},
        }

    def test_signed_receipt_verifies_and_is_idempotent(self) -> None:
        first = proof_receipts.create_receipt("job-1", self._payload(), artifact_path=self.artifact)
        second = proof_receipts.create_receipt("job-1", {"different": True}, artifact_path=self.artifact)
        verification = proof_receipts.verify_receipt("job-1")

        self.assertEqual(first["receipt_hash"], second["receipt_hash"])
        self.assertTrue(first["signed"])
        self.assertEqual(first["signature_algorithm"], "hmac-sha256")
        self.assertTrue(verification["valid"])
        self.assertTrue(verification["signature_valid"])
        self.assertEqual(verification["authenticity"], "verified")

    def test_artifact_tampering_invalidates_receipt(self) -> None:
        proof_receipts.create_receipt("job-1", self._payload(), artifact_path=self.artifact)
        self.artifact.write_bytes(b"tampered")

        verification = proof_receipts.verify_receipt("job-1")

        self.assertFalse(verification["valid"])
        self.assertFalse(verification["artifact_valid"])
        self.assertTrue(verification["hash_valid"])

    def test_payload_tampering_invalidates_hash_and_signature(self) -> None:
        proof_receipts.create_receipt("job-1", self._payload(), artifact_path=self.artifact)
        row = self.conn.execute(
            "SELECT payload FROM proof_of_creation_receipts WHERE job_id='job-1'"
        ).fetchone()
        tampered = str(row["payload"]).replace("network-model", "other-model")
        self.conn.execute(
            "UPDATE proof_of_creation_receipts SET payload=? WHERE job_id='job-1'", (tampered,)
        )
        self.conn.commit()

        verification = proof_receipts.verify_receipt("job-1")

        self.assertFalse(verification["valid"])
        self.assertFalse(verification["hash_valid"])

    def test_ed25519_receipt_verifies_using_only_public_key(self) -> None:
        proof_receipts.ED25519_PRIVATE_KEY = base64.b64encode(bytes(range(32))).decode("ascii")
        receipt = proof_receipts.create_receipt("job-1", self._payload(), artifact_path=self.artifact)
        published_key = proof_receipts.public_signing_key()
        proof_receipts.ED25519_PRIVATE_KEY = ""
        proof_receipts.SIGNING_KEY = ""

        verification = proof_receipts.verify_receipt("job-1")

        self.assertEqual(receipt["signature_algorithm"], "ed25519")
        self.assertEqual(receipt["key_id"], published_key["key_id"])
        self.assertEqual(receipt["public_key"], published_key["public_key"])
        self.assertTrue(verification["valid"])
        self.assertTrue(verification["signature_valid"])
        self.assertEqual(verification["authenticity"], "verified")


if __name__ == "__main__":
    unittest.main()
