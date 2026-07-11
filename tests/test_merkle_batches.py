"""Tests for receipt Merkle roots and inclusion proofs."""

from __future__ import annotations

import sqlite3
import unittest

from server import merkle_batches


class MerkleBatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_get_db = getattr(merkle_batches, "get_db", None)
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        merkle_batches.get_db = lambda: self.conn
        self.conn.execute(
            """
            CREATE TABLE proof_of_creation_receipts (
                job_id TEXT PRIMARY KEY,
                receipt_hash TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        merkle_batches.init_merkle_tables(self.conn)

    def tearDown(self) -> None:
        if self.original_get_db is None:
            delattr(merkle_batches, "get_db")
        else:
            merkle_batches.get_db = self.original_get_db
        self.conn.close()

    def _seed_receipts(self, count: int) -> list[str]:
        hashes = []
        for index in range(count):
            receipt_hash = f"{index + 1:064x}"
            hashes.append(receipt_hash)
            self.conn.execute(
                "INSERT INTO proof_of_creation_receipts VALUES (?, ?, ?)",
                (f"job-{index}", receipt_hash, float(index)),
            )
        self.conn.commit()
        return hashes

    def test_odd_sized_tree_produces_valid_proof_for_every_leaf(self) -> None:
        receipt_hashes = self._seed_receipts(3)
        root, proofs = merkle_batches.build_tree(receipt_hashes)

        self.assertEqual(len(proofs), 3)
        for receipt_hash, proof in zip(receipt_hashes, proofs):
            self.assertTrue(merkle_batches.verify_proof(receipt_hash, proof, root))
        self.assertFalse(merkle_batches.verify_proof(f"{99:064x}", proofs[0], root))

    def test_batch_is_immutable_and_receipts_are_not_rebatched(self) -> None:
        self._seed_receipts(3)
        first = merkle_batches.create_batch(limit=2, min_count=2)
        second = merkle_batches.create_batch(limit=2, min_count=1)
        empty = merkle_batches.create_batch(limit=2, min_count=1)

        self.assertEqual(first["leaf_count"], 2)
        self.assertEqual(second["leaf_count"], 1)
        self.assertIsNone(empty)
        self.assertNotEqual(first["merkle_root"], second["merkle_root"])

        proof = merkle_batches.get_inclusion_proof("job-2")
        self.assertTrue(proof["valid"])
        self.assertEqual(proof["batch_id"], second["id"])

    def test_batch_waits_for_configured_minimum(self) -> None:
        self._seed_receipts(2)
        self.assertIsNone(merkle_batches.create_batch(limit=3, min_count=3))
        self.assertEqual(len(merkle_batches.list_batches()), 0)


if __name__ == "__main__":
    unittest.main()
