"""Tests for immutable, operator-addressed node payout claim batches."""

from __future__ import annotations

import importlib.util
import sqlite3
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "havnai_server_payout_claims", ROOT / "server" / "payout_claims.py"
)
assert SPEC is not None and SPEC.loader is not None
claims = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = claims
SPEC.loader.exec_module(claims)

WALLET_A = "0x" + "11" * 20
WALLET_B = "0x" + "22" * 20


class PayoutClaimTests(unittest.TestCase):
    def setUp(self) -> None:
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(
            """
            CREATE TABLE node_payouts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                node_id TEXT NOT NULL,
                job_id TEXT NOT NULL,
                reward_amount REAL NOT NULL,
                reward_asset_type TEXT NOT NULL,
                status TEXT NOT NULL,
                tx_hash TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE TABLE node_wallets (
                node_id TEXT PRIMARY KEY,
                wallet TEXT,
                node_name TEXT,
                updated_at REAL
            );
            """
        )
        claims.get_db = lambda: self.conn
        claims.init_payout_claim_tables(self.conn)
        self.conn.executemany(
            "INSERT INTO node_wallets VALUES (?, ?, ?, 1)",
            [
                ("node-a", WALLET_A, "Node A"),
                ("node-a2", WALLET_A.upper().replace("0X", "0x"), "Node A2"),
                ("node-b", WALLET_B, "Node B"),
                ("node-invalid", "not-a-wallet", "Bad Node"),
            ],
        )

    def tearDown(self) -> None:
        self.conn.close()

    def _payout(
        self,
        node_id: str,
        amount: float,
        *,
        asset: str = "simulated_hai",
        status: str = "completed",
        tx_hash: str | None = None,
    ) -> int:
        cursor = self.conn.execute(
            """INSERT INTO node_payouts
               (node_id, job_id, reward_amount, reward_asset_type, status,
                tx_hash, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                node_id,
                f"job-{node_id}-{amount}",
                amount,
                asset,
                status,
                tx_hash,
                amount,
                amount,
            ),
        )
        self.conn.commit()
        return int(cursor.lastrowid)

    def test_leaf_encoding_matches_solidity_abi_vector(self) -> None:
        self.assertEqual(
            claims.payout_leaf(7, 0, WALLET_A, 400).hex(),
            "d8ade8d9391cc10edce8f6248103c34057d04f64bf6de4d203b2e18b9b835b72",
        )
        self.assertEqual(
            claims.payout_leaf(7, 1, WALLET_B, 600).hex(),
            "e334f55263b1a59996685f0f5590e30ae6106233e445272eea528a864784b669",
        )

    def test_amount_conversion_never_rounds_payout_up(self) -> None:
        self.assertEqual(claims.amount_to_wei("1.2345679"), 1_234_567_000_000_000_000)
        self.assertEqual(claims.format_wei(1_234_567_000_000_000_000), "1.234567")
        with self.assertRaises(claims.PayoutClaimError):
            claims.amount_to_wei("0.0000009")

    def test_batch_groups_payouts_by_durable_operator_wallet(self) -> None:
        payout_a1 = self._payout("node-a", 1.25)
        payout_a2 = self._payout("node-a2", 2.5)
        payout_b = self._payout("node-b", 1.0)
        self._payout("node-invalid", 99)
        self._payout("node-b", 99, status="pending")
        self._payout("node-b", 99, asset="onchain_hai", tx_hash="0x" + "9" * 64)

        batch = claims.create_batch()

        self.assertIsNotNone(batch)
        self.assertEqual(batch["leaf_count"], 2)
        self.assertEqual(batch["payout_count"], 3)
        self.assertEqual(batch["total_amount_hai"], "4.75")
        self.assertEqual(claims.count_unbatched_payouts(), 0)
        self.assertIsNone(claims.create_batch())

        wallet_claims = claims.get_wallet_claims(WALLET_A)
        self.assertEqual(len(wallet_claims), 1)
        claim = wallet_claims[0]
        self.assertEqual(claim["amount_hai"], "3.75")
        self.assertEqual(claim["payout_count"], 2)
        self.assertEqual(claim["node_ids"], ["node-a", "node-a2"])
        self.assertTrue(claim["valid"])
        self.assertTrue(all(len(item) == 64 for item in claim["proof"]))

        linked = self.conn.execute(
            "SELECT payout_id FROM node_payout_claim_items ORDER BY payout_id"
        ).fetchall()
        self.assertEqual([row[0] for row in linked], [payout_a1, payout_a2, payout_b])

    def test_odd_tree_proofs_verify_for_every_operator(self) -> None:
        self.conn.execute(
            "INSERT INTO node_wallets VALUES ('node-c', ?, 'Node C', 1)",
            ("0x" + "33" * 20,),
        )
        for node_id in ("node-a", "node-b", "node-c"):
            self._payout(node_id, 1.0)
        batch = claims.create_batch()
        self.assertIsNotNone(batch)

        for index in range(3):
            claim = claims.get_claim(batch["batch_id"], index)
            self.assertIsNotNone(claim)
            self.assertTrue(claim["valid"])

    def test_publish_and_claim_transitions_are_idempotent_and_conflict_safe(self) -> None:
        payout_a1 = self._payout("node-a", 1.25)
        payout_a2 = self._payout("node-a2", 2.5)
        batch = claims.create_batch()
        self.assertIsNotNone(batch)
        batch_id = batch["batch_id"]
        publish = {
            "network": "sepolia",
            "chain_id": 11155111,
            "contract": "0x" + "44" * 20,
            "tx_hash": "0x" + "a" * 64,
            "block_number": 123,
            "from": "0x" + "55" * 20,
        }
        self.assertEqual(claims.mark_publish_pending(batch_id, publish)["status"], "pending")
        self.assertEqual(claims.mark_published(batch_id, publish)["status"], "published")
        self.assertEqual(claims.mark_published(batch_id, publish)["status"], "published")
        with self.assertRaises(claims.PayoutClaimError):
            claims.mark_published(batch_id, {**publish, "tx_hash": "0x" + "b" * 64})

        verified_claim = {"tx_hash": "0x" + "c" * 64}
        claimed = claims.mark_claimed(batch_id, 0, verified_claim)
        self.assertTrue(claimed["claimed"])
        self.assertEqual(claimed["claimed_tx_hash"], verified_claim["tx_hash"])
        self.assertEqual(
            claims.mark_claimed(batch_id, 0, verified_claim)["claimed_tx_hash"],
            verified_claim["tx_hash"],
        )
        payout_rows = self.conn.execute(
            "SELECT id, reward_asset_type, tx_hash FROM node_payouts ORDER BY id"
        ).fetchall()
        self.assertEqual([row[0] for row in payout_rows], [payout_a1, payout_a2])
        self.assertTrue(all(row[1] == "onchain_hai" for row in payout_rows))
        self.assertTrue(all(row[2] == verified_claim["tx_hash"] for row in payout_rows))
        with self.assertRaises(claims.PayoutClaimError):
            claims.mark_claimed(batch_id, 0, {"tx_hash": "0x" + "d" * 64})


if __name__ == "__main__":
    unittest.main()
