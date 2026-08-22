"""Tests for prompt-free, content-addressed Astra artifact receipts."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "havnai_server_astra_receipts", ROOT / "server" / "astra_receipts.py"
)
assert SPEC is not None and SPEC.loader is not None
astra_receipts = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = astra_receipts
SPEC.loader.exec_module(astra_receipts)


WALLET = "0x" + "a" * 40


class AstraReceiptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.outputs = Path(self.temp.name) / "outputs"
        self.outputs.mkdir()
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(
            """
            CREATE TABLE jobs (
                id TEXT PRIMARY KEY, status TEXT, task_type TEXT, model TEXT,
                node_id TEXT, timestamp REAL, completed_at REAL, data TEXT
            );
            CREATE TABLE astra_reward_images (
                run_id TEXT PRIMARY KEY, job_id TEXT, video_job_id TEXT,
                wallet TEXT, pilot_id TEXT, outfit_id TEXT, map_id TEXT,
                grade TEXT, created_at REAL
            );
            CREATE TABLE artifacts (
                id TEXT PRIMARY KEY, job_id TEXT, kind TEXT, content_type TEXT,
                path TEXT, size_bytes INTEGER, sha256 TEXT, created_at REAL
            );
            CREATE TABLE job_settlement (
                job_id TEXT PRIMARY KEY, input_metadata TEXT,
                assigned_node_id TEXT, attempt_count INTEGER,
                execution_status TEXT, quality_status TEXT,
                settlement_outcome TEXT, spent_amount REAL, updated_at REAL
            );
            CREATE TABLE node_payouts (
                id INTEGER PRIMARY KEY, node_id TEXT, job_id TEXT,
                reward_amount REAL, reward_asset_type TEXT, tx_hash TEXT,
                created_at REAL
            );
            CREATE TABLE rewards (
                task_id TEXT PRIMARY KEY, reward_hai REAL, timestamp REAL
            );
            """
        )
        astra_receipts.init_receipt_tables(self.conn)

    def tearDown(self) -> None:
        self.conn.close()
        self.temp.cleanup()

    def _insert_job(
        self,
        job_id: str = "job-astra-1",
        *,
        status: str = "succeeded",
        indexed: bool = True,
    ) -> tuple[Path, str]:
        content = f"final artifact bytes for {job_id}".encode()
        path = self.outputs / f"{job_id}.png"
        path.write_bytes(content)
        digest = hashlib.sha256(content).hexdigest()
        private_data = {
            "prompt": "never expose this prompt",
            "negative_prompt": "also private",
            "routing_source": "player_affinity",
            "preferred_node_id": "creator-one",
        }
        self.conn.execute(
            "INSERT INTO jobs VALUES (?, ?, 'IMAGE_GEN', 'astra-model', 'creator-one', 100, 120, ?)",
            (job_id, status, json.dumps(private_data)),
        )
        self.conn.execute(
            "INSERT INTO astra_reward_images VALUES (?, ?, NULL, ?, 'pilot_nova', 'outfit_17', "
            "'nebula-runway', 'S', 99)",
            (f"run-{job_id}", job_id, WALLET),
        )
        if indexed:
            self.conn.execute(
                "INSERT INTO artifacts VALUES (?, ?, 'image', 'image/png', ?, ?, ?, 119)",
                (f"artifact-{job_id}", job_id, str(path), len(content), digest),
            )
        metadata = {
            "model_key": "astra-model",
            "model_name": "Astra Model",
            "pipeline": "sdxl",
            "tier": "A",
            "prompt": "private settlement prompt",
        }
        self.conn.execute(
            "INSERT INTO job_settlement VALUES (?, ?, 'creator-one', 1, 'settled', "
            "'valid', 'spent', 2.5, 121)",
            (job_id, json.dumps(metadata)),
        )
        self.conn.execute(
            "INSERT INTO node_payouts VALUES (NULL, 'creator-one', ?, 7.25, "
            "'simulated_hai', NULL, 121)",
            (job_id,),
        )
        self.conn.execute(
            "INSERT INTO rewards VALUES (?, 7.25, 121)", (job_id,)
        )
        self.conn.commit()
        return path, digest

    def test_issues_content_addressed_prompt_free_receipt(self) -> None:
        _, artifact_digest = self._insert_job()

        response = astra_receipts.get_or_issue_receipt(
            self.conn, "job-astra-1", self.outputs
        )
        receipt = response["receipt"]

        self.assertEqual(receipt["artifact"]["sha256"], artifact_digest)
        self.assertEqual(receipt["artifact"]["digest_source"], "node_upload")
        self.assertEqual(receipt["execution"]["creator_node_id"], "creator-one")
        self.assertTrue(receipt["routing"]["preference_honored"])
        self.assertEqual(receipt["settlement"]["node_reward"], 7.25)
        self.assertEqual(receipt["settlement"]["reward_asset_type"], "simulated_hai")
        self.assertEqual(
            response["receipt_sha256"],
            "sha256:" + hashlib.sha256(response["canonical_json"].encode()).hexdigest(),
        )
        self.assertEqual(json.loads(response["canonical_json"]), receipt)
        self.assertNotIn(WALLET, response["canonical_json"])
        self.assertNotIn("never expose", response["canonical_json"])
        self.assertNotIn("private settlement prompt", response["canonical_json"])

    def test_first_final_receipt_remains_stable(self) -> None:
        self._insert_job()
        first = astra_receipts.get_or_issue_receipt(
            self.conn, "job-astra-1", self.outputs
        )
        self.conn.execute(
            "UPDATE node_payouts SET reward_amount = 999 WHERE job_id = 'job-astra-1'"
        )
        self.conn.execute(
            "UPDATE jobs SET node_id = 'renamed-node' WHERE id = 'job-astra-1'"
        )
        self.conn.commit()

        second = astra_receipts.get_or_issue_receipt(
            self.conn, "job-astra-1", self.outputs
        )
        self.assertEqual(second["canonical_json"], first["canonical_json"])
        self.assertEqual(second["receipt_sha256"], first["receipt_sha256"])

    def test_legacy_output_is_hashed_by_coordinator(self) -> None:
        _, expected = self._insert_job("job-legacy", indexed=False)
        response = astra_receipts.get_or_issue_receipt(
            self.conn, "job-legacy", self.outputs
        )
        self.assertEqual(response["receipt"]["artifact"]["sha256"], expected)
        self.assertEqual(
            response["receipt"]["artifact"]["digest_source"], "coordinator_scan"
        )

    def test_pending_and_non_astra_jobs_do_not_issue_receipts(self) -> None:
        self._insert_job("job-pending", status="running")
        with self.assertRaises(astra_receipts.ReceiptUnavailable) as pending:
            astra_receipts.get_or_issue_receipt(
                self.conn, "job-pending", self.outputs
            )
        self.assertEqual((pending.exception.code, pending.exception.status), ("artifact_not_ready", 409))

        self.conn.execute(
            "INSERT INTO jobs VALUES ('job-other', 'succeeded', 'IMAGE_GEN', 'm', 'n', 1, 2, '{}')"
        )
        self.conn.commit()
        with self.assertRaises(astra_receipts.ReceiptUnavailable) as missing:
            astra_receipts.get_or_issue_receipt(self.conn, "job-other", self.outputs)
        self.assertEqual(
            (missing.exception.code, missing.exception.status),
            ("astra_artifact_not_found", 404),
        )

    def test_completed_output_waits_for_final_settlement(self) -> None:
        self._insert_job("job-unsettled")
        self.conn.execute(
            "UPDATE job_settlement SET settlement_outcome = 'pending' WHERE job_id = 'job-unsettled'"
        )
        self.conn.commit()
        with self.assertRaises(astra_receipts.ReceiptUnavailable) as pending:
            astra_receipts.get_or_issue_receipt(
                self.conn, "job-unsettled", self.outputs
            )
        self.assertEqual(
            (pending.exception.code, pending.exception.status),
            ("settlement_not_ready", 409),
        )

    def test_indexed_path_cannot_escape_outputs_directory(self) -> None:
        path, expected = self._insert_job("job-safe", indexed=False)
        outside = Path(self.temp.name) / "outside.png"
        outside.write_bytes(b"outside bytes")
        self.conn.execute(
            "INSERT INTO artifacts VALUES ('artifact-outside', 'job-safe', 'image', "
            "'image/png', ?, 13, ?, 119)",
            (str(outside), hashlib.sha256(b"outside bytes").hexdigest()),
        )
        self.conn.commit()

        response = astra_receipts.get_or_issue_receipt(
            self.conn, "job-safe", self.outputs
        )
        self.assertEqual(response["receipt"]["artifact"]["sha256"], expected)
        self.assertEqual(response["receipt"]["artifact"]["id"], "legacy:job-safe")
        self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
