"""Tests for database-backed, atomic network job dispatch."""

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from server import job_helpers


class AtomicJobClaimTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_dispatch_decision = job_helpers.get_dispatch_decision
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmpdir.name) / "jobs.db"
        self.connection = sqlite3.connect(self.db_path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute(
            """
            CREATE TABLE jobs (
                id TEXT PRIMARY KEY,
                wallet TEXT NOT NULL,
                model TEXT NOT NULL,
                data TEXT,
                task_type TEXT NOT NULL,
                weight REAL NOT NULL,
                status TEXT NOT NULL,
                node_id TEXT,
                timestamp REAL NOT NULL,
                assigned_at REAL,
                completed_at REAL,
                invite_code TEXT,
                lease_renewed_at REAL,
                lease_expires_at REAL,
                retry_count INTEGER NOT NULL DEFAULT 0,
                last_failure_reason TEXT,
                preferred_node_id TEXT,
                dispatch_score REAL,
                dispatch_reason TEXT
            )
            """
        )
        self.connection.commit()
        job_helpers.get_db = lambda: self.connection
        job_helpers.get_model_config = lambda name: (
            {"name": name, "pipeline": "sdxl"} if name == "network-model" else None
        )
        job_helpers.NODES = {
            "node-a": {
                "role": "creator",
                "supports": ["image"],
                "models": ["network-model"],
                "pipelines": ["sdxl"],
            },
            "node-b": {
                "role": "creator",
                "supports": ["image"],
                "models": ["network-model"],
                "pipelines": ["sdxl"],
            },
        }
        job_helpers.get_dispatch_decision = None

    def tearDown(self) -> None:
        job_helpers.get_dispatch_decision = self.original_dispatch_decision
        self.connection.close()
        self.tmpdir.cleanup()

    def _enqueue(self, job_id: str = "job-atomic") -> None:
        self.connection.execute(
            """
            INSERT INTO jobs
                (id, wallet, model, data, task_type, weight, status, timestamp)
            VALUES (?, ?, ?, ?, 'IMAGE_GEN', 1.0, 'queued', 1.0)
            """,
            (job_id, "0x1111111111111111111111111111111111111111", "network-model", json.dumps({"prompt": "test"})),
        )
        self.connection.commit()

    def test_claim_assigns_job_and_prevents_second_claim(self) -> None:
        self._enqueue()

        first = job_helpers.claim_next_job_for_node("node-a")
        second = job_helpers.claim_next_job_for_node("node-b")

        self.assertIsNotNone(first)
        self.assertEqual(first["node_id"], "node-a")
        self.assertEqual(first["status"], "running")
        self.assertIsNone(second)
        stored = self.connection.execute(
            "SELECT status, node_id, assigned_at FROM jobs WHERE id='job-atomic'"
        ).fetchone()
        self.assertEqual(stored["status"], "running")
        self.assertEqual(stored["node_id"], "node-a")
        self.assertIsNotNone(stored["assigned_at"])

    def test_heartbeat_renews_only_the_owning_nodes_claim(self) -> None:
        self._enqueue()
        claimed = job_helpers.claim_next_job_for_node("node-a")
        original_expiry = claimed["lease_expires_at"]

        wrong_owner = job_helpers.renew_job_leases("node-b", 3600, "job-atomic")
        renewed = job_helpers.renew_job_leases("node-a", 3600, "job-atomic")

        self.assertEqual(wrong_owner, 0)
        self.assertEqual(renewed, 1)
        stored = self.connection.execute(
            "SELECT lease_expires_at FROM jobs WHERE id='job-atomic'"
        ).fetchone()
        self.assertGreater(stored["lease_expires_at"], original_expiry)

    def test_expired_claim_requeues_then_exhausts_retry_budget(self) -> None:
        self._enqueue()
        job_helpers.claim_next_job_for_node("node-a")
        self.connection.execute(
            "UPDATE jobs SET lease_expires_at=10 WHERE id='job-atomic'"
        )
        self.connection.commit()

        first_recovery = job_helpers.recover_expired_leases(now=11, max_retries=1)
        self.assertEqual(first_recovery[0]["status"], "queued")
        self.assertEqual(first_recovery[0]["retry_count"], 1)

        job_helpers.claim_next_job_for_node("node-b")
        self.connection.execute(
            "UPDATE jobs SET lease_expires_at=20 WHERE id='job-atomic'"
        )
        self.connection.commit()
        final_recovery = job_helpers.recover_expired_leases(now=21, max_retries=1)

        self.assertEqual(final_recovery[0]["status"], "failed")
        stored = self.connection.execute(
            "SELECT status, retry_count, completed_at, last_failure_reason FROM jobs WHERE id='job-atomic'"
        ).fetchone()
        self.assertEqual(stored["status"], "failed")
        self.assertEqual(stored["retry_count"], 2)
        self.assertEqual(stored["completed_at"], 21)
        self.assertEqual(stored["last_failure_reason"], "lease_expired")

    def test_incompatible_node_leaves_job_queued(self) -> None:
        self._enqueue()
        job_helpers.NODES["node-a"]["pipelines"] = ["sd15"]

        claimed = job_helpers.claim_next_job_for_node("node-a")

        self.assertIsNone(claimed)
        stored = self.connection.execute(
            "SELECT status, node_id FROM jobs WHERE id='job-atomic'"
        ).fetchone()
        self.assertEqual(stored["status"], "queued")
        self.assertIsNone(stored["node_id"])

    def test_preferred_routing_blocks_lower_score_until_fallback(self) -> None:
        self._enqueue()
        allow_fallback = {"value": False}

        def decision(node_id: str, _job: dict) -> dict:
            preferred = node_id == "node-a"
            return {
                "allowed": preferred or allow_fallback["value"],
                "preferred_node_id": "node-a",
                "score": 90 if preferred else 60,
                "reason": "preferred_score" if preferred else (
                    "fallback_after_grace" if allow_fallback["value"] else "waiting_for_preferred_node"
                ),
            }

        job_helpers.get_dispatch_decision = decision
        blocked = job_helpers.claim_next_job_for_node("node-b")
        self.assertIsNone(blocked)

        allow_fallback["value"] = True
        claimed = job_helpers.claim_next_job_for_node("node-b")
        self.assertEqual(claimed["preferred_node_id"], "node-a")
        self.assertEqual(claimed["dispatch_score"], 60)
        self.assertEqual(claimed["dispatch_reason"], "fallback_after_grace")


if __name__ == "__main__":
    unittest.main()
