"""Regression tests for Node Operator Market-Worker layer v1."""

from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "server"))

import app as app_module


JOIN_TOKEN = "operator-layer-token"
NODE_ID = "node-operator-1"
WALLET = "0x1111111111111111111111111111111111111111"


class OperatorMarketWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = app_module.app.test_client()
        self._orig_db_path = app_module.DB_PATH
        self._orig_db_conn = app_module.DB_CONN
        self._orig_server_join_token = app_module.SERVER_JOIN_TOKEN
        self._orig_nodes = copy.deepcopy(app_module.NODES)
        self._orig_tasks = copy.deepcopy(app_module.TASKS)

        self._tmpdir = tempfile.TemporaryDirectory()
        app_module.DB_PATH = Path(self._tmpdir.name) / "ledger.db"
        app_module.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        app_module.DB_CONN = None

        app_module.init_db()
        app_module.stripe_payments.init_stripe_tables(app_module.get_db())
        app_module.settlement.init_settlement_tables(app_module.get_db())
        app_module.hai_funding.init_hai_funding_tables(app_module.get_db())
        app_module.blockchain.init_blockchain_tables(app_module.get_db())
        app_module.validators.init_validator_tables(app_module.get_db())
        app_module.workflows.init_workflow_tables(app_module.get_db())
        app_module.gallery.init_gallery_tables(app_module.get_db())

        app_module.SERVER_JOIN_TOKEN = JOIN_TOKEN
        app_module.NODES.clear()
        app_module.TASKS.clear()

    def tearDown(self) -> None:
        if app_module.DB_CONN is not None:
            try:
                app_module.DB_CONN.close()
            except Exception:
                pass

        app_module.DB_PATH = self._orig_db_path
        app_module.DB_CONN = self._orig_db_conn
        app_module.SERVER_JOIN_TOKEN = self._orig_server_join_token
        app_module.NODES.clear()
        app_module.NODES.update(self._orig_nodes)
        app_module.TASKS.clear()
        app_module.TASKS.update(self._orig_tasks)
        self._tmpdir.cleanup()

    def _register_node(self) -> None:
        response = self.client.post(
            "/register",
            headers={"X-Join-Token": JOIN_TOKEN},
            json={
                "node_id": NODE_ID,
                "role": "creator",
                "node_name": "Operator Alpha",
                "operator_display_name": "Alpha Operator",
                "supports": ["image", "face_swap", "video"],
                "models": ["perfectdeliberate_v60", "ltx2"],
                "pipelines": ["sdxl", "ltx2"],
                "gpu_stats": {"gpu_name": "RTX 4090", "memory_total_mb": 24576, "utilization": 23},
                "version": "1.2.3",
            },
        )
        self.assertEqual(response.status_code, 200)

    def test_registry_identity_and_capabilities_visible(self) -> None:
        self._register_node()
        link = self.client.post(
            "/link-wallet",
            headers={"X-Join-Token": JOIN_TOKEN},
            json={
                "node_id": NODE_ID,
                "wallet": WALLET,
                "node_name": "Operator Alpha",
                "operator_display_name": "Alpha Operator",
            },
        )
        self.assertEqual(link.status_code, 200)

        workers = self.client.get("/operators/workers?status=online")
        self.assertEqual(workers.status_code, 200)
        payload = workers.get_json()
        self.assertIn("workers", payload)
        self.assertEqual(len(payload["workers"]), 1)
        worker = payload["workers"][0]

        self.assertEqual(worker["node_id"], NODE_ID)
        self.assertEqual(worker["operator"]["wallet"], WALLET)
        self.assertEqual(worker["operator"]["display_name"], "Alpha Operator")
        self.assertIn("IMAGE_GEN", worker.get("supported_job_types", []))
        self.assertIn("FACE_SWAP", worker.get("supported_job_types", []))
        self.assertIn("VIDEO_GEN", worker.get("supported_job_types", []))
        self.assertEqual(worker["status"], "online")

    def test_operator_detail_includes_attempt_and_payout_metrics(self) -> None:
        self._register_node()

        # Completed/successful job => payout record
        app_module.settlement.create_job_ticket(
            job_id="job-success",
            wallet=WALLET,
            job_type="IMAGE_GEN",
            model="perfectdeliberate_v60",
            estimated_cost=0.0,
            prompt="test",
            input_metadata={"source": "test"},
        )
        app_module.settlement.record_claim("job-success", NODE_ID)
        app_module.settlement.complete_attempt("job-success", NODE_ID, "success")
        app_module.settlement.settle_job(
            job_id="job-success",
            node_id=NODE_ID,
            execution_status=app_module.settlement.STATUS_COMPLETED,
            quality_status=app_module.settlement.QUALITY_VALID,
            reward_amount=0.75,
            deposit_fn=app_module.credits.deposit_credits,
        )

        # Failed job => failed attempt metric
        app_module.settlement.create_job_ticket(
            job_id="job-failed",
            wallet=WALLET,
            job_type="IMAGE_GEN",
            model="perfectdeliberate_v60",
            estimated_cost=0.0,
            prompt="test",
            input_metadata={"source": "test"},
        )
        app_module.settlement.record_claim("job-failed", NODE_ID)
        app_module.settlement.complete_attempt("job-failed", NODE_ID, "failed", error_message="oom")
        app_module.settlement.settle_job(
            job_id="job-failed",
            node_id=NODE_ID,
            execution_status=app_module.settlement.STATUS_TECHNICAL_FAILED,
            quality_status=app_module.settlement.QUALITY_UNCHECKED,
            reward_amount=0.0,
            deposit_fn=app_module.credits.deposit_credits,
        )

        detail_resp = self.client.get(f"/operators/workers/{NODE_ID}")
        self.assertEqual(detail_resp.status_code, 200)
        detail = detail_resp.get_json()

        performance = detail.get("performance", {})
        payouts = detail.get("payouts", {})
        self.assertGreaterEqual(int(performance.get("attempts_total", 0)), 2)
        self.assertGreaterEqual(int(performance.get("failed_attempts", 0)), 1)
        self.assertGreater(float(payouts.get("total", 0.0)), 0.0)
        self.assertGreaterEqual(len(detail.get("recent_attempts", [])), 2)
        self.assertGreaterEqual(len(detail.get("recent_payouts", [])), 1)

        nodes_resp = self.client.get("/nodes")
        self.assertEqual(nodes_resp.status_code, 200)
        nodes_payload = nodes_resp.get_json()
        self.assertEqual(len(nodes_payload.get("nodes", [])), 1)
        node_row = nodes_payload["nodes"][0]
        self.assertIn("performance", node_row)
        self.assertIn("payouts", node_row)
        self.assertIn("trust", node_row)


if __name__ == "__main__":
    unittest.main()
