"""Regression tests for tester onboarding + controlled HAI distribution support."""

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


VALID_WALLET = "0x1111111111111111111111111111111111111111"
OTHER_WALLET = "0x2222222222222222222222222222222222222222"
JOIN_TOKEN = "test-join-token"


class TesterDistributionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = app_module.app.test_client()

        self._orig_db_path = app_module.DB_PATH
        self._orig_db_conn = app_module.DB_CONN
        self._orig_server_join_token = app_module.SERVER_JOIN_TOKEN
        self._orig_nodes = copy.deepcopy(app_module.NODES)
        self._orig_tasks = copy.deepcopy(app_module.TASKS)
        self._orig_enabled = app_module.hai_funding.TESTER_DISTRIBUTION_ENABLED
        self._orig_allowlist = set(app_module.hai_funding.TESTER_DISTRIBUTION_ALLOWED_WALLETS)
        self._orig_default_hai = app_module.hai_funding.TESTER_DISTRIBUTION_DEFAULT_HAI
        self._orig_cooldown = app_module.hai_funding.TESTER_DISTRIBUTION_COOLDOWN_HOURS

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

        app_module.hai_funding.TESTER_DISTRIBUTION_ENABLED = True
        app_module.hai_funding.TESTER_DISTRIBUTION_ALLOWED_WALLETS = set()
        app_module.hai_funding.TESTER_DISTRIBUTION_DEFAULT_HAI = 100.0
        app_module.hai_funding.TESTER_DISTRIBUTION_COOLDOWN_HOURS = 24

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

        app_module.hai_funding.TESTER_DISTRIBUTION_ENABLED = self._orig_enabled
        app_module.hai_funding.TESTER_DISTRIBUTION_ALLOWED_WALLETS = set(self._orig_allowlist)
        app_module.hai_funding.TESTER_DISTRIBUTION_DEFAULT_HAI = self._orig_default_hai
        app_module.hai_funding.TESTER_DISTRIBUTION_COOLDOWN_HOURS = self._orig_cooldown

        self._tmpdir.cleanup()

    def test_tester_distribution_request_and_wallet_history(self) -> None:
        create_resp = self.client.post(
            "/credits/tester-distribution/request",
            json={
                "wallet": VALID_WALLET,
                "requested_hai": 150,
                "request_note": "Need test funding for onboarding",
            },
        )
        self.assertEqual(create_resp.status_code, 201)
        body = create_resp.get_json()
        self.assertEqual(body.get("status"), "pending")
        self.assertEqual(body["request"]["wallet"], VALID_WALLET)
        self.assertEqual(body["request"]["requested_hai"], 150.0)

        history_resp = self.client.get(
            f"/credits/tester-distribution/requests?wallet={VALID_WALLET}"
        )
        self.assertEqual(history_resp.status_code, 200)
        history = history_resp.get_json()
        self.assertTrue(history.get("tester_distribution", {}).get("enabled"))
        self.assertEqual(len(history.get("requests", [])), 1)
        self.assertEqual(history["requests"][0]["wallet"], VALID_WALLET)

    def test_tester_distribution_respects_wallet_allowlist(self) -> None:
        app_module.hai_funding.TESTER_DISTRIBUTION_ALLOWED_WALLETS = {OTHER_WALLET.lower()}
        resp = self.client.post(
            "/credits/tester-distribution/request",
            json={"wallet": VALID_WALLET, "requested_hai": 50},
        )
        self.assertEqual(resp.status_code, 403)
        self.assertEqual(resp.get_json().get("error"), "wallet_not_allowed")

    def test_admin_can_resolve_request_and_grant_credits(self) -> None:
        create_resp = self.client.post(
            "/credits/tester-distribution/request",
            json={"wallet": VALID_WALLET, "requested_hai": 100},
        )
        self.assertEqual(create_resp.status_code, 201)
        request_id = int(create_resp.get_json()["request"]["id"])

        unauthorized = self.client.post(
            f"/credits/tester-distribution/requests/{request_id}/resolve",
            json={"status": "completed", "credits_granted": 20},
        )
        self.assertEqual(unauthorized.status_code, 403)

        authorized = self.client.post(
            f"/credits/tester-distribution/requests/{request_id}/resolve",
            headers={"X-Join-Token": JOIN_TOKEN},
            json={
                "status": "completed",
                "credits_granted": 20,
                "admin_note": "Approved for private alpha",
            },
        )
        self.assertEqual(authorized.status_code, 200)
        body = authorized.get_json()
        self.assertEqual(body.get("status"), "completed")
        self.assertEqual(body["request"]["credits_granted"], 20.0)

        balance = app_module.credits.get_credit_balance(VALID_WALLET)
        self.assertAlmostEqual(balance, 20.0)

    def test_pending_request_is_not_duplicated(self) -> None:
        first = self.client.post(
            "/credits/tester-distribution/request",
            json={"wallet": VALID_WALLET, "requested_hai": 80},
        )
        self.assertEqual(first.status_code, 201)
        first_id = int(first.get_json()["request"]["id"])

        second = self.client.post(
            "/credits/tester-distribution/request",
            json={"wallet": VALID_WALLET, "requested_hai": 80},
        )
        self.assertEqual(second.status_code, 200)
        second_body = second.get_json()
        self.assertEqual(second_body.get("status"), "pending_exists")
        self.assertEqual(int(second_body.get("request_id")), first_id)

    def test_credit_grant_not_double_counted_across_status_updates(self) -> None:
        create_resp = self.client.post(
            "/credits/tester-distribution/request",
            json={"wallet": VALID_WALLET, "requested_hai": 100},
        )
        self.assertEqual(create_resp.status_code, 201)
        request_id = int(create_resp.get_json()["request"]["id"])

        approved = self.client.post(
            f"/credits/tester-distribution/requests/{request_id}/resolve",
            headers={"X-Join-Token": JOIN_TOKEN},
            json={"status": "approved", "credits_granted": 20},
        )
        self.assertEqual(approved.status_code, 200)

        completed = self.client.post(
            f"/credits/tester-distribution/requests/{request_id}/resolve",
            headers={"X-Join-Token": JOIN_TOKEN},
            json={"status": "completed", "credits_granted": 20},
        )
        self.assertEqual(completed.status_code, 200)
        self.assertEqual(completed.get_json()["request"]["credits_granted"], 20.0)

        balance = app_module.credits.get_credit_balance(VALID_WALLET)
        self.assertAlmostEqual(balance, 20.0)


if __name__ == "__main__":
    unittest.main()
