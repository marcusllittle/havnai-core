"""Failure/retry and concurrency regressions; no live payments or chain calls."""
import copy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sqlite3
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "server"))
import app as api
import network_status

SELLER = "0x" + "1" * 40
BUYER = "0x" + "2" * 40
NEXT_BUYER = "0x" + "3" * 40
TX = "0x" + "a" * 64


class BackendReliabilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.old_path, self.old_conn = api.DB_PATH, api.DB_CONN
        self.nodes = copy.deepcopy(api.NODES)
        api.NODES.clear()
        api.DB_PATH, api.DB_CONN = Path(self.temp.name) / "test.db", None
        api.init_db()
        for init in [api.gallery.init_gallery_tables, api.stripe_payments.init_stripe_tables,
                     api.hai_funding.init_hai_funding_tables, api.settlement.init_settlement_tables,
                     api.astra_receipts.init_receipt_tables, api.merkle_batches.init_merkle_tables]:
            init(api.get_db())
        self.client = api.app.test_client()

    def tearDown(self):
        api.DB_CONN.close()
        api.DB_PATH, api.DB_CONN = self.old_path, self.old_conn
        api.NODES.clear()
        api.NODES.update(self.nodes)
        self.temp.cleanup()

    def fail_credit_insert(self):
        api.get_db().execute("CREATE TRIGGER fail_credit BEFORE INSERT ON credits BEGIN SELECT RAISE(ABORT, 'test credit failure'); END")
        api.get_db().commit()

    def test_hai_credit_failure_rolls_back_completion_and_retry_grants_once(self):
        self.fail_credit_insert()
        with patch.object(api.hai_funding, "verify_hai_transfer", return_value={"verified": True}):
            with self.assertRaises(sqlite3.IntegrityError):
                api.hai_funding.fund_credits_with_hai(BUYER, 10, TX)
            self.assertEqual(api.get_db().execute("SELECT status FROM hai_fundings").fetchone()[0], "pending")
            api.get_db().execute("DROP TRIGGER fail_credit")
            api.get_db().commit()
            first = api.hai_funding.fund_credits_with_hai(BUYER, 10, TX)
            again = api.hai_funding.fund_credits_with_hai(BUYER, 10, TX)
        self.assertEqual(first["status"], "completed")
        self.assertEqual(again["status"], "already_processed")
        self.assertEqual(api.credits.get_credit_balance(BUYER), 10)

    def test_failed_verifier_cannot_reopen_a_concurrently_completed_payment(self):
        def concurrent_completion(*args):
            db = api.get_db()
            with db:
                db.execute("UPDATE hai_fundings SET status='completed', credits_granted=10 WHERE tx_hash=?", (TX,))
                api.credits.deposit_in_transaction(db, BUYER, 10)
            return {"verified": False, "error": "temporary RPC failure"}
        with patch.object(api.hai_funding, "verify_hai_transfer", side_effect=concurrent_completion):
            result = api.hai_funding.fund_credits_with_hai(BUYER, 10, TX)
        self.assertEqual(result["status"], "already_processed")
        self.assertEqual(api.get_db().execute("SELECT status FROM hai_fundings").fetchone()[0], "completed")
        self.assertEqual(api.credits.get_credit_balance(BUYER), 10)

    def test_stripe_credit_failure_is_retryable_and_uses_stored_checkout(self):
        db = api.get_db()
        db.execute("""INSERT INTO stripe_payments (stripe_session_id,wallet,package_id,credits_amount,price_cents,created_at)
            VALUES ('cs_test',?,'starter',50,500,?)""", (BUYER, time.time()))
        db.commit()
        event = {"type": "checkout.session.completed", "data": {"object": {
            "id": "cs_test", "payment_status": "paid", "metadata": {"wallet": SELLER, "credits": "999", "package_id": "pro"}
        }}}
        stripe = SimpleNamespace(Webhook=SimpleNamespace(construct_event=lambda *args: event))
        self.fail_credit_insert()
        with patch.dict(sys.modules, {"stripe": stripe}):
            with self.assertRaises(sqlite3.IntegrityError):
                api.stripe_payments.handle_webhook_event(b"", "test")
            self.assertEqual(db.execute("SELECT status FROM stripe_payments").fetchone()[0], "pending")
            db.execute("DROP TRIGGER fail_credit")
            db.commit()
            api.stripe_payments.handle_webhook_event(b"", "test")
            api.stripe_payments.handle_webhook_event(b"", "test")
        self.assertEqual(api.credits.get_credit_balance(BUYER), 50)
        self.assertEqual(api.credits.get_credit_balance(SELLER), 0)

    def test_sale_failure_rolls_back_both_balances_and_ownership(self):
        listing = api.gallery.create_listing("asset", SELLER, "Test", 3)
        api.credits.deposit_credits(BUYER, 10)
        db = api.get_db()
        db.execute("CREATE TRIGGER fail_sale BEFORE INSERT ON gallery_sales BEGIN SELECT RAISE(ABORT, 'sale failed'); END")
        db.commit()
        with self.assertRaises(sqlite3.IntegrityError):
            api.gallery.purchase_listing(listing["id"], BUYER, settle_credits=True, expected_price=3)
        self.assertEqual(api.credits.get_credit_balance(BUYER), 10)
        self.assertEqual(api.credits.get_credit_balance(SELLER), 0)
        self.assertEqual(api.gallery.get_asset_owner("asset"), SELLER)
        self.assertEqual(db.execute("SELECT COUNT(*) FROM gallery_sales").fetchone()[0], 0)

    def test_competing_buyers_cannot_buy_the_same_listing(self):
        listing = api.gallery.create_listing("asset", SELLER, "Test", 3)
        for wallet in [BUYER, NEXT_BUYER]:
            api.credits.deposit_credits(wallet, 10)
        def buy(wallet):
            with api.app.app_context():
                return api.gallery.purchase_listing(listing["id"], wallet, settle_credits=True, expected_price=3)
        with ThreadPoolExecutor(2) as pool:
            results = list(pool.map(buy, [BUYER, NEXT_BUYER]))
        self.assertEqual(sum(bool(result["ok"]) for result in results), 1)
        self.assertEqual(api.credits.get_credit_balance(SELLER), 3)
        self.assertEqual(sum(api.credits.get_credit_balance(w) for w in [BUYER, NEXT_BUYER]), 17)
        self.assertEqual(api.get_db().execute("SELECT COUNT(*) FROM gallery_sales").fetchone()[0], 1)

    def test_former_owner_cannot_relist_or_see_resold_asset_in_owned_collection(self):
        first = api.gallery.create_listing("asset", SELLER, "Test", 3)
        api.gallery.purchase_listing(first["id"], BUYER)
        second = api.gallery.relist_owned_asset("asset", BUYER, "Resale", 4)["listing"]
        api.gallery.purchase_listing(second["id"], NEXT_BUYER)
        self.assertEqual(api.gallery.get_owned_assets(BUYER), [])
        self.assertEqual(api.gallery.relist_owned_asset("asset", BUYER, "Invalid", 5)["error"], "not_owner")
        self.assertEqual(len(api.gallery.get_owned_assets(NEXT_BUYER)), 1)

    def test_account_ownership_overrides_every_legacy_gallery_surface(self):
        first = api.gallery.create_listing("migrated", SELLER, "Private", 3, prompt="private prompt")
        api.gallery.purchase_listing(first["id"], BUYER)
        active = api.gallery.relist_owned_asset("migrated", BUYER, "Resale", 4)["listing"]
        public = api.gallery.create_listing("public", SELLER, "Public", 2)
        db = api.get_db()
        with db:
            db.execute("INSERT INTO accounts(id,created_at) VALUES ('acct_owner',?)", (time.time(),))
            db.execute("""INSERT INTO jobs
                (id,wallet,model,task_type,weight,status,timestamp,owner_account_id,creator_account_id)
                VALUES ('migrated',?,'model','IMAGE_GEN',1,'completed',?,'acct_owner','acct_owner')""",
                (SELLER, time.time()))
        for listing in (first, active):
            self.assertIsNone(api.gallery.get_listing(listing["id"]))
            self.assertEqual(self.client.get(f"/gallery/listings/{listing['id']}").status_code, 404)
            self.assertEqual(self.client.get(
                f"/gallery/listings/{listing['id']}/download?wallet={BUYER}").status_code, 404)
        result = api.gallery.browse_gallery(limit=1)
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["listings"][0]["id"], public["id"])
        self.assertEqual(api.gallery.get_owned_assets(BUYER), [])
        self.assertEqual(api.gallery.buyer_purchases(BUYER), [])
        self.assertEqual(api.gallery.seller_listings(BUYER, include_sold=True), [])
        self.assertEqual(api.gallery.get_ownership_history("migrated"), [])
        self.assertIsNone(api.gallery.get_asset_owner("migrated"))
        self.assertFalse(api.gallery.delist(active["id"], BUYER))
        self.assertEqual(api.gallery.relist_owned_asset("migrated", BUYER, "Steal", 1)["error"], "not_owner")
        with self.assertRaisesRegex(ValueError, "account_owned_job"):
            api.gallery.create_listing("migrated", SELLER, "Reclaim", 1)
        api.credits.deposit_credits(NEXT_BUYER, 10)
        self.assertEqual(api.gallery.purchase_listing(active["id"], NEXT_BUYER,
                         settle_credits=True)["error"], "listing_not_found")
        self.assertEqual(api.credits.get_credit_balance(NEXT_BUYER), 10)
        self.assertEqual(db.execute("SELECT owner_account_id FROM jobs WHERE id='migrated'").fetchone()[0], "acct_owner")
        self.assertEqual(db.execute("SELECT COUNT(*) FROM gallery_sales").fetchone()[0], 1)

    def test_listing_insert_rechecks_account_ownership_after_initial_check(self):
        db = api.get_db()
        with db:
            db.execute("INSERT INTO accounts(id,created_at) VALUES ('acct_owner',?)", (time.time(),))
            db.execute("""INSERT INTO jobs (id,wallet,model,task_type,weight,status,timestamp)
                VALUES ('racing',?,'model','IMAGE_GEN',1,'completed',?)""", (SELLER, time.time()))
        original = api.gallery.create_listing("racing", SELLER, "Original", 3)
        api.gallery.delist(original["id"], SELLER)

        def migrated_after_read(conn, job_id):
            with conn:
                conn.execute("UPDATE jobs SET owner_account_id='acct_owner' WHERE id=?", (job_id,))
            return False

        for operation in (api.gallery.create_listing, api.gallery.relist_owned_asset):
            with self.subTest(operation=operation.__name__):
                with db:
                    db.execute("UPDATE jobs SET owner_account_id=NULL WHERE id='racing'")
                with patch.object(api.gallery, "_account_owned_job", side_effect=migrated_after_read):
                    with self.assertRaisesRegex(ValueError, "account_owned_job"):
                        operation("racing", SELLER, "Racing listing", 4)
                self.assertEqual(db.execute("SELECT COUNT(*) FROM gallery_listings").fetchone()[0], 1)
                self.assertEqual(db.execute("SELECT COUNT(*) FROM gallery_ownership_log").fetchone()[0], 1)

    def test_original_creator_cannot_reclaim_a_sold_job_via_create_listing(self):
        db = api.get_db()
        with db:
            db.execute("""INSERT INTO jobs (id,wallet,model,task_type,weight,status,timestamp)
                VALUES ('sold-job',?,'model','IMAGE_GEN',1,'completed',?)""", (SELLER, time.time()))
        listing = api.gallery.create_listing("sold-job", SELLER, "Original", 3)
        api.gallery.purchase_listing(listing["id"], BUYER)
        with patch.object(api, "_verify_wallet_signature", return_value=(True, None)):
            response = self.client.post("/gallery/listings", json={
                "job_id": "sold-job", "wallet": SELLER, "price_credits": 2,
                "nonce": "verified-test-nonce", "signature": "verified-test-signature"})
        self.assertEqual(response.status_code, 403)
        self.assertEqual(api.gallery.get_asset_owner("sold-job"), BUYER)

    def test_network_routes_return_real_empty_state_without_mutation(self):
        for path in ["/v1/network/summary", "/v1/network/control-plane"]:
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
            self.assertEqual(response.headers["Cache-Control"], "no-store")
            self.assertEqual(response.get_json()["queue"]["queued"], 0)
        data = self.client.get("/v1/network/control-plane").get_json()
        self.assertIsNone(data["latency_24h"]["run_p95_seconds"])
        self.assertEqual(data["health"]["status"], "degraded")

    def test_network_counts_legacy_states_expired_claims_and_measured_latency(self):
        db = api.get_db()
        for job_id, state, queued, assigned, completed, lease in [
            ("queued", "pending", 900, None, None, None),
            ("running", "running", 800, 810, None, 990),
            ("complete", "success", 850, 860, 890, None),
        ]:
            db.execute("""INSERT INTO jobs (id,wallet,model,task_type,weight,status,timestamp,assigned_at,completed_at,lease_expires_at,node_id)
                VALUES (?,?,'model','IMAGE_GEN',1,?,?,?,?,?,'creator')""", (job_id, BUYER, state, queued, assigned, completed, lease))
        db.commit()
        nodes = [{"node_id": "creator", "role": "creator", "online": True, "wallet": BUYER,
                  "supported_job_types": ["IMAGE_GEN", "IMAGE_GEN"], "gpu": {"memory_total": 24000}, "utilization": 50},
                 {"node_id": "offline", "role": "creator", "online": False, "wallet": SELLER,
                  "supported_job_types": ["MUSIC_GEN"], "gpu": {"memory_total_mb": 24000}}]
        summary, control = network_status.snapshot(db, nodes, version="test", lease_seconds=90,
                                                   receipt_count=0, receipt_batches=[], now=1000)
        self.assertEqual(summary["capacity"]["by_job_type"], {"IMAGE_GEN": 1})
        self.assertEqual(summary["capacity"]["total_vram_mb"], 24000)
        self.assertEqual(summary["queue"], {"queued": 1, "running": 1, "completed": 1, "failed": 0})
        self.assertEqual(summary["recovery"]["expired_claims"], 1)
        self.assertEqual(control["nodes"]["busy"], 1)
        self.assertEqual(control["queue"]["oldest_wait_seconds"], 100)
        self.assertEqual(control["latency_24h"]["run_p95_seconds"], 30)
        self.assertEqual(db.execute("SELECT status FROM jobs WHERE id='running'").fetchone()[0], "running")
