"""Regression tests for gallery marketplace API behavior."""

from __future__ import annotations

import copy
import sqlite3
import sys
import tempfile
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "server"))

import app as app_module


SELLER_WALLET = "0x1111111111111111111111111111111111111111"
BUYER_WALLET = "0x2222222222222222222222222222222222222222"
OTHER_BUYER_WALLET = "0x3333333333333333333333333333333333333333"


class GalleryApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = app_module.app.test_client()
        self.tmpdir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.tmpdir.name)
        self.outputs_dir = self.base_path / "static" / "outputs"
        (self.outputs_dir / "videos").mkdir(parents=True, exist_ok=True)
        self.db_path = self.base_path / "ledger.db"

        self._db_path_backup = app_module.DB_PATH
        self._outputs_backup = app_module.OUTPUTS_DIR
        self._rate_limit_backup = copy.deepcopy(app_module.RATE_LIMIT_BUCKETS)

        if app_module.DB_CONN is not None:
            app_module.DB_CONN.close()
        app_module.DB_CONN = None
        app_module.DB_PATH = self.db_path
        app_module.OUTPUTS_DIR = self.outputs_dir
        app_module.RATE_LIMIT_BUCKETS.clear()

        app_module.ensure_directories()
        app_module.init_db()
        app_module.workflows.init_workflow_tables(app_module.get_db())
        app_module.gallery.init_gallery_tables(app_module.get_db())

    def tearDown(self) -> None:
        if app_module.DB_CONN is not None:
            app_module.DB_CONN.close()
        app_module.DB_CONN = None
        app_module.DB_PATH = self._db_path_backup
        app_module.OUTPUTS_DIR = self._outputs_backup
        app_module.RATE_LIMIT_BUCKETS.clear()
        app_module.RATE_LIMIT_BUCKETS.update(self._rate_limit_backup)
        self.tmpdir.cleanup()

    def _seed_job(
        self,
        job_id: str,
        wallet: str = SELLER_WALLET,
        task_type: str = "IMAGE_GEN",
        status: str = "completed",
        model: str = "epicrealismxl_vxviicrystalclear",
    ) -> None:
        now = time.time()
        conn = app_module.get_db()
        conn.execute(
            """
            INSERT INTO jobs (id, wallet, model, data, task_type, weight, status, node_id, timestamp, completed_at, invite_code)
            VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, NULL)
            """,
            (
                job_id,
                wallet,
                model,
                '{"prompt": "cinematic portrait"}',
                task_type,
                10.0,
                status,
                now,
                now,
            ),
        )
        conn.commit()

    def _write_artifact(self, job_id: str, asset_type: str) -> None:
        if asset_type == "video":
            path = self.outputs_dir / "videos" / f"{job_id}.mp4"
        else:
            path = self.outputs_dir / f"{job_id}.png"
        path.write_bytes(b"artifact")

    def _create_listing_via_api(
        self,
        job_id: str,
        wallet: str = SELLER_WALLET,
        price: float = 12.5,
        title: str | None = None,
        category: str = "Portrait",
    ) -> dict:
        resp = self.client.post(
            "/gallery/listings",
            json={
                "wallet": wallet,
                "job_id": job_id,
                "title": title or f"Listing {job_id}",
                "description": "Gallery test listing",
                "price_credits": price,
                "category": category,
            },
        )
        self.assertEqual(resp.status_code, 201)
        body = resp.get_json()
        self.assertIsInstance(body, dict)
        return body

    def test_gallery_browse_returns_image_preview_fields(self) -> None:
        self._seed_job("job-image")
        self._write_artifact("job-image", "image")
        self._create_listing_via_api("job-image")

        resp = self.client.get("/gallery/browse")
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        listing = body["listings"][0]
        self.assertEqual(listing["image_url"], "/static/outputs/job-image.png")
        self.assertIsNone(listing["video_url"])
        self.assertEqual(listing["preview_url"], "/static/outputs/job-image.png")
        self.assertEqual(listing["status"], "active")

    def test_gallery_browse_returns_video_preview_fields(self) -> None:
        self._seed_job("job-video", task_type="VIDEO_GEN", model="ltx2")
        self._write_artifact("job-video", "video")
        self._create_listing_via_api("job-video", price=20.0, title="Video listing")

        resp = self.client.get("/gallery/browse?asset_type=video")
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        listing = body["listings"][0]
        self.assertEqual(listing["asset_type"], "video")
        self.assertIsNone(listing["image_url"])
        self.assertEqual(listing["video_url"], "/static/outputs/videos/job-video.mp4")
        self.assertEqual(listing["preview_url"], "/static/outputs/videos/job-video.mp4")
        self.assertEqual(listing["status"], "active")

    def test_my_listings_include_normalized_status_values(self) -> None:
        self._seed_job("job-active")
        self._seed_job("job-sold")
        self._seed_job("job-delisted")
        self._write_artifact("job-active", "image")
        self._write_artifact("job-sold", "image")
        self._write_artifact("job-delisted", "image")

        active = self._create_listing_via_api("job-active")
        sold = self._create_listing_via_api("job-sold")
        delisted = self._create_listing_via_api("job-delisted")

        app_module.credits.deposit_credits(BUYER_WALLET, 100.0, reason="gallery-test")
        purchase_resp = self.client.post(
            f"/gallery/listings/{sold['id']}/purchase",
            json={"wallet": BUYER_WALLET},
        )
        self.assertEqual(purchase_resp.status_code, 200)

        delist_resp = self.client.delete(
            f"/gallery/listings/{delisted['id']}",
            json={"wallet": SELLER_WALLET},
        )
        self.assertEqual(delist_resp.status_code, 200)

        resp = self.client.get(
            f"/gallery/my-listings?wallet={SELLER_WALLET}&include_sold=true"
        )
        self.assertEqual(resp.status_code, 200)
        listings = {entry["job_id"]: entry["status"] for entry in resp.get_json()["listings"]}
        self.assertEqual(listings["job-active"], "active")
        self.assertEqual(listings["job-sold"], "sold")
        self.assertEqual(listings["job-delisted"], "delisted")
        self.assertIn(active["job_id"], listings)

    def test_gallery_purchases_include_preview_fields(self) -> None:
        self._seed_job("job-purchase", task_type="VIDEO_GEN", model="ltx2")
        self._write_artifact("job-purchase", "video")
        listing = self._create_listing_via_api("job-purchase", price=18.0)

        app_module.credits.deposit_credits(BUYER_WALLET, 100.0, reason="gallery-test")
        resp = self.client.post(
            f"/gallery/listings/{listing['id']}/purchase",
            json={"wallet": BUYER_WALLET},
        )
        self.assertEqual(resp.status_code, 200)

        purchases_resp = self.client.get(f"/gallery/purchases?wallet={BUYER_WALLET}")
        self.assertEqual(purchases_resp.status_code, 200)
        purchase = purchases_resp.get_json()["purchases"][0]
        self.assertEqual(purchase["job_id"], "job-purchase")
        self.assertEqual(purchase["video_url"], "/static/outputs/videos/job-purchase.mp4")
        self.assertEqual(purchase["preview_url"], "/static/outputs/videos/job-purchase.mp4")

    def test_purchase_same_listing_twice_only_records_one_sale(self) -> None:
        self._seed_job("job-double")
        self._write_artifact("job-double", "image")
        listing = self._create_listing_via_api("job-double", price=14.0)

        app_module.credits.deposit_credits(BUYER_WALLET, 100.0, reason="gallery-test")
        app_module.credits.deposit_credits(OTHER_BUYER_WALLET, 100.0, reason="gallery-test")

        first = self.client.post(
            f"/gallery/listings/{listing['id']}/purchase",
            json={"wallet": BUYER_WALLET},
        )
        self.assertEqual(first.status_code, 200)

        second = self.client.post(
            f"/gallery/listings/{listing['id']}/purchase",
            json={"wallet": OTHER_BUYER_WALLET},
        )
        self.assertEqual(second.status_code, 404)
        self.assertEqual(second.get_json()["error"], "listing_not_available")

        conn = app_module.get_db()
        sales_count = conn.execute(
            "SELECT COUNT(*) FROM gallery_sales WHERE listing_id = ?",
            (listing["id"],),
        ).fetchone()[0]
        self.assertEqual(sales_count, 1)

    def test_self_purchase_is_rejected(self) -> None:
        self._seed_job("job-self")
        self._write_artifact("job-self", "image")
        listing = self._create_listing_via_api("job-self")
        app_module.credits.deposit_credits(SELLER_WALLET, 100.0, reason="gallery-test")

        resp = self.client.post(
            f"/gallery/listings/{listing['id']}/purchase",
            json={"wallet": SELLER_WALLET},
        )
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.get_json()["error"], "cannot_buy_own_listing")


if __name__ == "__main__":
    unittest.main()
