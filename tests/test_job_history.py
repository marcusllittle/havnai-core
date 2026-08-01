"""Tests for per-wallet job history.

These cover the reason the Library page lost images: it was localStorage
only, capped at 200, and scoped to one browser. The server-side history is
what makes the Collection durable, so the things worth pinning are that it
is wallet-scoped, ordered newest-first, paginates without dropping rows,
and reports a total that is not clamped to the page size.
"""

from __future__ import annotations

import importlib.util
import sqlite3
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MODULE_PATH = ROOT / "server" / "job_history.py"
SPEC = importlib.util.spec_from_file_location("havnai_server_job_history", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
job_history = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = job_history
SPEC.loader.exec_module(job_history)


WALLET = "0x" + "a" * 40
OTHER_WALLET = "0x" + "b" * 40


class JobHistoryTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        self.conn.execute(
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
                invite_code TEXT
            )
            """
        )
        job_history.get_db = lambda: self.conn

    def tearDown(self) -> None:
        self.conn.close()

    def _add(
        self,
        job_id: str,
        wallet: str = WALLET,
        task_type: str = "IMAGE_GEN",
        timestamp: float = 1000.0,
        status: str = "completed",
    ) -> None:
        self.conn.execute(
            "INSERT INTO jobs (id, wallet, model, task_type, weight, status, timestamp)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (job_id, wallet, "sdxl", task_type, 1.0, status, timestamp),
        )

    # -- scoping ----------------------------------------------------------
    def test_only_returns_the_requested_wallets_jobs(self) -> None:
        self._add("mine-1")
        self._add("mine-2")
        self._add("theirs-1", wallet=OTHER_WALLET)

        result = job_history.get_wallet_jobs(WALLET)

        self.assertEqual({j["job_id"] for j in result["jobs"]}, {"mine-1", "mine-2"})
        self.assertEqual(result["total"], 2)

    def test_unknown_wallet_returns_empty_not_error(self) -> None:
        self._add("mine-1")
        result = job_history.get_wallet_jobs(OTHER_WALLET)
        self.assertEqual(result["jobs"], [])
        self.assertEqual(result["total"], 0)

    # -- ordering and paging ----------------------------------------------
    def test_newest_first(self) -> None:
        self._add("old", timestamp=100.0)
        self._add("newest", timestamp=300.0)
        self._add("middle", timestamp=200.0)

        ids = [j["job_id"] for j in job_history.get_wallet_jobs(WALLET)["jobs"]]

        self.assertEqual(ids, ["newest", "middle", "old"])

    def test_paging_covers_every_row_exactly_once(self) -> None:
        for i in range(25):
            self._add(f"job-{i:02d}", timestamp=float(i))

        seen = []
        for offset in range(0, 25, 10):
            page = job_history.get_wallet_jobs(WALLET, limit=10, offset=offset)
            seen.extend(j["job_id"] for j in page["jobs"])

        self.assertEqual(len(seen), 25)
        self.assertEqual(len(set(seen)), 25, "a job appeared on two pages")

    def test_total_is_not_clamped_to_the_page(self) -> None:
        """The 200-entry cap is exactly the bug being fixed; total must be honest."""
        for i in range(250):
            self._add(f"job-{i:03d}", timestamp=float(i))

        page = job_history.get_wallet_jobs(WALLET, limit=10)

        self.assertEqual(len(page["jobs"]), 10)
        self.assertEqual(page["total"], 250)

    def test_limit_is_clamped_to_max(self) -> None:
        self._add("only")
        page = job_history.get_wallet_jobs(WALLET, limit=10_000)
        self.assertEqual(page["limit"], job_history.MAX_LIMIT)

    def test_negative_paging_values_do_not_break_the_query(self) -> None:
        self._add("only")
        page = job_history.get_wallet_jobs(WALLET, limit=-5, offset=-20)
        self.assertEqual(page["offset"], 0)
        self.assertEqual([j["job_id"] for j in page["jobs"]], ["only"])

    # -- media typing ------------------------------------------------------
    def test_video_task_types_are_typed_as_video(self) -> None:
        self._add("v1", task_type="VIDEO_GEN")
        self._add("v2", task_type="ANIMATEDIFF")
        self._add("i1", task_type="IMAGE_GEN")
        self._add("i2", task_type="FACE_SWAP")

        types = {j["job_id"]: j["type"] for j in job_history.get_wallet_jobs(WALLET)["jobs"]}

        self.assertEqual(types["v1"], "video")
        self.assertEqual(types["v2"], "video")
        self.assertEqual(types["i1"], "image")
        self.assertEqual(types["i2"], "image")

    def test_task_type_casing_does_not_change_the_answer(self) -> None:
        self._add("v1", task_type="video_gen")
        types = {j["job_id"]: j["type"] for j in job_history.get_wallet_jobs(WALLET)["jobs"]}
        self.assertEqual(types["v1"], "video")

    # -- status ------------------------------------------------------------
    def test_incomplete_jobs_are_included_with_their_status(self) -> None:
        """A queued job still belongs in the Collection, greyed out."""
        self._add("running-1", status="running", timestamp=200.0)
        self._add("done-1", status="completed", timestamp=100.0)

        jobs = job_history.get_wallet_jobs(WALLET)["jobs"]

        self.assertEqual([j["status"] for j in jobs], ["running", "completed"])


if __name__ == "__main__":
    unittest.main()
