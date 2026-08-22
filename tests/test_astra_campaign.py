"""Tests for the shared Astra campaign's authoritative aggregation rules."""

from __future__ import annotations

import importlib.util
import sqlite3
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "havnai_server_astra_campaign", ROOT / "server" / "astra_campaign.py"
)
assert SPEC is not None and SPEC.loader is not None
campaign = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = campaign
SPEC.loader.exec_module(campaign)

WALLET = "0x" + "a" * 40
OTHER_WALLET = "0x" + "b" * 40


class AstraCampaignTests(unittest.TestCase):
    def setUp(self) -> None:
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(
            """
            CREATE TABLE astra_runs (
                run_id TEXT PRIMARY KEY, wallet TEXT, score INTEGER, grade TEXT,
                duration_s REAL, map_id TEXT, reward REAL, run_hash TEXT, created_at REAL
            );
            CREATE TABLE astra_reward_images (
                run_id TEXT PRIMARY KEY, job_id TEXT, video_job_id TEXT, wallet TEXT,
                pilot_id TEXT, outfit_id TEXT, map_id TEXT, grade TEXT, created_at REAL
            );
            CREATE TABLE jobs (
                id TEXT PRIMARY KEY, status TEXT, node_id TEXT, completed_at REAL
            );
            CREATE TABLE job_settlement (
                job_id TEXT PRIMARY KEY, execution_status TEXT,
                quality_status TEXT, settlement_outcome TEXT,
                assigned_node_id TEXT, updated_at REAL
            );
            """
        )
        campaign.get_db = lambda: self.conn
        self.now = campaign.MONDAY_EPOCH + 3000 * campaign.WEEK_SECONDS + 3600
        _, self.starts_at, _ = campaign._campaign_window(self.now)

    def tearDown(self) -> None:
        self.conn.close()

    def _run(
        self,
        run_id: str,
        *,
        wallet: str = WALLET,
        score: int = 5000,
        map_id: str = "nebula-runway",
        reward: float = 2.0,
        created_at: float | None = None,
    ) -> None:
        self.conn.execute(
            "INSERT INTO astra_runs VALUES (?, ?, ?, 'A', 60, ?, ?, ?, ?)",
            (
                run_id,
                wallet,
                score,
                map_id,
                reward,
                f"hash-{run_id}",
                self.starts_at + 100 if created_at is None else created_at,
            ),
        )

    def _artifact(
        self,
        job_id: str,
        *,
        wallet: str = WALLET,
        map_id: str = "nebula-runway",
        status: str = "succeeded",
        execution_status: str = "settled",
        quality_status: str = "valid",
        outcome: str = "released",
        video_job_id: str | None = None,
    ) -> None:
        self.conn.execute(
            "INSERT INTO astra_reward_images VALUES (?, ?, ?, ?, 'pilot_nova', "
            "'outfit_17', ?, 'A', ?)",
            (f"run-{job_id}", job_id, video_job_id, wallet, map_id, self.starts_at + 200),
        )
        self.conn.execute(
            "INSERT INTO jobs VALUES (?, ?, 'creator-one', ?)",
            (job_id, status, self.starts_at + 300),
        )
        self.conn.execute(
            "INSERT INTO job_settlement VALUES (?, ?, ?, ?, 'creator-one', ?)",
            (job_id, execution_status, quality_status, outcome, self.starts_at + 300),
        )
        if video_job_id:
            self.conn.execute(
                "INSERT INTO jobs VALUES (?, 'succeeded', 'creator-two', ?)",
                (video_job_id, self.starts_at + 400),
            )
            self.conn.execute(
                "INSERT INTO job_settlement VALUES (?, 'settled', 'valid', 'released', 'creator-two', ?)",
                (video_job_id, self.starts_at + 400),
            )

    def test_aggregates_only_target_runs_and_final_creator_settlements(self) -> None:
        self._run("run-low", score=5000)
        self._run("run-high", wallet=OTHER_WALLET, score=50000)
        self._run("run-off-sector", map_id="solar-rift", score=50000)
        self._run("run-unrewarded", score=50000, reward=0)
        self._artifact("job-complete", video_job_id="job-video")
        self._artifact("job-running", status="running")
        self._artifact("job-unsettled", execution_status="claimed", outcome="pending")
        self._artifact("job-malformed", quality_status="malformed", outcome="full_refund")
        self.conn.commit()

        result = campaign.get_campaign(WALLET, now=self.now)

        self.assertEqual(result["map_id"], "nebula-runway")
        self.assertEqual(result["combat"]["current"], 11)
        self.assertEqual(result["combat"]["accepted_runs"], 2)
        self.assertEqual(result["combat"]["contributors"], 2)
        self.assertEqual(result["forge"]["current"], 3)
        self.assertEqual(result["forge"]["settled_artifacts"], 2)
        self.assertEqual(result["forge"]["creator_nodes"], 2)
        self.assertEqual(result["personal"]["combat_points"], 1)
        self.assertEqual(result["personal"]["forge_points"], 3)

    def test_public_events_use_campaign_scoped_aliases(self) -> None:
        self._run("run-private")
        self.conn.commit()

        result = campaign.get_campaign(now=self.now)
        serialized = str(result)

        self.assertNotIn(WALLET, serialized)
        self.assertRegex(result["recent_events"][0]["actor"], r"^pilot-[0-9a-f]{8}$")
        self.assertIsNone(result["personal"])

    def test_forge_work_counts_when_it_settles_not_when_it_was_requested(self) -> None:
        self._artifact("job-cross-week")
        self.conn.execute(
            "UPDATE astra_reward_images SET created_at = ? WHERE job_id = 'job-cross-week'",
            (self.starts_at - 60,),
        )
        self.conn.commit()

        result = campaign.get_campaign(now=self.now)

        self.assertEqual(result["forge"]["current"], 1)
        self.assertEqual(result["forge"]["settled_artifacts"], 1)

    def test_run_contribution_is_bounded_and_sector_specific(self) -> None:
        self._run("run-front", score=999999)
        self._run("run-away", map_id="solar-rift", score=999999)
        self.conn.commit()

        front = campaign.get_run_contribution("run-front", now=self.now)
        away = campaign.get_run_contribution("run-away", now=self.now)

        self.assertTrue(front["eligible"])
        self.assertEqual(front["combat_points"], 10)
        self.assertFalse(away["eligible"])
        self.assertEqual(away["combat_points"], 0)
        self.assertEqual(away["target_map_id"], "nebula-runway")

    def test_campaign_rotates_deterministically_each_week(self) -> None:
        current = campaign.get_campaign(now=self.now)
        following = campaign.get_campaign(now=self.now + campaign.WEEK_SECONDS)

        self.assertEqual(current["map_id"], "nebula-runway")
        self.assertEqual(following["map_id"], "solar-rift")
        self.assertNotEqual(current["campaign_id"], following["campaign_id"])
        self.assertEqual(current["ends_at"] - current["starts_at"], campaign.WEEK_SECONDS)

    def test_phase_requires_both_players_and_creator_network(self) -> None:
        self.assertEqual(campaign._phase(0, 0), "contested")
        self.assertEqual(campaign._phase(campaign.COMBAT_TARGET, 0), "awaiting_forge")
        self.assertEqual(campaign._phase(0, campaign.FORGE_TARGET), "awaiting_victories")
        self.assertEqual(
            campaign._phase(campaign.COMBAT_TARGET, campaign.FORGE_TARGET),
            "secured",
        )


if __name__ == "__main__":
    unittest.main()
