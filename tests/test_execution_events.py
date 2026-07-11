"""Regression tests for durable job execution timelines."""

from __future__ import annotations

import sqlite3
import unittest

from server import execution_events


class ExecutionEventTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_get_db = getattr(execution_events, "get_db", None)
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        execution_events.get_db = lambda: self.conn
        execution_events.init_execution_event_tables(self.conn)

    def tearDown(self) -> None:
        if self.original_get_db is None:
            delattr(execution_events, "get_db")
        else:
            execution_events.get_db = self.original_get_db
        self.conn.close()

    def test_timeline_is_ordered_and_calculates_stage_latency(self) -> None:
        execution_events.record_event("job-1", "QUEUED", "QUEUED", created_at=100.0)
        execution_events.record_event(
            "job-1",
            "CLAIMED",
            "RUNNING",
            node_id="node-a",
            attempt_number=1,
            metadata={"dispatch_score": 87.5},
            created_at=102.5,
        )
        execution_events.record_event(
            "job-1", "SETTLED", "SUCCEEDED", node_id="node-a", created_at=110.0
        )

        timeline = execution_events.get_timeline("job-1")

        self.assertEqual(timeline["event_count"], 3)
        self.assertEqual(timeline["current_stage"], "SETTLED")
        self.assertEqual(timeline["total_elapsed_ms"], 10000)
        self.assertEqual([event["sequence"] for event in timeline["events"]], [1, 2, 3])
        self.assertEqual(timeline["events"][1]["stage_latency_ms"], 2500)
        self.assertEqual(timeline["events"][1]["metadata"]["dispatch_score"], 87.5)

    def test_lease_renewal_deduplication_limits_event_volume(self) -> None:
        first = execution_events.record_event(
            "job-2",
            "LEASE_RENEWED",
            "RUNNING",
            node_id="node-a",
            created_at=200.0,
            dedupe_window_seconds=60,
        )
        duplicate = execution_events.record_event(
            "job-2",
            "LEASE_RENEWED",
            "RUNNING",
            node_id="node-a",
            created_at=230.0,
            dedupe_window_seconds=60,
        )
        later = execution_events.record_event(
            "job-2",
            "LEASE_RENEWED",
            "RUNNING",
            node_id="node-a",
            created_at=261.0,
            dedupe_window_seconds=60,
        )

        self.assertIsNotNone(first)
        self.assertIsNone(duplicate)
        self.assertIsNotNone(later)
        self.assertEqual(execution_events.get_timeline("job-2")["event_count"], 2)


if __name__ == "__main__":
    unittest.main()
