"""Tests for the Astra Valkyries game economy.

Covers the three defects that made the live endpoints unsafe:
  - MAX_CREDITS_PER_RUN was multiplied by the bonus multiplier, so a
    first-win streak run paid 45 credits instead of the documented 15.
  - Replay detection bucketed on wall-clock minutes, so an identical run
    could be resubmitted one minute later.
  - process_spend committed the deduction before writing its audit row,
    and had no idempotency key, so a retry double-charged.
"""

from __future__ import annotations

import importlib.util
import sqlite3
import sys
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
ASTRA_PATH = ROOT / "server" / "astra_rewards.py"
ASTRA_SPEC = importlib.util.spec_from_file_location("havnai_server_astra_rewards", ASTRA_PATH)
assert ASTRA_SPEC is not None and ASTRA_SPEC.loader is not None
astra = importlib.util.module_from_spec(ASTRA_SPEC)
sys.modules[ASTRA_SPEC.name] = astra
ASTRA_SPEC.loader.exec_module(astra)


WALLET = "0x" + "a" * 40
OTHER_WALLET = "0x" + "b" * 40


class AstraTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        astra.get_db = lambda: self.conn
        astra.log_event = lambda *a, **k: None
        astra.init_astra_tables(self.conn)

    def tearDown(self) -> None:
        self.conn.close()

    # -- helpers ----------------------------------------------------------
    def _open_run(self, wallet: str = WALLET, age_seconds: float = 0.0) -> str:
        """Start a run, optionally backdating it past MIN_RUN_DURATION_SECONDS."""
        token = astra.start_run(wallet, "nebula-runway")["run_token"]
        if age_seconds:
            self.conn.execute(
                "UPDATE astra_run_tokens SET started_at = ? WHERE wallet = ? AND consumed_at IS NULL",
                (time.time() - age_seconds, wallet),
            )
            self.conn.commit()
        return token

    def _deposit_fn(self, wallet, amount, reason):
        self.deposited.append(amount)
        return 1000.0

    @staticmethod
    def _ledger_fn(*args, **kwargs):
        return None


class SessionTests(AstraTestCase):
    def test_session_resolves_to_wallet(self) -> None:
        token = astra.create_session(WALLET, 3600)["token"]
        self.assertEqual(astra.resolve_session(token), WALLET)

    def test_unknown_and_empty_tokens_rejected(self) -> None:
        self.assertIsNone(astra.resolve_session("fabricated"))
        self.assertIsNone(astra.resolve_session(""))

    def test_expired_session_rejected(self) -> None:
        token = astra.create_session(WALLET, -5)["token"]
        self.assertIsNone(astra.resolve_session(token))


class RunTokenTests(AstraTestCase):
    def test_token_can_only_be_consumed_once(self) -> None:
        token = self._open_run()
        started_at, error = astra.consume_run_token(WALLET, token)
        self.assertIsNone(error)
        self.assertIsNotNone(started_at)

        _, second_error = astra.consume_run_token(WALLET, token)
        self.assertEqual(second_error, "run_token_used")

    def test_token_is_bound_to_issuing_wallet(self) -> None:
        token = self._open_run()
        _, error = astra.consume_run_token(OTHER_WALLET, token)
        self.assertEqual(error, "run_token_wallet_mismatch")

    def test_unknown_token_rejected(self) -> None:
        _, error = astra.consume_run_token(WALLET, "fabricated")
        self.assertEqual(error, "unknown_run_token")


class RewardTests(AstraTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.deposited: list[float] = []

    def _submit(self, *, score: int, run_token: str, duration_s: float = 999.0):
        return astra.submit_reward(
            wallet=WALLET,
            score=score,
            grade="S",
            duration_s=duration_s,
            map_id="nebula-runway",
            deposit_fn=self._deposit_fn,
            ledger_fn=self._ledger_fn,
            run_token=run_token,
        )

    def test_max_score_run_is_capped_at_max_credits_per_run(self) -> None:
        """Regression: first-win (2.0x) and streak (1.5x) previously lifted the
        ceiling to 45 because the multiplier was applied to the cap itself."""
        token = self._open_run(age_seconds=300)
        result = self._submit(score=100_000, run_token=token)

        self.assertTrue(result["ok"])
        self.assertLessEqual(result["reward"], astra.MAX_CREDITS_PER_RUN)
        self.assertEqual(result["reward"], float(astra.MAX_CREDITS_PER_RUN))

    def test_reward_requires_a_run_token(self) -> None:
        result = self._submit(score=100_000, run_token="")
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "missing_run_token")

    def test_fabricated_run_token_rejected(self) -> None:
        result = self._submit(score=100_000, run_token="fabricated")
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "unknown_run_token")

    def test_client_cannot_fake_run_duration(self) -> None:
        """Duration comes from the server's own start timestamp."""
        token = self._open_run()  # opened just now
        result = self._submit(score=100_000, run_token=token, duration_s=9999.0)
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "run_too_short")

    def test_run_token_is_burned_even_on_rejection(self) -> None:
        """Otherwise a client could probe thresholds by resubmitting."""
        token = self._open_run(age_seconds=300)
        first = self._submit(score=10, run_token=token)
        self.assertEqual(first["reason"], "score_too_low")

        second = self._submit(score=100_000, run_token=token)
        self.assertEqual(second["reason"], "run_token_used")

    def test_low_score_rejected(self) -> None:
        token = self._open_run(age_seconds=300)
        result = self._submit(score=astra.MIN_SCORE_THRESHOLD - 1, run_token=token)
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "score_too_low")


class SpendTests(AstraTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.balance = 500.0

    def _deduct_fn(self, wallet, amount, reason):
        if self.balance < amount:
            return False, self.balance
        self.balance -= amount
        return True, self.balance

    def _spend(self, action="gacha_1", idempotency_key=None):
        return astra.process_spend(
            wallet=WALLET,
            action=action,
            deduct_fn=self._deduct_fn,
            ledger_fn=self._ledger_fn,
            idempotency_key=idempotency_key,
            balance_fn=lambda w: self.balance,
        )

    def test_spend_deducts_once(self) -> None:
        result = self._spend(idempotency_key="key-1")
        self.assertTrue(result["ok"])
        self.assertEqual(self.balance, 500.0 - astra.SPEND_COSTS["gacha_1"])

    def test_retry_with_same_key_does_not_double_charge(self) -> None:
        self._spend(idempotency_key="key-1")
        balance_after_first = self.balance

        replay = self._spend(idempotency_key="key-1")
        self.assertTrue(replay["ok"])
        self.assertTrue(replay["replayed"])
        self.assertEqual(self.balance, balance_after_first)

    def test_distinct_keys_charge_separately(self) -> None:
        self._spend(idempotency_key="key-1")
        self._spend(idempotency_key="key-2")
        self.assertEqual(self.balance, 500.0 - 2 * astra.SPEND_COSTS["gacha_1"])

    def test_invalid_action_rejected_without_charging(self) -> None:
        result = self._spend(action="definitely_not_an_action")
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "invalid_action")
        self.assertEqual(self.balance, 500.0)

    def test_insufficient_credits_leaves_audit_row(self) -> None:
        self.balance = 1.0
        result = self._spend(action="gacha_10", idempotency_key="key-broke")

        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "insufficient_credits")
        self.assertEqual(self.balance, 1.0)

        row = self.conn.execute(
            "SELECT status FROM astra_spends WHERE idempotency_key = 'key-broke'"
        ).fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row["status"], "failed")

    def test_failed_spend_is_not_replayed_as_success(self) -> None:
        self.balance = 1.0
        self._spend(action="gacha_10", idempotency_key="key-broke")

        self.balance = 500.0
        retry = self._spend(action="gacha_10", idempotency_key="key-broke")
        self.assertFalse(retry["ok"])
        self.assertEqual(self.balance, 500.0)


if __name__ == "__main__":
    unittest.main()
