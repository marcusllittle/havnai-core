"""Tests for strict Sepolia receipt-root transaction verification."""

from __future__ import annotations

import unittest

from server import receipt_anchors


TREASURY = "0x" + "a" * 40
TX_HASH = "0x" + "b" * 64
ROOT = "c" * 64
BATCH_ID = 7


class ReceiptAnchorTests(unittest.TestCase):
    def _rpc(self, overrides: dict[str, object] | None = None):
        calldata = receipt_anchors.build_calldata(BATCH_ID, ROOT)
        values: dict[str, object] = {
            "eth_chainId": hex(receipt_anchors.SEPOLIA_CHAIN_ID),
            "eth_getTransactionByHash": {
                "hash": TX_HASH,
                "from": TREASURY,
                "to": TREASURY,
                "value": "0x0",
                "input": calldata,
            },
            "eth_getTransactionReceipt": {
                "transactionHash": TX_HASH,
                "status": "0x1",
                "blockNumber": "0x64",
            },
            "eth_blockNumber": "0x65",
        }
        values.update(overrides or {})

        def call(method: str, _params: list):
            return values[method]

        return call

    def test_calldata_is_deterministic_and_commits_batch_and_root(self) -> None:
        calldata = receipt_anchors.build_calldata(BATCH_ID, ROOT)
        raw = bytes.fromhex(calldata[2:])
        self.assertTrue(raw.startswith(receipt_anchors.ANCHOR_MAGIC))
        offset = len(receipt_anchors.ANCHOR_MAGIC)
        self.assertEqual(int.from_bytes(raw[offset : offset + 8], "big"), BATCH_ID)
        self.assertEqual(raw[offset + 8 :], bytes.fromhex(ROOT))

    def test_accepts_exact_confirmed_treasury_self_transaction(self) -> None:
        result = receipt_anchors.verify_anchor_transaction(
            TX_HASH,
            BATCH_ID,
            ROOT,
            self._rpc(),
            treasury_wallet=TREASURY,
        )
        self.assertTrue(result["verified"])
        self.assertEqual(result["confirmations"], 2)
        self.assertEqual(result["block_number"], 100)

    def test_rejects_wrong_sender_and_payload(self) -> None:
        transaction = self._rpc()("eth_getTransactionByHash", [])
        assert isinstance(transaction, dict)
        wrong_sender = self._rpc(
            {"eth_getTransactionByHash": {**transaction, "from": "0x" + "d" * 40}}
        )
        sender_result = receipt_anchors.verify_anchor_transaction(
            TX_HASH, BATCH_ID, ROOT, wrong_sender, treasury_wallet=TREASURY
        )
        self.assertEqual(sender_result["error"], "wrong_sender")

        wrong_payload = self._rpc(
            {"eth_getTransactionByHash": {**transaction, "input": "0x1234"}}
        )
        payload_result = receipt_anchors.verify_anchor_transaction(
            TX_HASH, BATCH_ID, ROOT, wrong_payload, treasury_wallet=TREASURY
        )
        self.assertEqual(payload_result["error"], "anchor_payload_mismatch")

    def test_pending_transaction_and_confirmations_are_retryable(self) -> None:
        missing = receipt_anchors.verify_anchor_transaction(
            TX_HASH,
            BATCH_ID,
            ROOT,
            self._rpc({"eth_getTransactionReceipt": None}),
            treasury_wallet=TREASURY,
        )
        self.assertTrue(missing["pending"])
        self.assertEqual(missing["error"], "transaction_pending")

        one_confirmation = receipt_anchors.verify_anchor_transaction(
            TX_HASH,
            BATCH_ID,
            ROOT,
            self._rpc({"eth_blockNumber": "0x64"}),
            treasury_wallet=TREASURY,
        )
        self.assertTrue(one_confirmation["pending"])
        self.assertEqual(one_confirmation["confirmations"], 1)
        self.assertEqual(one_confirmation["tx_hash"], TX_HASH)
        self.assertEqual(one_confirmation["from"], TREASURY)
        self.assertEqual(
            one_confirmation["calldata"],
            receipt_anchors.build_calldata(BATCH_ID, ROOT),
        )

    def test_rejects_value_transfer_wrong_chain_and_failed_transaction(self) -> None:
        transaction = self._rpc()("eth_getTransactionByHash", [])
        receipt = self._rpc()("eth_getTransactionReceipt", [])
        assert isinstance(transaction, dict) and isinstance(receipt, dict)
        cases = [
            (
                {"eth_getTransactionByHash": {**transaction, "value": "0x1"}},
                "nonzero_value",
            ),
            ({"eth_chainId": "0x1"}, "wrong_chain"),
            ({"eth_getTransactionReceipt": {**receipt, "status": "0x0"}}, "transaction_failed"),
        ]
        for overrides, expected in cases:
            with self.subTest(expected):
                result = receipt_anchors.verify_anchor_transaction(
                    TX_HASH,
                    BATCH_ID,
                    ROOT,
                    self._rpc(overrides),
                    treasury_wallet=TREASURY,
                )
                self.assertFalse(result["verified"])
                self.assertEqual(result["error"], expected)


if __name__ == "__main__":
    unittest.main()
