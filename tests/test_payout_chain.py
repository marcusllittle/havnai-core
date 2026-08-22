"""Tests for strict node payout publication and claim verification."""

from __future__ import annotations

import unittest

from server import payout_chain


TREASURY = "0x" + "aa" * 20
CONTRACT = "0x" + "bb" * 20
OPERATOR = "0x" + "cc" * 20
TX_HASH = "0x" + "dd" * 32
ROOT = "ee" * 32
BATCH_ID = 7
INDEX = 2
AMOUNT = 1_250_000_000_000_000_000


class PayoutChainTests(unittest.TestCase):
    def _claim_log(self, **overrides):
        log = {
            "address": CONTRACT,
            "topics": [
                "0x" + payout_chain.CLAIM_EVENT_TOPIC.hex(),
                "0x" + BATCH_ID.to_bytes(32, "big").hex(),
                "0x" + INDEX.to_bytes(32, "big").hex(),
                "0x" + (bytes(12) + bytes.fromhex(OPERATOR[2:])).hex(),
            ],
            "data": hex(AMOUNT),
        }
        return {**log, **overrides}

    def _rpc(self, *, calldata="0x", sender=TREASURY, logs=None, overrides=None):
        values = {
            "eth_chainId": hex(payout_chain.SEPOLIA_CHAIN_ID),
            "eth_getTransactionByHash": {
                "hash": TX_HASH,
                "from": sender,
                "to": CONTRACT,
                "value": "0x0",
                "input": calldata,
            },
            "eth_getTransactionReceipt": {
                "transactionHash": TX_HASH,
                "status": "0x1",
                "blockNumber": "0x64",
                "logs": logs or [],
            },
            "eth_blockNumber": "0x65",
        }
        values.update(overrides or {})

        def call(method: str, _params: list):
            return values[method]

        return call

    def test_publish_calldata_is_exact_standard_abi(self) -> None:
        calldata = bytes.fromhex(
            payout_chain.build_publish_calldata(BATCH_ID, ROOT, AMOUNT)[2:]
        )
        self.assertEqual(calldata[:4], payout_chain.PUBLISH_SELECTOR)
        self.assertEqual(int.from_bytes(calldata[4:36], "big"), BATCH_ID)
        self.assertEqual(calldata[36:68], bytes.fromhex(ROOT))
        self.assertEqual(int.from_bytes(calldata[68:100], "big"), AMOUNT)

    def test_accepts_exact_confirmed_treasury_publication(self) -> None:
        calldata = payout_chain.build_publish_calldata(BATCH_ID, ROOT, AMOUNT)
        result = payout_chain.verify_publish_transaction(
            TX_HASH,
            BATCH_ID,
            ROOT,
            AMOUNT,
            self._rpc(calldata=calldata),
            treasury_wallet=TREASURY,
            claim_contract=CONTRACT,
        )
        self.assertTrue(result["verified"])
        self.assertEqual(result["confirmations"], 2)
        self.assertEqual(result["contract"], CONTRACT)

    def test_rejects_wrong_publish_sender_contract_value_and_payload(self) -> None:
        calldata = payout_chain.build_publish_calldata(BATCH_ID, ROOT, AMOUNT)
        transaction = self._rpc(calldata=calldata)("eth_getTransactionByHash", [])
        cases = [
            (
                self._rpc(calldata=calldata, sender="0x" + "11" * 20),
                "wrong_sender",
            ),
            (
                self._rpc(
                    calldata=calldata,
                    overrides={"eth_getTransactionByHash": {**transaction, "to": "0x" + "22" * 20}},
                ),
                "wrong_contract",
            ),
            (
                self._rpc(
                    calldata=calldata,
                    overrides={"eth_getTransactionByHash": {**transaction, "value": "0x1"}},
                ),
                "nonzero_value",
            ),
            (self._rpc(calldata="0x1234"), "publish_payload_mismatch"),
        ]
        for rpc, expected in cases:
            with self.subTest(expected):
                result = payout_chain.verify_publish_transaction(
                    TX_HASH,
                    BATCH_ID,
                    ROOT,
                    AMOUNT,
                    rpc,
                    treasury_wallet=TREASURY,
                    claim_contract=CONTRACT,
                )
                self.assertEqual(result["error"], expected)

    def test_pending_publish_can_be_rechecked_after_confirmations(self) -> None:
        calldata = payout_chain.build_publish_calldata(BATCH_ID, ROOT, AMOUNT)
        result = payout_chain.verify_publish_transaction(
            TX_HASH,
            BATCH_ID,
            ROOT,
            AMOUNT,
            self._rpc(calldata=calldata, overrides={"eth_blockNumber": "0x64"}),
            treasury_wallet=TREASURY,
            claim_contract=CONTRACT,
        )
        self.assertTrue(result["pending"])
        self.assertEqual(result["confirmations"], 1)
        self.assertNotIn("receipt", result)

    def test_accepts_exact_claim_event_even_when_relayed(self) -> None:
        result = payout_chain.verify_claim_transaction(
            TX_HASH,
            BATCH_ID,
            INDEX,
            OPERATOR,
            AMOUNT,
            self._rpc(sender="0x" + "44" * 20, logs=[self._claim_log()]),
            claim_contract=CONTRACT,
        )
        self.assertTrue(result["verified"])
        self.assertEqual(result["wallet"], OPERATOR)
        self.assertEqual(result["amount_wei"], str(AMOUNT))

    def test_rejects_claim_event_with_wrong_account_or_amount(self) -> None:
        cases = [
            self._claim_log(
                topics=[
                    "0x" + payout_chain.CLAIM_EVENT_TOPIC.hex(),
                    "0x" + BATCH_ID.to_bytes(32, "big").hex(),
                    "0x" + INDEX.to_bytes(32, "big").hex(),
                    "0x" + (bytes(12) + bytes.fromhex(("0x" + "55" * 20)[2:])).hex(),
                ]
            ),
            self._claim_log(data=hex(AMOUNT + 1)),
        ]
        for log in cases:
            result = payout_chain.verify_claim_transaction(
                TX_HASH,
                BATCH_ID,
                INDEX,
                OPERATOR,
                AMOUNT,
                self._rpc(logs=[log]),
                claim_contract=CONTRACT,
            )
            self.assertEqual(result["error"], "claim_event_mismatch")


if __name__ == "__main__":
    unittest.main()
