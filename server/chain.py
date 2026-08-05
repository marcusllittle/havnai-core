"""EVM chain integration for HavnAI.

Wraps web3.py for HAI token interactions.  All public functions degrade
gracefully (return None / False) when CHAIN_RPC_URL is unset, so the
server runs normally in dev without a node.

Environment vars:
  CHAIN_RPC_URL       HTTP/WSS RPC endpoint (Alchemy, Infura, local Hardhat/Anvil)
  HAI_TOKEN_ADDRESS   ERC-20 contract address for the HAI token
  HAVNAI_PAYER_KEY    Hex private key of the server-side funding wallet
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

CHAIN_RPC_URL: Optional[str] = os.getenv("CHAIN_RPC_URL", "").strip() or None
HAI_TOKEN_ADDRESS: Optional[str] = os.getenv("HAI_TOKEN_ADDRESS", "").strip() or None
HAVNAI_PAYER_KEY: Optional[str] = os.getenv("HAVNAI_PAYER_KEY", "").strip() or None

_ERC20_ABI = [
    {
        "inputs": [{"name": "account", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [
            {"name": "to", "type": "address"},
            {"name": "amount", "type": "uint256"},
        ],
        "name": "transfer",
        "outputs": [{"name": "", "type": "bool"}],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "decimals",
        "outputs": [{"name": "", "type": "uint8"}],
        "stateMutability": "view",
        "type": "function",
    },
]

_w3 = None
_token_contract = None
_decimals: Optional[int] = None


def _get_w3():
    global _w3
    if _w3 is not None:
        return _w3
    if not CHAIN_RPC_URL:
        return None
    try:
        from web3 import Web3
        w3 = Web3(Web3.HTTPProvider(CHAIN_RPC_URL))
        if w3.is_connected():
            _w3 = w3
            logger.info("chain: connected to %s (chain_id=%s)", CHAIN_RPC_URL, w3.eth.chain_id)
        else:
            logger.warning("chain: could not connect to %s", CHAIN_RPC_URL)
    except Exception as exc:
        logger.warning("chain: web3 init failed: %s", exc)
    return _w3


def _get_token():
    global _token_contract, _decimals
    if _token_contract is not None:
        return _token_contract
    w3 = _get_w3()
    if w3 is None or not HAI_TOKEN_ADDRESS:
        return None
    try:
        from web3 import Web3
        addr = Web3.to_checksum_address(HAI_TOKEN_ADDRESS)
        _token_contract = w3.eth.contract(address=addr, abi=_ERC20_ABI)
        _decimals = _token_contract.functions.decimals().call()
        logger.info("chain: HAI token at %s (decimals=%s)", addr, _decimals)
    except Exception as exc:
        logger.warning("chain: token init failed: %s", exc)
    return _token_contract


def get_decimals() -> int:
    global _decimals
    if _decimals is None:
        _get_token()
    return _decimals if _decimals is not None else 18


def to_wei(amount: float) -> int:
    """Convert a float credit amount to token wei."""
    return int(amount * (10 ** get_decimals()))


def from_wei(amount_wei: int) -> float:
    """Convert raw token units to float credits."""
    return amount_wei / (10 ** get_decimals())


def is_connected() -> bool:
    """Return True if a live chain connection exists."""
    w3 = _get_w3()
    return w3 is not None and w3.is_connected()


def get_hai_balance(wallet: str) -> Optional[int]:
    """Return raw HAI balance (wei) for a wallet, or None if offline."""
    token = _get_token()
    if token is None:
        return None
    try:
        from web3 import Web3
        return token.functions.balanceOf(Web3.to_checksum_address(wallet)).call()
    except Exception as exc:
        logger.warning("chain: balanceOf(%s) failed: %s", wallet, exc)
        return None


def send_hai(to: str, amount_float: float) -> Optional[str]:
    """Transfer HAI from the configured payer wallet to a recipient.

    Returns the tx_hash hex string on success, or None on failure.
    Requires HAVNAI_PAYER_KEY and HAI_TOKEN_ADDRESS to be set.
    """
    w3 = _get_w3()
    token = _get_token()
    if w3 is None or token is None or not HAVNAI_PAYER_KEY:
        logger.warning("chain: send_hai skipped — chain not configured")
        return None
    try:
        from web3 import Web3
        account = w3.eth.account.from_key(HAVNAI_PAYER_KEY)
        to_addr = Web3.to_checksum_address(to)
        amount_wei = to_wei(amount_float)
        nonce = w3.eth.get_transaction_count(account.address, "pending")
        tx = token.functions.transfer(to_addr, amount_wei).build_transaction({
            "from": account.address,
            "nonce": nonce,
            "gasPrice": w3.eth.gas_price,
        })
        tx["gas"] = w3.eth.estimate_gas(tx)
        signed = account.sign_transaction(tx)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
        hex_hash = tx_hash.hex()
        logger.info("chain: sent %.4f HAI to %s tx=%s", amount_float, to, hex_hash)
        return hex_hash
    except Exception as exc:
        logger.error("chain: send_hai(%.4f -> %s) failed: %s", amount_float, to, exc)
        return None


def verify_eip191_signature(wallet: str, message: str, signature: str) -> bool:
    """Verify an EIP-191 personal_sign signature.

    Returns True if `signature` was produced by `wallet` signing `message`.
    Works without a live node — eth_account handles crypto locally.
    """
    try:
        from web3 import Web3
        from eth_account.messages import encode_defunct
        w3_local = _get_w3() or Web3()
        msg = encode_defunct(text=message)
        recovered = w3_local.eth.account.recover_message(msg, signature=signature)
        return recovered.lower() == wallet.lower()
    except Exception as exc:
        logger.warning("chain: signature verification failed: %s", exc)
        return False
