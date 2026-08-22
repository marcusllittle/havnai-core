# HavnAI Node Reward Claims

`HavnNodeRewardClaims.sol` turns finalized coordinator payout records into
self-custodial Sepolia HAI claims. The coordinator builds proofs and verifies
transactions; it never stores a treasury or operator private key.

This contract is testnet code and has not received an independent production
security audit.

## Compile

```bash
npx --yes solc@0.8.30 --bin --abi --optimize \
  -o /tmp/havnai-node-claims-build \
  contracts/HavnNodeRewardClaims.sol
```

Deploy `HavnNodeRewardClaims` with:

1. `tokenAddress`: the Sepolia HAI ERC-20 contract.
2. `initialOwner`: the treasury wallet that will publish payout roots.

The treasury remains in MetaMask. Do not add a treasury private key to the
coordinator, node service, systemd environment, or repository.

## Coordinator Environment

```text
HAVNAI_SEPOLIA_RPC_URL=<Sepolia JSON-RPC endpoint>
HAVNAI_HAI_TOKEN_ADDRESS=<Sepolia HAI token address>
HAVNAI_HAI_TREASURY_WALLET=<treasury owner address>
HAVNAI_NODE_CLAIM_CONTRACT=<deployed HavnNodeRewardClaims address>
HAVNAI_NODE_CLAIM_CONFIRMATIONS=2
```

Restart the coordinator after changing these values. The worker node does not
need contract credentials or a private key; it only needs its durable operator
wallet registration.

## Settlement Flow

1. The coordinator records finalized node work as `simulated_hai`.
2. The treasury signs a nonce in `/node-rewards` to snapshot eligible payouts.
3. MetaMask funds only the contract reserve deficit and publishes the immutable
   Merkle root on Sepolia.
4. Each operator opens `/node-rewards` with the registered wallet and submits
   its proof-bound claim directly to the contract.
5. The coordinator changes the source records to `onchain_hai` only after the
   exact `RewardClaimed` event reaches the configured confirmation count.

Contract funding and root publication are separate transactions. If funding
succeeds but publication is rejected, the tokens remain uncommitted surplus and
can be reused by a later batch or withdrawn by the treasury owner.
