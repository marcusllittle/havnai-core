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

## Local MetaMask deployment helper

`contracts/deploy` builds a local page for deploying this contract using MetaMask.
The prepared configuration is Sepolia, HAI token
`0x8aaa7f1075de91b2103542379bfd3b889c89105f`, and treasury owner
`0x7110347e2bcd02f5f3485dc6bec5e0b5f9eb9262`.

Install `solc@0.8.30`, `ethers@6.17.0`, and (for local tests) `ganache@7.9.2`
in a separate tools directory. Set `NODE_PATH` to that directory's `node_modules`:

```text
node contracts/deploy/build.cjs <output-directory>
node contracts/deploy/verify-local.cjs <output-directory>
node <output-directory>/serve.cjs
```

Open `http://127.0.0.1:8787` in the browser containing MetaMask. Connect the
treasury account, inspect the estimate, then explicitly approve deployment.
Compilation pins Solidity 0.8.30, optimizer 200 runs, and Shanghai EVM.
Compiler input and source are included for review and explorer verification.

The page persists a deployment intent before requesting a transaction to avoid
duplicates on refresh or interrupted wallet responses. Under **Resume an
existing deployment**, check the saved transaction hash after confirmation.
If the wallet response was interrupted before a hash was saved, obtain that
hash from MetaMask Activity instead of deploying again. A rejected request
(4001) clears the intent and permits retry.

Verification checks transaction sender, creation data, zero ETH value, runtime
bytecode (with immutable slots masked), token, owner, claim domain, and two
confirmations. Download the resulting deployment record and use its address in
`HAVNAI_NODE_CLAIM_CONTRACT`. The page does not fund or publish reward batches.

## Coordinator Environment

Verified Sepolia deployment: `0x37ADe176ac43cd4e37Cd72054028cde7B8653E25`.
Transaction and build provenance are recorded in
[`deployments/sepolia-node-rewards.json`](deployments/sepolia-node-rewards.json).
Its owner and token match the local deployment helper's prepared configuration.
Use this existing contract for that treasury rather than deploying another copy.

```text
HAVNAI_SEPOLIA_RPC_URL=<Sepolia JSON-RPC endpoint>
HAVNAI_HAI_TOKEN_ADDRESS=<Sepolia HAI token address>
HAVNAI_HAI_TREASURY_WALLET=<treasury owner address>
HAVNAI_NODE_CLAIM_CONTRACT=0x37ADe176ac43cd4e37Cd72054028cde7B8653E25
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
