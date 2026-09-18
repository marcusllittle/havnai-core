// Local EVM only; never sends a Sepolia transaction.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const solc = require('solc');
const ganache = require('ganache');
const { BrowserProvider, ContractFactory, AbiCoder, keccak256, id, parseEther, ZeroAddress } = require('ethers');
(async () => {
  const artifact = JSON.parse(fs.readFileSync(path.join(process.argv[2], 'artifact.json')));
  const input = { language: 'Solidity', sources: { 'MockHAI.sol': { content: fs.readFileSync(path.join(__dirname, '../test/MockHAI.sol'), 'utf8') } }, settings: { evmVersion: 'shanghai', outputSelection: { '*': { '*': ['abi', 'evm.bytecode.object'] } } } };
  const mock = JSON.parse(solc.compile(JSON.stringify(input))).contracts['MockHAI.sol'].MockHAI;
  const local = ganache.provider({ logging: { quiet: true }, chain: { hardfork: 'shanghai' } });
  const provider = new BrowserProvider(local);
  try {
    const owner = await provider.getSigner(0), operator = await provider.getSigner(1);
    const token = await new ContractFactory(mock.abi, '0x' + mock.evm.bytecode.object, owner).deploy();
    await token.waitForDeployment();
    const factory = new ContractFactory(artifact.abi, artifact.bytecode, owner);
    await assert.rejects(factory.deploy(ZeroAddress, await owner.getAddress()));
    const claims = await factory.deploy(await token.getAddress(), await owner.getAddress());
    await claims.waitForDeployment();
    assert.equal(await claims.owner(), await owner.getAddress());
    assert.equal(await claims.token(), await token.getAddress());
    assert.equal(await claims.LEAF_DOMAIN(), id('HAVNAI_NODE_PAYOUT_CLAIM_V1'));
    const amount = parseEther('10');
    const root = keccak256(AbiCoder.defaultAbiCoder().encode(['bytes32','uint256','uint256','address','uint256'], [await claims.LEAF_DOMAIN(), 1, 0, await operator.getAddress(), amount]));
    await assert.rejects(claims.publishRoot.staticCall(1, root, amount));
    await (await token.mint(await claims.getAddress(), amount)).wait();
    await assert.rejects(claims.connect(operator).publishRoot.staticCall(1, root, amount));
    await (await claims.publishRoot(1, root, amount)).wait();
    await assert.rejects(claims.publishRoot.staticCall(1, root, amount));
    await assert.rejects(claims.withdrawSurplus.staticCall(await owner.getAddress(), 1));
    await assert.rejects(claims.claim.staticCall(1, 0, await owner.getAddress(), amount, []));
    await (await claims.connect(operator).claim(1, 0, await operator.getAddress(), amount, [])).wait();
    assert.equal(await token.balanceOf(await operator.getAddress()), amount);
    assert.equal(await claims.totalOutstanding(), 0n);
    assert.equal(await claims.isClaimed(1, 0), true);
    await assert.rejects(claims.claim.staticCall(1, 0, await operator.getAddress(), amount, []));
    console.log('PASS: deployment, constructor checks, owner/token/domain, funding requirement, owner authorization, immutable root, reserved funds, proof binding, payout and duplicate-claim rejection.');
  } finally { provider.destroy(); await local.disconnect(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
