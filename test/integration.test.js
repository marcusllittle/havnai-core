const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("HavnAI integration", function () {
  it("registers workflow, records execution, and splits payouts", async function () {
    const [deployer, creator, nodeOperator, validator1, validator2, upstream1, upstream2, treasury] =
      await ethers.getSigners();

    const HAI = await ethers.getContractFactory("HAI");
    const hai = await HAI.deploy(deployer.address, deployer.address);
    await hai.waitForDeployment();

    const WorkflowRegistry = await ethers.getContractFactory("WorkflowRegistry");
    const registry = await WorkflowRegistry.deploy(deployer.address);
    await registry.waitForDeployment();

    const ExecutionLog = await ethers.getContractFactory("ExecutionLog");
    const executionLog = await ExecutionLog.deploy(deployer.address, deployer.address, 2);
    await executionLog.waitForDeployment();

    const PayoutSplitter = await ethers.getContractFactory("PayoutSplitter");
    const splitter = await PayoutSplitter.deploy(
      await hai.getAddress(),
      await executionLog.getAddress(),
      await registry.getAddress(),
      treasury.address
    );
    await splitter.waitForDeployment();

    const StakingManager = await ethers.getContractFactory("StakingManager");
    const staking = await StakingManager.deploy(await hai.getAddress(), ethers.parseEther("100"), ethers.parseEther("50"), 3600);
    await staking.waitForDeployment();
    expect(await staking.minNodeStake()).to.equal(ethers.parseEther("100"));

    // Register workflow
    const workflowId = ethers.keccak256(ethers.toUtf8Bytes("demo-workflow"));
    const dependencyInputs = [
      { assetId: ethers.keccak256(ethers.toUtf8Bytes("asset-1")), weightBps: 5000 },
      { assetId: ethers.keccak256(ethers.toUtf8Bytes("asset-2")), weightBps: 5000 }
    ];

    await expect(
      registry.connect(creator).registerWorkflow(
        workflowId,
        "ipfs://workflow",
        {
          creatorBps: 4500,
          upstreamBps: 2000,
          nodeBps: 2500,
          validatorBps: 500,
          treasuryBps: 500
        },
        dependencyInputs
      )
    ).to.emit(registry, "WorkflowRegistered");

    await splitter.setAssetRecipient(dependencyInputs[0].assetId, upstream1.address);
    await splitter.setAssetRecipient(dependencyInputs[1].assetId, upstream2.address);

    await executionLog.setValidator(validator1.address, true);
    await executionLog.setValidator(validator2.address, true);

    // Record execution
    const jobId = ethers.keccak256(ethers.toUtf8Bytes("job-1"));
    const approvals = [
      { validator: validator1.address, signature: ethers.toUtf8Bytes("sig1") },
      { validator: validator2.address, signature: ethers.toUtf8Bytes("sig2") }
    ];

    await expect(
      executionLog.recordExecution(
        jobId,
        workflowId,
        deployer.address,
        nodeOperator.address,
        1000n * 10n ** 18n,
        ethers.keccak256(ethers.toUtf8Bytes("receipt")),
        approvals
      )
    ).to.emit(executionLog, "ExecutionRecorded");

    const totalFee = ethers.parseEther("1000");
    await hai.mint(await splitter.getAddress(), totalFee);

    await expect(
      splitter.settlePayout(jobId, nodeOperator.address, [validator1.address, validator2.address], [], [])
    ).to.emit(splitter, "PayoutSettled");

    const creatorShare = totalFee * 4500n / 10000n;
    const upstreamShare = totalFee * 2000n / 10000n / 2n;
    const nodeShare = totalFee * 2500n / 10000n;
    const validatorShare = totalFee * 500n / 10000n / 2n;
    const treasuryShare = totalFee * 500n / 10000n;

    expect(await hai.balanceOf(creator.address)).to.equal(creatorShare);
    expect(await hai.balanceOf(upstream1.address)).to.equal(upstreamShare);
    expect(await hai.balanceOf(upstream2.address)).to.equal(upstreamShare);
    expect(await hai.balanceOf(nodeOperator.address)).to.equal(nodeShare);
    expect(await hai.balanceOf(validator1.address)).to.equal(validatorShare);
    expect(await hai.balanceOf(validator2.address)).to.equal(validatorShare);
    expect(await hai.balanceOf(treasury.address)).to.equal(treasuryShare);
    expect(await splitter.jobSettled(jobId)).to.equal(true);
  });
});
