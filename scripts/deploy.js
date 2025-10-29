const hre = require("hardhat");

async function main() {
  const [deployer] = await hre.ethers.getSigners();
  console.log(`Deploying contracts with ${deployer.address}`);

  const HAI = await hre.ethers.getContractFactory("HAI");
  const hai = await HAI.deploy(deployer.address, deployer.address);
  await hai.waitForDeployment();
  console.log(`HAI deployed at ${await hai.getAddress()}`);

  const WorkflowRegistry = await hre.ethers.getContractFactory("WorkflowRegistry");
  const registry = await WorkflowRegistry.deploy(deployer.address);
  await registry.waitForDeployment();
  console.log(`WorkflowRegistry deployed at ${await registry.getAddress()}`);

  const ExecutionLog = await hre.ethers.getContractFactory("ExecutionLog");
  const executionLog = await ExecutionLog.deploy(deployer.address, deployer.address, 2);
  await executionLog.waitForDeployment();
  console.log(`ExecutionLog deployed at ${await executionLog.getAddress()}`);

  const PayoutSplitter = await hre.ethers.getContractFactory("PayoutSplitter");
  const splitter = await PayoutSplitter.deploy(
    await hai.getAddress(),
    await executionLog.getAddress(),
    await registry.getAddress(),
    deployer.address
  );
  await splitter.waitForDeployment();
  console.log(`PayoutSplitter deployed at ${await splitter.getAddress()}`);

  const StakingManager = await hre.ethers.getContractFactory("StakingManager");
  const staking = await StakingManager.deploy(
    await hai.getAddress(),
    hre.ethers.parseEther("100"),
    hre.ethers.parseEther("50"),
    3600
  );
  await staking.waitForDeployment();
  console.log(`StakingManager deployed at ${await staking.getAddress()}`);

  console.log("Deployment complete.");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
