// NODE_PATH must contain solc@0.8.30 and ethers@6.17.0.
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const solc = require('solc');
const out = path.resolve(process.argv[2] || 'reward-deployment');
if (!solc.version().startsWith('0.8.30+')) throw new Error('Use solc 0.8.30');
const source = fs.readFileSync(path.join(__dirname, '../HavnNodeRewardClaims.sol'), 'utf8');
const input = { language: 'Solidity', sources: { 'HavnNodeRewardClaims.sol': { content: source } }, settings: {
  optimizer: { enabled: true, runs: 200 }, evmVersion: 'shanghai',
  outputSelection: { '*': { '*': ['abi', 'evm.bytecode.object', 'evm.deployedBytecode'] } }
} };
const result = JSON.parse(solc.compile(JSON.stringify(input)));
for (const error of result.errors || []) {
  if (error.severity === 'error') throw new Error(error.formattedMessage);
}
const contract = result.contracts['HavnNodeRewardClaims.sol'].HavnNodeRewardClaims;
const artifact = {
  name: 'HavnNodeRewardClaims', compiler: solc.version(),
  sourceSha256: crypto.createHash('sha256').update(source).digest('hex'),
  chainId: 11155111,
  token: '0x8aaa7f1075de91b2103542379bfd3b889c89105f',
  owner: '0x7110347e2bcd02f5f3485dc6bec5e0b5f9eb9262',
  abi: contract.abi, bytecode: '0x' + contract.evm.bytecode.object,
  runtime: contract.evm.deployedBytecode.object,
  immutableReferences: contract.evm.deployedBytecode.immutableReferences
};
fs.mkdirSync(out, { recursive: true });
fs.writeFileSync(path.join(out, 'artifact.json'), JSON.stringify(artifact, null, 2));
fs.writeFileSync(path.join(out, 'compiler-input.json'), JSON.stringify(input, null, 2));
fs.writeFileSync(path.join(out, 'HavnNodeRewardClaims.sol'), source);
const ethersRoot = path.dirname(path.dirname(require.resolve('ethers')));
fs.copyFileSync(path.join(ethersRoot, 'dist/ethers.umd.min.js'), path.join(out, 'ethers.js'));
for (const file of ['index.html', 'client.js', 'serve.cjs']) fs.copyFileSync(path.join(__dirname, file), path.join(out, file));
console.log(JSON.stringify({ output: out, compiler: artifact.compiler, sourceSha256: artifact.sourceSha256, bytecodeBytes: (artifact.bytecode.length - 2) / 2 }));
