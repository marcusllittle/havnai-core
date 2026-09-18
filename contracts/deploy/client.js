/* global ethers */
(async () => {
  'use strict';
  const $ = id => document.getElementById(id);
  const artifact = await (await fetch('/artifact.json')).json();
  const key = 'havnai-rewards-deployment:' + artifact.chainId + ':' + artifact.owner + ':' + artifact.sourceSha256;
  let injected, provider, ready = false, busy = false, record;
  const announced = new Map();
  window.addEventListener('eip6963:announceProvider', event => {
    if (event.detail?.info?.rdns === 'io.metamask') announced.set(event.detail.info.uuid, event.detail.provider);
  });
  window.dispatchEvent(new Event('eip6963:requestProvider'));
  $('token').textContent = artifact.token;
  $('owner').textContent = artifact.owner;
  function status(message, error = false) { $('status').textContent = message; $('status').className = error ? 'error' : ''; }
  function buttons() {
    $('connect').disabled = busy;
    $('deploy').disabled = busy || !ready || !!record;
    $('check').disabled = busy || !provider;
  }
  function save(value) { localStorage.setItem(key, JSON.stringify(value)); record = value; $('hash').value = value.transactionHash || ''; }
  try { record = JSON.parse(localStorage.getItem(key) || 'null'); } catch { status('Browser storage is unavailable. Enable local storage before deploying.', true); return; }
  if (record) { $('hash').value = record.transactionHash || ''; status('A deployment is saved. Connect MetaMask, then check its transaction. A second deployment is disabled.'); }
  else status('Connect the treasury account in MetaMask to prepare the deployment.');
  function selectMetaMask() {
    const announcedProvider = [...announced.values()][0];
    if (announcedProvider) return announcedProvider;
    const candidates = window.ethereum?.providers || [window.ethereum];
    const found = candidates.find(item => item?.isMetaMask && !item.isBraveWallet && !item.isRabby);
    if (!found) throw new Error('Open this page in the browser where your MetaMask extension is installed.');
    return found;
  }
  async function assertWallet() {
    const chain = await injected.request({ method: 'eth_chainId' });
    if (BigInt(chain) !== BigInt(artifact.chainId)) throw new Error('Select Sepolia in MetaMask, then reconnect.');
    const accounts = await injected.request({ method: 'eth_accounts' });
    if (accounts[0]?.toLowerCase() !== artifact.owner) throw new Error('Select the treasury account ' + artifact.owner + ' in MetaMask.');
  }
  const factory = new ethers.ContractFactory(artifact.abi, artifact.bytecode);
  const transaction = await factory.getDeployTransaction(artifact.token, artifact.owner);
  async function run(action) {
    if (busy) return;
    busy = true; buttons();
    try { await action(); } catch (error) { status(error.shortMessage || error.message || String(error), true); }
    finally { busy = false; buttons(); }
  }
  $('connect').onclick = () => run(async () => {
    ready = false;
    injected = selectMetaMask();
    status('Approve the account connection in MetaMask.');
    await injected.request({ method: 'eth_requestAccounts' });
    if (BigInt(await injected.request({ method: 'eth_chainId' })) !== BigInt(artifact.chainId)) {
      await injected.request({ method: 'wallet_switchEthereumChain', params: [{ chainId: '0xaa36a7' }] });
    }
    await assertWallet();
    provider = new ethers.BrowserProvider(injected);
    const token = new ethers.Contract(artifact.token, ['function symbol() view returns(string)', 'function decimals() view returns(uint8)'], provider);
    if (await provider.getCode(artifact.token) === '0x' || await token.symbol() !== 'HAI' || await token.decimals() !== 18n) throw new Error('The configured Sepolia HAI token did not pass verification.');
    const gas = await provider.estimateGas({ ...transaction, from: artifact.owner });
    const fees = await provider.getFeeData();
    const fee = gas * (fees.maxFeePerGas || fees.gasPrice || 0n);
    $('cost').textContent = 'Estimated ' + ethers.formatEther(fee) + ' Sepolia ETH (' + gas + ' gas). MetaMask shows the final fee.';
    if (await provider.getBalance(artifact.owner) < fee) throw new Error('The treasury needs more Sepolia ETH for deployment gas.');
    localStorage.setItem(key + ':storage-check', 'ok'); localStorage.removeItem(key + ':storage-check');
    ready = true;
    status(record ? 'Connected. Check the saved deployment transaction below.' : 'Ready. Deploy on Sepolia opens one contract-creation transaction for you to review in MetaMask.');
    injected.on?.('accountsChanged', () => { ready = false; buttons(); });
    injected.on?.('chainChanged', () => { ready = false; buttons(); });
  });
  async function verify(hash) {
    if (!/^0x[0-9a-f]{64}$/i.test(hash)) throw new Error('Enter a complete transaction hash.');
    if (BigInt(await injected.request({ method: 'eth_chainId' })) !== BigInt(artifact.chainId)) throw new Error('Switch MetaMask to Sepolia before checking.');
    const tx = await provider.getTransaction(hash);
    if (!tx) throw new Error('Transaction not found yet. Check again shortly.');
    if (tx.to !== null || tx.from.toLowerCase() !== artifact.owner || tx.data.toLowerCase() !== transaction.data.toLowerCase() || tx.value !== 0n) throw new Error('This transaction does not match the prepared rewards deployment.');
    save({ transactionHash: hash, status: 'pending' });
    const receipt = await provider.getTransactionReceipt(hash);
    if (!receipt) { status('Deployment pending. Keep this hash: ' + hash + '\nUse Check transaction after it confirms.'); return; }
    if (receipt.status !== 1 || !receipt.contractAddress) throw new Error('Deployment reverted. Send this transaction hash to Codex before trying again: ' + hash);
    const address = receipt.contractAddress;
    let actual = (await provider.getCode(address)).slice(2).toLowerCase();
    let expected = artifact.runtime.toLowerCase();
    for (const references of Object.values(artifact.immutableReferences)) for (const ref of references) {
      const start = ref.start * 2, end = start + ref.length * 2;
      actual = actual.slice(0, start) + '0'.repeat(ref.length * 2) + actual.slice(end);
      expected = expected.slice(0, start) + '0'.repeat(ref.length * 2) + expected.slice(end);
    }
    if (actual !== expected) throw new Error('Deployed bytecode does not match the compiled source.');
    const contract = new ethers.Contract(address, artifact.abi, provider);
    if ((await contract.owner()).toLowerCase() !== artifact.owner || (await contract.token()).toLowerCase() !== artifact.token || await contract.LEAF_DOMAIN() !== ethers.id('HAVNAI_NODE_PAYOUT_CLAIM_V1')) throw new Error('Contract configuration verification failed.');
    if (await receipt.confirmations() < 2) { status('Contract verified at ' + address + '. Waiting for a second confirmation; check again shortly.'); return; }
    save({ transactionHash: hash, status: 'verified', contractAddress: address, chainId: artifact.chainId, token: artifact.token, owner: artifact.owner, compiler: artifact.compiler, sourceSha256: artifact.sourceSha256, blockNumber: receipt.blockNumber });
    $('result').textContent = 'Contract verified on Sepolia:\n' + address + '\n\nCoordinator setting:\nHAVNAI_NODE_CLAIM_CONTRACT=' + address + '\n\nSend this address or the deployment record to Codex to finish setup.';
    $('download').disabled = false;
    status('Deployment confirmed and verified. No HAI has been transferred by this page.');
  }
  $('deploy').onclick = () => run(async () => {
    if (!ready || record) throw new Error('Connect and check the existing deployment before continuing.');
    await assertWallet();
    // Persist intent before opening the wallet. An interrupted response must not enable a duplicate deployment.
    save({ status: 'awaiting_wallet', transactionHash: '' });
    status('Review and confirm the contract deployment in MetaMask.');
    let hash;
    try { hash = await injected.request({ method: 'eth_sendTransaction', params: [{ from: artifact.owner, data: transaction.data, value: '0x0', chainId: '0xaa36a7' }] }); }
    catch (error) {
      if (error.code === 4001) { localStorage.removeItem(key); record = null; }
      throw error;
    }
    save({ status: 'pending', transactionHash: hash });
    status('Transaction submitted: ' + hash + '\nCheck transaction below after confirmation.');
    await verify(hash);
  });
  $('check').onclick = () => run(() => verify($('hash').value.trim()));
  $('download').onclick = () => {
    const url = URL.createObjectURL(new Blob([JSON.stringify(record, null, 2)], { type: 'application/json' }));
    const link = document.createElement('a'); link.href = url; link.download = 'havnai-rewards-sepolia-deployment.json'; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  buttons();
})().catch(error => { document.getElementById('status').textContent = 'Unable to prepare deployment: ' + error.message; });
