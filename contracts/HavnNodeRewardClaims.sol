// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

interface IERC20BalanceTransfer {
    function balanceOf(address account) external view returns (uint256);
    function transfer(address recipient, uint256 amount) external returns (bool);
}

/// @title HavnNodeRewardClaims
/// @notice Solvent, immutable Merkle claims for HavnAI node operator rewards.
/// @dev The coordinator never signs transactions. A treasury publishes a
///      funded root, then operators (or relayers) submit their own proofs.
contract HavnNodeRewardClaims {
    bytes32 public constant LEAF_DOMAIN = keccak256("HAVNAI_NODE_PAYOUT_CLAIM_V1");

    IERC20BalanceTransfer public immutable token;
    address public owner;
    uint256 public totalOutstanding;

    mapping(uint256 batchId => bytes32 root) public roots;
    mapping(uint256 batchId => uint256 amount) public remaining;
    mapping(uint256 batchId => mapping(uint256 wordIndex => uint256 word)) private claimedBitMap;

    event RootPublished(uint256 indexed batchId, bytes32 indexed root, uint256 totalAmount);
    event RewardClaimed(
        uint256 indexed batchId,
        uint256 indexed index,
        address indexed account,
        uint256 amount
    );
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);
    event SurplusWithdrawn(address indexed recipient, uint256 amount);

    error AlreadyClaimed();
    error BatchAlreadyPublished();
    error BatchNotPublished();
    error InsolventCommitment();
    error InsufficientBatchBalance();
    error InsufficientSurplus();
    error InvalidProof();
    error InvalidValue();
    error NotOwner();
    error TransferFailed();
    error ZeroAddress();

    modifier onlyOwner() {
        if (msg.sender != owner) revert NotOwner();
        _;
    }

    constructor(address tokenAddress, address initialOwner) {
        if (tokenAddress == address(0) || initialOwner == address(0)) revert ZeroAddress();
        token = IERC20BalanceTransfer(tokenAddress);
        owner = initialOwner;
        emit OwnershipTransferred(address(0), initialOwner);
    }

    /// @notice Commit one coordinator-generated payout batch permanently.
    /// @dev The contract must already hold every token promised by all open batches.
    function publishRoot(uint256 batchId, bytes32 root, uint256 totalAmount) external onlyOwner {
        if (batchId == 0 || root == bytes32(0) || totalAmount == 0) revert InvalidValue();
        if (roots[batchId] != bytes32(0)) revert BatchAlreadyPublished();
        if (token.balanceOf(address(this)) < totalOutstanding + totalAmount) {
            revert InsolventCommitment();
        }

        roots[batchId] = root;
        remaining[batchId] = totalAmount;
        totalOutstanding += totalAmount;
        emit RootPublished(batchId, root, totalAmount);
    }

    /// @notice Claim a proof-bound operator reward. Anyone may relay; funds always go to account.
    function claim(
        uint256 batchId,
        uint256 index,
        address account,
        uint256 amount,
        bytes32[] calldata proof
    ) external {
        bytes32 root = roots[batchId];
        if (root == bytes32(0)) revert BatchNotPublished();
        if (account == address(0) || amount == 0) revert InvalidValue();
        if (isClaimed(batchId, index)) revert AlreadyClaimed();

        bytes32 leaf = keccak256(abi.encode(LEAF_DOMAIN, batchId, index, account, amount));
        if (!_verify(proof, root, leaf)) revert InvalidProof();
        if (remaining[batchId] < amount || totalOutstanding < amount) {
            revert InsufficientBatchBalance();
        }

        _setClaimed(batchId, index);
        remaining[batchId] -= amount;
        totalOutstanding -= amount;
        _safeTransfer(account, amount);
        emit RewardClaimed(batchId, index, account, amount);
    }

    function isClaimed(uint256 batchId, uint256 index) public view returns (bool) {
        uint256 wordIndex = index >> 8;
        uint256 bitIndex = index & 255;
        return claimedBitMap[batchId][wordIndex] & (uint256(1) << bitIndex) != 0;
    }

    /// @notice Withdraw only tokens not committed to published payout roots.
    function withdrawSurplus(address recipient, uint256 amount) external onlyOwner {
        if (recipient == address(0)) revert ZeroAddress();
        uint256 balance = token.balanceOf(address(this));
        if (balance < totalOutstanding || amount > balance - totalOutstanding) {
            revert InsufficientSurplus();
        }
        _safeTransfer(recipient, amount);
        emit SurplusWithdrawn(recipient, amount);
    }

    function transferOwnership(address newOwner) external onlyOwner {
        if (newOwner == address(0)) revert ZeroAddress();
        emit OwnershipTransferred(owner, newOwner);
        owner = newOwner;
    }

    function _setClaimed(uint256 batchId, uint256 index) private {
        uint256 wordIndex = index >> 8;
        uint256 bitIndex = index & 255;
        claimedBitMap[batchId][wordIndex] |= uint256(1) << bitIndex;
    }

    function _verify(
        bytes32[] calldata proof,
        bytes32 root,
        bytes32 leaf
    ) private pure returns (bool) {
        bytes32 computed = leaf;
        for (uint256 i = 0; i < proof.length; i++) {
            bytes32 sibling = proof[i];
            computed = computed <= sibling
                ? keccak256(abi.encodePacked(computed, sibling))
                : keccak256(abi.encodePacked(sibling, computed));
        }
        return computed == root;
    }

    function _safeTransfer(address recipient, uint256 amount) private {
        (bool ok, bytes memory result) = address(token).call(
            abi.encodeWithSelector(IERC20BalanceTransfer.transfer.selector, recipient, amount)
        );
        if (!ok || (result.length != 0 && !abi.decode(result, (bool)))) revert TransferFailed();
    }
}
