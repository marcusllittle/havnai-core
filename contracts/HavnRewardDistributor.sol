// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @notice Minimal interface for the HAI ERC-20 token.
interface IERC20 {
    function transfer(address to, uint256 amount) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
}

/// @title HavnRewardDistributor
/// @notice Owner-controlled contract that distributes HAI rewards to
///         node operators and game players in single or batch transfers.
///
/// Deployment flow:
///   1. Deploy with the HAI token address.
///   2. Transfer the funding HAI balance to the contract.
///   3. Set HAVNAI_DISTRIBUTOR_ADDRESS in havnai-core .env.
///   4. The payout_worker will call distribute() / batchDistribute()
///      via the owner wallet (HAVNAI_PAYER_KEY).
contract HavnRewardDistributor {
    address public owner;
    IERC20 public immutable token;

    event RewardSent(address indexed recipient, uint256 amount);
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);
    event FundsWithdrawn(address indexed recipient, uint256 amount);

    error NotOwner();
    error ZeroAddress();
    error LengthMismatch();
    error BatchTooLarge();
    error TransferFailed();

    modifier onlyOwner() {
        if (msg.sender != owner) revert NotOwner();
        _;
    }

    constructor(address _token) {
        if (_token == address(0)) revert ZeroAddress();
        owner = msg.sender;
        token = IERC20(_token);
    }

    /// @notice Transfer ownership to a new address.
    function transferOwnership(address newOwner) external onlyOwner {
        if (newOwner == address(0)) revert ZeroAddress();
        emit OwnershipTransferred(owner, newOwner);
        owner = newOwner;
    }

    /// @notice Send HAI to a single recipient.
    function distribute(address recipient, uint256 amount) external onlyOwner {
        if (!token.transfer(recipient, amount)) revert TransferFailed();
        emit RewardSent(recipient, amount);
    }

    /// @notice Send HAI to multiple recipients in one transaction.
    ///         Maximum batch size is 200 to stay within block gas limits.
    function batchDistribute(
        address[] calldata recipients,
        uint256[] calldata amounts
    ) external onlyOwner {
        if (recipients.length != amounts.length) revert LengthMismatch();
        if (recipients.length > 200) revert BatchTooLarge();
        for (uint256 i = 0; i < recipients.length; i++) {
            if (!token.transfer(recipients[i], amounts[i])) revert TransferFailed();
            emit RewardSent(recipients[i], amounts[i]);
        }
    }

    /// @notice Emergency withdrawal of HAI back to the owner.
    function withdraw(uint256 amount) external onlyOwner {
        if (!token.transfer(owner, amount)) revert TransferFailed();
        emit FundsWithdrawn(owner, amount);
    }

    /// @notice Current HAI balance held by this contract.
    function balance() external view returns (uint256) {
        return token.balanceOf(address(this));
    }
}
