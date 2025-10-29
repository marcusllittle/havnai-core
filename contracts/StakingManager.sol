// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {HAI} from "./HAI.sol";

/// @title StakingManager
/// @notice Bonds and slashes stake for HavnAI nodes and validators.
contract StakingManager {
    /// @notice $HAI token reference used for staking.
    HAI public immutable hai;

    /// @notice Governance authority.
    address public governance;
    /// @notice Minimum stake required for node operators.
    uint256 public minNodeStake;
    /// @notice Minimum stake required for validators.
    uint256 public minValidatorStake;
    /// @notice Time required before stake withdrawals become available.
    uint256 public unbondingPeriod;

    /// @notice Role enumeration for stakers.
    enum Role {
        None,
        Node,
        Validator
    }

    /// @notice State snapshot for each staker.
    struct StakeInfo {
        uint256 amount;
        uint256 unlockTimestamp;
        Role role;
        bool active;
    }

    mapping(address => StakeInfo) public stakes;

    event GovernanceUpdated(address indexed newGovernance);
    event StakeBonded(address indexed operator, Role indexed role, uint256 newAmount);
    event StakeUnbondRequested(address indexed operator, uint256 unlockTimestamp);
    event StakeWithdrawn(address indexed operator, uint256 amount);
    event StakeSlashed(address indexed operator, uint256 penalty, address indexed recipient);
    event ParametersUpdated(uint256 minNodeStake, uint256 minValidatorStake, uint256 unbondingPeriod);

    error Unauthorized();
    error InvalidRole();
    error CooldownActive();
    error InsufficientStake();

    constructor(HAI haiToken, uint256 nodeStake, uint256 validatorStake, uint256 cooldown) {
        require(address(haiToken) != address(0), "Staking: HAI zero");
        hai = haiToken;
        governance = msg.sender;
        minNodeStake = nodeStake;
        minValidatorStake = validatorStake;
        unbondingPeriod = cooldown;
    }

    /// @notice Bonds stake for a given role.
    function bondStake(uint256 amount, Role role) external {
        if (role != Role.Node && role != Role.Validator) revert InvalidRole();
        if (amount == 0) revert InsufficientStake();

        StakeInfo storage info = stakes[msg.sender];
        if (!info.active) {
            info.role = role;
            info.active = true;
        } else if (info.role != role) {
            revert InvalidRole();
        }

        uint256 newAmount = info.amount + amount;
        info.amount = newAmount;

        uint256 requiredStake = role == Role.Node ? minNodeStake : minValidatorStake;
        if (newAmount < requiredStake) revert InsufficientStake();

        bool ok = hai.transferFrom(msg.sender, address(this), amount);
        require(ok, "Staking: transfer failed");
        emit StakeBonded(msg.sender, role, newAmount);
    }

    /// @notice Initiates unbonding; stake becomes withdrawable after cooldown.
    function requestUnbond() external {
        StakeInfo storage info = stakes[msg.sender];
        if (!info.active) revert InsufficientStake();
        info.active = false;
        info.unlockTimestamp = block.timestamp + unbondingPeriod;
        emit StakeUnbondRequested(msg.sender, info.unlockTimestamp);
    }

    /// @notice Withdraws stake after cooldown has elapsed.
    function withdrawStake() external {
        StakeInfo storage info = stakes[msg.sender];
        if (info.active) revert CooldownActive();
        if (info.amount == 0) revert InsufficientStake();
        if (block.timestamp < info.unlockTimestamp) revert CooldownActive();

        uint256 amount = info.amount;
        info.amount = 0;
        info.role = Role.None;
        info.unlockTimestamp = 0;

        bool ok = hai.transfer(msg.sender, amount);
        require(ok, "Staking: withdraw failed");
        emit StakeWithdrawn(msg.sender, amount);
    }

    /// @notice Slashes stake for misbehavior by sending penalty to recipient.
    function slash(address operator, uint256 penalty, address recipient) external {
        if (msg.sender != governance) revert Unauthorized();
        StakeInfo storage info = stakes[operator];
        if (info.amount < penalty) revert InsufficientStake();
        if (recipient == address(0)) revert Unauthorized();

        info.amount -= penalty;
        bool ok = hai.transfer(recipient, penalty);
        require(ok, "Staking: slash transfer failed");

        emit StakeSlashed(operator, penalty, recipient);
    }

    /// @notice Updates system parameters.
    function setParameters(uint256 nodeStake, uint256 validatorStake, uint256 cooldown) external {
        if (msg.sender != governance) revert Unauthorized();
        minNodeStake = nodeStake;
        minValidatorStake = validatorStake;
        unbondingPeriod = cooldown;
        emit ParametersUpdated(nodeStake, validatorStake, cooldown);
    }

    /// @notice Transfers governance role.
    function setGovernance(address newGovernance) external {
        if (msg.sender != governance) revert Unauthorized();
        require(newGovernance != address(0), "Staking: zero governance");
        governance = newGovernance;
        emit GovernanceUpdated(newGovernance);
    }
}
