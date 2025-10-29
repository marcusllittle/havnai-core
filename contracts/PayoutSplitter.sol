// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {ExecutionLog} from "./ExecutionLog.sol";
import {WorkflowRegistry} from "./WorkflowRegistry.sol";
import {HAI} from "./HAI.sol";

/// @title PayoutSplitter
/// @notice Splits $HAI fees between creators, contributors, validators, and treasury per workflow configuration.
contract PayoutSplitter {
    /// @notice Basis points denominator shared with registry.
    uint16 public constant BPS = 10_000;

    /// @notice Stable reference to $HAI token.
    HAI public immutable hai;
    /// @notice Reference to execution log.
    ExecutionLog public immutable executionLog;
    /// @notice Reference to workflow registry.
    WorkflowRegistry public immutable registry;

    /// @notice Governance controller for treasury and overrides.
    address public governance;
    /// @notice Treasury address receiving protocol share.
    address public treasury;

    /// @notice Tracks whether a job has been settled.
    mapping(bytes32 => bool) public jobSettled;
    /// @notice Maps upstream asset identifiers to recipient wallets.
    mapping(bytes32 => address) public assetRecipient;

    event GovernanceUpdated(address indexed newGovernance);
    event TreasuryUpdated(address indexed newTreasury);
    event AssetRecipientUpdated(bytes32 indexed assetId, address indexed recipient);
    event PayoutSettled(bytes32 indexed jobId, bytes32 indexed workflowId, uint256 totalFee);

    error AlreadySettled();
    error Unauthorized();
    error InvalidOverrides();
    error MissingRecipient();

    constructor(HAI haiToken, ExecutionLog log, WorkflowRegistry reg, address initialTreasury) {
        require(address(haiToken) != address(0), "Splitter: HAI zero");
        hai = haiToken;
        executionLog = log;
        registry = reg;
        governance = msg.sender;
        treasury = initialTreasury;
    }

    /// @notice Settles payouts for a validated job.
    /// @dev Assumes this contract already holds the fee amount in $HAI.
    function settlePayout(
        bytes32 jobId,
        address nodeRecipient,
        address[] calldata validatorRecipients,
        bytes32[] calldata overrideAssetIds,
        address[] calldata overrideRecipients
    ) external {
        if (jobSettled[jobId]) revert AlreadySettled();
        ExecutionLog.ExecutionRecord memory record = executionLog.getExecution(jobId);
        jobSettled[jobId] = true;

        (WorkflowRegistry.Workflow memory wf, WorkflowRegistry.Dependency[] memory deps) = registry.getWorkflow(
            record.workflowId
        );
        require(!wf.flagged, "Splitter: workflow flagged");

        WorkflowRegistry.RoyaltyRoute memory route = wf.route;
        uint256 totalFee = record.fee;
        require(hai.balanceOf(address(this)) >= totalFee, "Splitter: insufficient balance");

        _transferOrRevert(wf.creator, (totalFee * route.creatorBps) / BPS);

        uint256 nodeShare = (totalFee * route.nodeBps) / BPS;
        if (nodeShare > 0) {
            _transferOrRevert(nodeRecipient, nodeShare);
        }

        _distributeUpstream(
            deps,
            (totalFee * route.upstreamBps) / BPS,
            overrideAssetIds,
            overrideRecipients
        );

        _distributeValidators(validatorRecipients, (totalFee * route.validatorBps) / BPS);

        uint256 treasuryShare = (totalFee * route.treasuryBps) / BPS;
        if (treasuryShare > 0 && treasury != address(0)) {
            _transferOrRevert(treasury, treasuryShare);
        }

        emit PayoutSettled(jobId, record.workflowId, totalFee);
    }

    /// @notice Updates governance authority.
    function setGovernance(address newGovernance) external {
        if (msg.sender != governance) revert Unauthorized();
        require(newGovernance != address(0), "Splitter: zero governance");
        governance = newGovernance;
        emit GovernanceUpdated(newGovernance);
    }

    /// @notice Sets the treasury address for protocol fees.
    function setTreasury(address newTreasury) external {
        if (msg.sender != governance) revert Unauthorized();
        treasury = newTreasury;
        emit TreasuryUpdated(newTreasury);
    }

    /// @notice Registers a wallet for a dependency asset.
    function setAssetRecipient(bytes32 assetId, address recipient) external {
        if (msg.sender != governance) revert Unauthorized();
        assetRecipient[assetId] = recipient;
        emit AssetRecipientUpdated(assetId, recipient);
    }

    function _distributeUpstream(
        WorkflowRegistry.Dependency[] memory deps,
        uint256 upstreamPool,
        bytes32[] calldata overrideAssetIds,
        address[] calldata overrideRecipients
    ) internal {
        if (upstreamPool == 0 || deps.length == 0) {
            return;
        }
        if (overrideAssetIds.length != overrideRecipients.length) revert InvalidOverrides();

        for (uint256 i = 0; i < deps.length; i++) {
            WorkflowRegistry.Dependency memory dep = deps[i];
            address recipient = assetRecipient[dep.assetId];
            for (uint256 j = 0; j < overrideAssetIds.length; j++) {
                if (overrideAssetIds[j] == dep.assetId) {
                    recipient = overrideRecipients[j];
                    break;
                }
            }
            if (recipient == address(0)) revert MissingRecipient();
            uint256 share = (upstreamPool * dep.weightBps) / BPS;
            if (share > 0) {
                _transferOrRevert(recipient, share);
            }
        }
    }

    function _distributeValidators(address[] calldata recipients, uint256 pool) internal {
        if (pool == 0 || recipients.length == 0) {
            return;
        }
        uint256 share = pool / recipients.length;
        require(share > 0, "Splitter: validator share zero");
        for (uint256 i = 0; i < recipients.length; i++) {
            _transferOrRevert(recipients[i], share);
        }
    }

    function _transferOrRevert(address to, uint256 amount) internal {
        if (amount == 0) return;
        require(to != address(0), "Splitter: transfer zero");
        bool ok = hai.transfer(to, amount);
        require(ok, "Splitter: transfer failed");
    }
}
