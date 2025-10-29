// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title ExecutionLog
/// @notice Canonical ledger of workflow executions attested by validators.
contract ExecutionLog {
    /// @notice Address empowered to manage validator roster and quorum.
    address public governance;
    /// @notice Address permitted to submit batched validator attestations.
    address public attestationAggregator;
    /// @notice Number of validator approvals required to accept an execution.
    uint16 public quorumSize;

    /// @notice Metadata for each recorded execution.
    struct ExecutionRecord {
        bytes32 workflowId;
        address caller;
        address node;
        uint96 fee;
        uint64 timestamp;
        uint16 validatorCount;
        bytes32 receiptHash;
    }

    /// @notice Simplified validator approval referenced for auditability.
    struct ValidatorApproval {
        address validator;
        bytes signature;
    }

    mapping(bytes32 => ExecutionRecord) private _records;
    mapping(address => bool) public validators;

    event ExecutionRecorded(bytes32 indexed jobId, bytes32 indexed workflowId, address indexed node, uint96 fee);
    event GovernanceUpdated(address indexed newGovernance);
    event AggregatorUpdated(address indexed newAggregator);
    event ValidatorSet(address indexed validator, bool active);
    event QuorumChanged(uint16 newQuorum);

    error Unauthorized();
    error AlreadyRecorded();
    error InvalidQuorum();

    modifier onlyGovernance() {
        if (msg.sender != governance) revert Unauthorized();
        _;
    }

    constructor(address initialGovernance, address initialAggregator, uint16 initialQuorum) {
        governance = initialGovernance == address(0) ? msg.sender : initialGovernance;
        attestationAggregator = initialAggregator;
        quorumSize = initialQuorum == 0 ? 1 : initialQuorum;
    }

    /// @notice Records an execution after validator quorum is achieved.
    function recordExecution(
        bytes32 jobId,
        bytes32 workflowId,
        address caller,
        address node,
        uint96 fee,
        bytes32 receiptHash,
        ValidatorApproval[] calldata approvals
    ) external {
        if (msg.sender != attestationAggregator) revert Unauthorized();
        if (_records[jobId].timestamp != 0) revert AlreadyRecorded();
        if (approvals.length < quorumSize) revert InvalidQuorum();

        for (uint256 i = 0; i < approvals.length; i++) {
            ValidatorApproval calldata approval = approvals[i];
            if (!validators[approval.validator]) revert Unauthorized();
            if (approval.signature.length == 0) revert InvalidQuorum();
        }

        _records[jobId] = ExecutionRecord({
            workflowId: workflowId,
            caller: caller,
            node: node,
            fee: fee,
            timestamp: uint64(block.timestamp),
            validatorCount: uint16(approvals.length),
            receiptHash: receiptHash
        });

        emit ExecutionRecorded(jobId, workflowId, node, fee);
    }

    /// @notice Fetches execution record for downstream modules.
    function getExecution(bytes32 jobId) external view returns (ExecutionRecord memory) {
        ExecutionRecord memory record = _records[jobId];
        require(record.timestamp != 0, "ExecutionLog: unknown job");
        return record;
    }

    /// @notice Enables or disables a validator.
    function setValidator(address validator, bool active) external onlyGovernance {
        validators[validator] = active;
        emit ValidatorSet(validator, active);
    }

    /// @notice Updates quorum requirement.
    function setQuorumSize(uint16 newQuorum) external onlyGovernance {
        quorumSize = newQuorum;
        emit QuorumChanged(newQuorum);
    }

    /// @notice Updates attestation aggregator.
    function setAttestationAggregator(address newAggregator) external onlyGovernance {
        attestationAggregator = newAggregator;
        emit AggregatorUpdated(newAggregator);
    }

    /// @notice Transfers governance.
    function setGovernance(address newGovernance) external onlyGovernance {
        require(newGovernance != address(0), "ExecutionLog: zero governance");
        governance = newGovernance;
        emit GovernanceUpdated(newGovernance);
    }
}
