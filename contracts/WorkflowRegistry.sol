// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title WorkflowRegistry
/// @notice Manages workflow manifests, dependency graphs, and royalty splits for the HavnAI protocol.
contract WorkflowRegistry {
    /// @notice Basis points denominator used for royalty splits.
    uint16 public constant BPS = 10_000;

    /// @notice Address authorized to perform governance actions.
    address public governance;

    /// @notice Metadata representing a registered workflow manifest.
    struct Workflow {
        address creator;
        string metadataURI;
        RoyaltyRoute route;
        bool flagged;
        uint64 createdAt;
    }

    /// @notice Royalty configuration used by PayoutSplitter.
    struct RoyaltyRoute {
        uint16 creatorBps;
        uint16 upstreamBps;
        uint16 nodeBps;
        uint16 validatorBps;
        uint16 treasuryBps;
    }

    /// @notice Describes an upstream dependency receiving royalties.
    struct Dependency {
        bytes32 assetId;
        uint16 weightBps;
    }

    /// @notice Input helper for dependency registration.
    struct DependencyInput {
        bytes32 assetId;
        uint16 weightBps;
    }

    mapping(bytes32 => Workflow) private _workflows;
    mapping(bytes32 => Dependency[]) private _dependencies;
    mapping(address => bool) public attestors;

    event WorkflowRegistered(bytes32 indexed workflowId, address indexed creator, string metadataURI);
    event WorkflowUpdated(bytes32 indexed workflowId);
    event WorkflowFlagged(bytes32 indexed workflowId, bool flagged, address indexed attestor);
    event GovernanceUpdated(address indexed newGovernance);
    event AttestorUpdated(address indexed attestor, bool approved);

    error Unauthorized();
    error InvalidSplit();
    error WorkflowExists();
    error UnknownWorkflow();

    modifier onlyGovernance() {
        if (msg.sender != governance) revert Unauthorized();
        _;
    }

    constructor(address initialGovernance) {
        governance = initialGovernance == address(0) ? msg.sender : initialGovernance;
    }

    /// @notice Registers a new workflow manifest with royalty route and dependencies.
    function registerWorkflow(
        bytes32 workflowId,
        string calldata metadataURI,
        RoyaltyRoute calldata route,
        DependencyInput[] calldata dependencyInputs
    ) external {
        Workflow storage wf = _workflows[workflowId];
        if (wf.creator != address(0)) revert WorkflowExists();
        if (!_validRoute(route)) revert InvalidSplit();

        wf.creator = msg.sender;
        wf.metadataURI = metadataURI;
        wf.route = route;
        wf.createdAt = uint64(block.timestamp);

        _setDependencies(workflowId, dependencyInputs);
        emit WorkflowRegistered(workflowId, msg.sender, metadataURI);
    }

    /// @notice Updates the royalty route for a workflow. Callable by creator or governance.
    function updateRoyaltyRoute(bytes32 workflowId, RoyaltyRoute calldata route) external {
        if (!_validRoute(route)) revert InvalidSplit();
        Workflow storage wf = _workflows[workflowId];
        if (wf.creator == address(0)) revert UnknownWorkflow();
        if (msg.sender != wf.creator && msg.sender != governance) revert Unauthorized();

        wf.route = route;
        emit WorkflowUpdated(workflowId);
    }

    /// @notice Updates dependency topology. Callable by creator or governance.
    function updateDependencies(bytes32 workflowId, DependencyInput[] calldata dependencyInputs) external {
        Workflow storage wf = _workflows[workflowId];
        if (wf.creator == address(0)) revert UnknownWorkflow();
        if (msg.sender != wf.creator && msg.sender != governance) revert Unauthorized();

        delete _dependencies[workflowId];
        _setDependencies(workflowId, dependencyInputs);
        emit WorkflowUpdated(workflowId);
    }

    /// @notice Flags a workflow for plagiarism or abuse.
    function flagWorkflow(bytes32 workflowId, bool flagged) external {
        if (!attestors[msg.sender] && msg.sender != governance) revert Unauthorized();
        Workflow storage wf = _workflows[workflowId];
        if (wf.creator == address(0)) revert UnknownWorkflow();

        wf.flagged = flagged;
        emit WorkflowFlagged(workflowId, flagged, msg.sender);
    }

    /// @notice Returns workflow information along with dependencies.
    function getWorkflow(bytes32 workflowId)
        external
        view
        returns (Workflow memory wf, Dependency[] memory deps)
    {
        wf = _workflows[workflowId];
        if (wf.creator == address(0)) revert UnknownWorkflow();
        deps = _dependencies[workflowId];
    }

    /// @notice Assigns governance privileges to a new address.
    function setGovernance(address newGovernance) external onlyGovernance {
        if (newGovernance == address(0)) revert Unauthorized();
        governance = newGovernance;
        emit GovernanceUpdated(newGovernance);
    }

    /// @notice Adds or removes an attestor.
    function setAttestor(address attestor, bool approved) external onlyGovernance {
        attestors[attestor] = approved;
        emit AttestorUpdated(attestor, approved);
    }

    function _setDependencies(bytes32 workflowId, DependencyInput[] calldata dependencyInputs) internal {
        uint256 totalWeight;
        for (uint256 i = 0; i < dependencyInputs.length; i++) {
            DependencyInput calldata dep = dependencyInputs[i];
            if (dep.assetId == bytes32(0)) revert Unauthorized();
            totalWeight += dep.weightBps;
            _dependencies[workflowId].push(Dependency({assetId: dep.assetId, weightBps: dep.weightBps}));
        }
        require(totalWeight <= BPS, "Registry: overweight");
    }

    function _validRoute(RoyaltyRoute calldata route) internal pure returns (bool) {
        uint256 total = uint256(route.creatorBps) + route.upstreamBps + route.nodeBps + route.validatorBps + route.treasuryBps;
        return total == BPS;
    }
}
