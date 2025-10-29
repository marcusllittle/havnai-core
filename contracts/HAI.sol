// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title HAI Token
/// @notice ERC-20 token powering the HavnAI protocol with governance-managed emission.
contract HAI {
    /// @notice Mapping that stores account balances.
    mapping(address => uint256) private _balances;
    /// @notice Mapping that stores allowances granted to spenders.
    mapping(address => mapping(address => uint256)) private _allowances;

    /// @notice Total supply of tokens in circulation.
    uint256 private _totalSupply;

    /// @notice Token metadata name.
    string public constant name = "HavnAI";
    /// @notice Token metadata symbol.
    string public constant symbol = "HAI";
    /// @notice Token decimals as per ERC-20 specification.
    uint8 public constant decimals = 18;

    /// @notice Address of contract governance.
    address public governance;
    /// @notice Optional emission controller allowed to mint in addition to governance.
    address public emitter;

    /// @notice Emitted when tokens are moved between accounts.
    event Transfer(address indexed from, address indexed to, uint256 value);
    /// @notice Emitted when an allowance is granted or updated.
    event Approval(address indexed owner, address indexed spender, uint256 value);
    /// @notice Emitted when the emitter role is updated.
    event EmitterUpdated(address indexed newEmitter);
    /// @notice Emitted when governance is transferred.
    event GovernanceUpdated(address indexed newGovernance);

    /// @param initialGovernance Address that receives governance rights.
    /// @param initialEmitter Address capable of minting tokens alongside governance.
    constructor(address initialGovernance, address initialEmitter) {
        governance = initialGovernance == address(0) ? msg.sender : initialGovernance;
        emitter = initialEmitter;
    }

    /// @notice Returns the total amount of tokens in circulation.
    function totalSupply() external view returns (uint256) {
        return _totalSupply;
    }

    /// @notice Reads the balance of an account.
    function balanceOf(address account) external view returns (uint256) {
        return _balances[account];
    }

    /// @notice Moves tokens from caller to recipient.
    function transfer(address to, uint256 amount) external returns (bool) {
        _transfer(msg.sender, to, amount);
        return true;
    }

    /// @notice Reads allowance from owner to spender.
    function allowance(address owner, address spender) external view returns (uint256) {
        return _allowances[owner][spender];
    }

    /// @notice Approves spender to transfer tokens on sender's behalf.
    function approve(address spender, uint256 amount) external returns (bool) {
        _approve(msg.sender, spender, amount);
        return true;
    }

    /// @notice Moves tokens using an allowance previously granted by `from`.
    function transferFrom(address from, address to, uint256 amount) external returns (bool) {
        uint256 currentAllowance = _allowances[from][msg.sender];
        require(currentAllowance >= amount, "HAI: allowance exceeded");

        unchecked {
            _approve(from, msg.sender, currentAllowance - amount);
        }
        _transfer(from, to, amount);
        return true;
    }

    /// @notice Mints new tokens to `to`. Callable by governance or emitter.
    function mint(address to, uint256 amount) external {
        require(msg.sender == governance || msg.sender == emitter, "HAI: unauthorized mint");
        _mint(to, amount);
    }

    /// @notice Burns tokens from the caller's balance.
    function burn(uint256 amount) external {
        _burn(msg.sender, amount);
    }

    /// @notice Updates the emitter role. Governance-only action.
    function setEmitter(address newEmitter) external {
        require(msg.sender == governance, "HAI: not governance");
        emitter = newEmitter;
        emit EmitterUpdated(newEmitter);
    }

    /// @notice Transfers governance to a new address.
    function setGovernance(address newGovernance) external {
        require(msg.sender == governance, "HAI: not governance");
        require(newGovernance != address(0), "HAI: zero governance");
        governance = newGovernance;
        emit GovernanceUpdated(newGovernance);
    }

    function _transfer(address from, address to, uint256 amount) internal {
        require(to != address(0), "HAI: transfer to zero");
        uint256 balance = _balances[from];
        require(balance >= amount, "HAI: insufficient balance");

        unchecked {
            _balances[from] = balance - amount;
        }
        _balances[to] += amount;
        emit Transfer(from, to, amount);
    }

    function _mint(address to, uint256 amount) internal {
        require(to != address(0), "HAI: mint to zero");
        _totalSupply += amount;
        _balances[to] += amount;
        emit Transfer(address(0), to, amount);
    }

    function _burn(address from, uint256 amount) internal {
        uint256 balance = _balances[from];
        require(balance >= amount, "HAI: burn exceeds balance");
        unchecked {
            _balances[from] = balance - amount;
        }
        _totalSupply -= amount;
        emit Transfer(from, address(0), amount);
    }

    function _approve(address owner, address spender, uint256 amount) internal {
        require(owner != address(0) && spender != address(0), "HAI: approve zero");
        _allowances[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }
}
