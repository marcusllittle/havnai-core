"""Identity foundation tests use real signatures and independent SQLite connections."""
import concurrent.futures
import sqlite3
import sys
import threading
from pathlib import Path
from unittest.mock import patch

import pytest
from eth_account import Account
from eth_account.messages import encode_defunct

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
import account_identity as identity


ALICE = identity.VerifiedPrincipal("https://auth.example", "alice", "session-a")
BOB = identity.VerifiedPrincipal("https://auth.example", "bob", "session-b")


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "accounts.db"
    conn = sqlite3.connect(path)
    identity.initialize(conn)
    identity.ensure_account(conn, ALICE)
    identity.ensure_account(conn, BOB)
    yield conn, path
    conn.close()


def challenge(conn, principal, signer, **kwargs):
    return identity.issue_wallet_challenge(conn, principal, wallet=signer.address,
        origin="https://joinhavn.io", chain_id=11155111, **kwargs)


def signature(proof, signer):
    return signer.sign_message(encode_defunct(text=proof["message"])).signature.hex()


def complete(conn, principal, proof, signer, purpose="wallet_link"):
    return identity.complete_wallet_challenge(conn, principal,
        challenge_id=proof["challenge_id"], signature=signature(proof, signer), purpose=purpose)


def test_identity_is_stable_and_issuer_scoped(database):
    conn, _ = database
    original = identity.ensure_account(conn, ALICE)
    assert original == identity.ensure_account(conn, identity.VerifiedPrincipal(ALICE.issuer, ALICE.subject, "new-session"))
    assert original != identity.ensure_account(conn, identity.VerifiedPrincipal("https://other.example", ALICE.subject, ALICE.session_id))
    identity.initialize(conn)
    assert original == identity.ensure_account(conn, ALICE)


@pytest.mark.parametrize("principal", [
    identity.VerifiedPrincipal("", "alice", "session"),
    identity.VerifiedPrincipal("issuer", "", "session"),
    identity.VerifiedPrincipal("issuer", "alice", ""),
])
def test_incomplete_principal_rejected(database, principal):
    with pytest.raises(identity.IdentityError, match="invalid_principal"):
        identity.ensure_account(database[0], principal)


def test_link_is_audited_but_does_not_change_legacy_data(database):
    conn, _ = database
    signer = Account.create()
    conn.execute("CREATE TABLE legacy_assets (wallet TEXT, content TEXT)")
    conn.execute("INSERT INTO legacy_assets VALUES (?, 'song')", (signer.address,))
    conn.commit()
    proof = challenge(conn, ALICE, signer)
    assert "This action does not move content" in proof["message"]
    link = complete(conn, ALICE, proof, signer)
    assert conn.execute("SELECT wallet FROM wallet_links WHERE id=?", (link,)).fetchone()[0] == signer.address.lower()
    assert conn.execute("SELECT * FROM legacy_assets").fetchall() == [(signer.address, "song")]
    assert conn.execute("SELECT operation FROM account_audit_events").fetchall() == [("wallet_link",)]
    with pytest.raises(identity.IdentityError, match="invalid_challenge"):
        complete(conn, ALICE, proof, signer)


@pytest.mark.parametrize("principal,purpose", [
    (BOB, "wallet_link"),
    (identity.VerifiedPrincipal(ALICE.issuer, ALICE.subject, "different-session"), "wallet_link"),
    (ALICE, "wallet_unlink"),
])
def test_wrong_account_session_or_purpose_cannot_use_proof(database, principal, purpose):
    conn, _ = database
    signer = Account.create()
    proof = challenge(conn, ALICE, signer)
    with pytest.raises(identity.IdentityError, match="invalid_challenge"):
        complete(conn, principal, proof, signer, purpose)
    assert conn.execute("SELECT used_at FROM account_wallet_challenges").fetchone()[0] is None
    complete(conn, ALICE, proof, signer)


def test_wrong_signer_and_changed_message_rejected(database):
    conn, _ = database
    signer = Account.create()
    proof = challenge(conn, ALICE, signer)
    with pytest.raises(identity.IdentityError, match="invalid_signature"):
        complete(conn, ALICE, proof, Account.create())
    changed = {**proof, "message": proof["message"] + "changed"}
    with pytest.raises(identity.IdentityError, match="invalid_signature"):
        complete(conn, ALICE, changed, signer)
    complete(conn, ALICE, proof, signer)


def test_expiry_and_suspension(database):
    conn, _ = database
    signer = Account.create()
    proof = challenge(conn, ALICE, signer)
    with patch.object(identity.time, "time", return_value=proof["expires_at"]):
        with pytest.raises(identity.IdentityError, match="invalid_challenge"):
            complete(conn, ALICE, proof, signer)
    conn.execute("UPDATE accounts SET status='suspended' WHERE id=?", (identity.ensure_account(conn, ALICE),))
    conn.commit()
    with pytest.raises(identity.IdentityError, match="account_suspended"):
        complete(conn, ALICE, proof, signer)
    with pytest.raises(identity.IdentityError, match="account_suspended"):
        identity.ensure_account(conn, ALICE)


def test_unlink_requires_own_proof_and_preserves_account(database):
    conn, _ = database
    signer = Account.create()
    original_account = identity.ensure_account(conn, ALICE)
    link = complete(conn, ALICE, challenge(conn, ALICE, signer), signer)
    with pytest.raises(identity.IdentityError, match="wallet_link_not_found"):
        challenge(conn, BOB, signer, purpose="wallet_unlink", link_id=link)
    proof = challenge(conn, ALICE, signer, purpose="wallet_unlink", link_id=link)
    outstanding = challenge(conn, ALICE, signer)
    assert complete(conn, ALICE, proof, signer, "wallet_unlink") == link
    assert conn.execute("SELECT unlinked_at FROM wallet_links WHERE id=?", (link,)).fetchone()[0] is not None
    with pytest.raises(identity.IdentityError, match="invalid_challenge"):
        complete(conn, ALICE, outstanding, signer)
    complete(conn, BOB, challenge(conn, BOB, signer), signer)
    assert identity.ensure_account(conn, ALICE) == original_account
    assert conn.execute("SELECT COUNT(*) FROM wallet_links").fetchone()[0] == 2


def test_two_accounts_racing_for_wallet_have_one_winner(database):
    conn, path = database
    signer = Account.create()
    proofs = [challenge(conn, principal, signer) for principal in (ALICE, BOB)]
    barrier = threading.Barrier(2)

    def attach(index):
        db = sqlite3.connect(path, timeout=5)
        try:
            identity.initialize(db)
            barrier.wait(timeout=10)
            return complete(db, (ALICE, BOB)[index], proofs[index], signer)
        except identity.IdentityError as exc:
            return str(exc)
        finally:
            db.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(attach, range(2)))
    assert sum(result.startswith("link_") for result in results) == 1
    assert results.count("wallet_already_linked") == 1
    assert conn.execute("SELECT COUNT(*) FROM account_audit_events").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM account_wallet_challenges WHERE used_at IS NOT NULL").fetchone()[0] == 1


def test_audit_failure_rolls_back_link_and_nonce(database):
    conn, _ = database
    signer = Account.create()
    proof = challenge(conn, ALICE, signer)
    conn.execute("""CREATE TRIGGER reject_audit BEFORE INSERT ON account_audit_events
        BEGIN SELECT RAISE(ABORT, 'disk simulation'); END""")
    with pytest.raises(sqlite3.IntegrityError, match="disk simulation"):
        complete(conn, ALICE, proof, signer)
    assert conn.execute("SELECT COUNT(*) FROM wallet_links").fetchone()[0] == 0
    assert conn.execute("SELECT used_at FROM account_wallet_challenges").fetchone()[0] is None


def test_concurrent_account_creation_is_one_account(database):
    conn, path = database
    principal = identity.VerifiedPrincipal(ALICE.issuer, "new-user", "new-session")
    barrier = threading.Barrier(2)

    def create(_):
        db = sqlite3.connect(path, timeout=5)
        try:
            identity.initialize(db)
            barrier.wait(timeout=10)
            return identity.ensure_account(db, principal)
        finally:
            db.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(create, range(2)))
    assert results[0] == results[1]
    assert conn.execute("SELECT COUNT(*) FROM accounts").fetchone()[0] == 3


def test_concurrent_replay_cannot_create_two_links(database):
    conn, path = database
    signer = Account.create()
    proof = challenge(conn, ALICE, signer)
    barrier = threading.Barrier(2)

    def attach(_):
        db = sqlite3.connect(path, timeout=5)
        try:
            identity.initialize(db)
            barrier.wait(timeout=10)
            return complete(db, ALICE, proof, signer)
        except identity.IdentityError as exc:
            return str(exc)
        finally:
            db.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(attach, range(2)))
    assert sum(result.startswith("link_") for result in results) == 1
    assert results.count("invalid_challenge") == 1
    assert conn.execute("SELECT COUNT(*) FROM account_audit_events").fetchone()[0] == 1


def test_account_can_link_multiple_wallets(database):
    conn, _ = database
    for _ in range(2):
        signer = Account.create()
        complete(conn, ALICE, challenge(conn, ALICE, signer), signer)
    assert conn.execute("SELECT COUNT(*) FROM wallet_links WHERE account_id=?",
                        (identity.ensure_account(conn, ALICE),)).fetchone()[0] == 2


def test_audit_is_append_only(database):
    conn, _ = database
    signer = Account.create()
    complete(conn, ALICE, challenge(conn, ALICE, signer), signer)
    for statement in ["DELETE FROM account_audit_events", "UPDATE account_audit_events SET operation='changed'"]:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            with conn:
                conn.execute(statement)


@pytest.mark.parametrize("origin", ["http://joinhavn.io", "https://joinhavn.io/path", "https://user@joinhavn.io", "https://joinhavn.io\n"])
def test_origin_not_signable_when_malformed(database, origin):
    with pytest.raises(identity.IdentityError, match="invalid_origin"):
        identity.issue_wallet_challenge(database[0], ALICE, wallet=Account.create().address,
            origin=origin, chain_id=1)
