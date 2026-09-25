"""Run an authorized import on a restored copy; never mutate its source."""
import sqlite3
from pathlib import Path

from tests.test_account_import_execution import ready
from tests.test_account_import import inventory, platform, keys, token
import app
import account_import
import account_ledger
from scripts.account_restore_drill import run_drill


def test_signed_import_on_restored_copy_isolated_from_source(ready, tmp_path):
    _, _, account, signer, snapshot, challenge, signature, principal = ready
    with app.app.app_context():
        source = app.get_db()
        source_path = source.execute("PRAGMA database_list").fetchone()[2]
        report = run_drill(Path(source_path), tmp_path / "drill")
        assert report["verified"]
        copy = sqlite3.connect(tmp_path / "drill" / "restored.sqlite3")
        copy.row_factory = sqlite3.Row
        copy.execute("PRAGMA foreign_keys=ON")
        try:
            arguments = dict(challenge_id=challenge["challenge_id"], signature=signature,
                             origin="https://joinhavn.io", chain_id=11155111)
            receipt = account_import.execute(copy, principal, snapshot["id"], **arguments)
            assert receipt["credit_units"] == 2125
            assert account_import.execute(copy, principal, snapshot["id"], **arguments) == receipt
            assert account_ledger.balance(copy, account)["settled_units"] == 12125
            assert copy.execute("SELECT balance FROM credits WHERE wallet=?", (signer.address.lower(),)).fetchone()[0] == 0
            assert copy.execute("SELECT COUNT(*) FROM account_import_receipts").fetchone()[0] == 1
            assert copy.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            assert copy.execute("PRAGMA foreign_key_check").fetchall() == []
        finally:
            copy.close()
        assert account_ledger.balance(source, account)["settled_units"] == 10000
        assert source.execute("SELECT balance FROM credits WHERE wallet=?", (signer.address.lower(),)).fetchone()[0] == 2.125
        assert source.execute("SELECT owner_account_id FROM jobs WHERE id='01-ready'").fetchone()[0] is None
        assert source.execute("SELECT used_at FROM account_import_challenges WHERE id=?", (challenge["challenge_id"],)).fetchone()[0] is None
        assert source.execute("SELECT COUNT(*) FROM account_import_receipts").fetchone()[0] == 0
