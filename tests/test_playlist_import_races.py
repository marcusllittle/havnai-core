import sqlite3

import pytest

from tests.test_account_import_playlists import playlists, select_playlists
from tests.test_account_import_music import published
from tests.test_account_import_execution import ready, execute
from tests.test_account_import import inventory, platform, keys, token
import app


def mutate(operation, wallet):
    music = app.music_discover
    args = {"owner_wallet": wallet}
    if operation == "update":
        return music.update_playlist("playlist-private", title="Edited", **args)
    if operation == "delete":
        return music.delete_playlist("playlist-private", **args)
    if operation == "add":
        return music.add_playlist_item("playlist-private", "legacy-song", **args)
    if operation == "remove":
        return music.remove_playlist_item("playlist-private", "legacy-song", **args)
    return music.reorder_playlist_items("playlist-private", ["legacy-song"], **args)


@pytest.mark.parametrize("operation", ["update", "delete", "add", "remove", "reorder"])
def test_import_cannot_interleave_after_legacy_ownership_check(playlists, monkeypatch, operation):
    original = app.music_discover._owner_playlist_row
    attempted = []

    def check_then_race(*args):
        row = original(*args)
        conn = app.get_db()
        assert conn.in_transaction
        path = conn.execute("PRAGMA database_list").fetchone()[2]
        other = sqlite3.connect(path, timeout=0)
        try:
            # This is the ownership write the import would attempt after the
            # old wallet passed authorization. It must wait for the whole edit.
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                other.execute("UPDATE music_playlists SET owner_account_id=? WHERE id='playlist-private'", (playlists[2],))
            attempted.append(True)
        finally:
            other.close()
        return row

    monkeypatch.setattr(app.music_discover, "_owner_playlist_row", check_then_race)
    with app.app.app_context():
        assert mutate(operation, playlists[3].address.lower())["ok"]
        assert not app.get_db().in_transaction
    assert attempted == [True]


@pytest.mark.parametrize("operation", ["update", "delete", "add", "remove", "reorder"])
def test_legacy_edits_are_rejected_after_completed_import(playlists, operation):
    execute(select_playlists(playlists))
    with app.app.app_context():
        conn = app.get_db()
        before = conn.total_changes
        assert mutate(operation, playlists[3].address.lower()) == {"ok": False, "error": "playlist_not_found"}
        assert conn.total_changes == before
        assert not conn.in_transaction


def test_multirow_legacy_edit_rolls_back_on_failure(playlists, monkeypatch):
    def fail(*args):
        raise RuntimeError("position update failed")
    monkeypatch.setattr(app.music_discover, "_compact_playlist_positions", fail)
    with app.app.app_context():
        conn = app.get_db()
        with pytest.raises(RuntimeError, match="position update failed"):
            mutate("remove", playlists[3].address.lower())
        assert not conn.in_transaction
        assert conn.execute("SELECT COUNT(*) FROM music_playlist_items WHERE playlist_id='playlist-private'").fetchone()[0] == 1
