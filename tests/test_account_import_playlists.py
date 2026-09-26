import pytest
from eth_account.messages import encode_defunct

from tests.test_account_import_music import published
from tests.test_account_import_execution import ready, execute
from tests.test_account_import import inventory, platform, keys, token, PREPARE, challenge_request
import app
import account_import


@pytest.fixture
def playlists(published):
    with app.app.app_context():
        conn = app.get_db()
        with conn:
            for playlist_id, public in (("playlist-private", 0), ("playlist-public", 1), ("playlist-unselected", 0)):
                conn.execute("""INSERT INTO music_playlists
                    (id,owner_wallet,title,description,is_public,artwork_seed,created_at,updated_at)
                    VALUES (?,?,'My tracks','Keep this description',?,'seed',1,1)""",
                    (playlist_id, published[3].address.lower(), public))
                conn.execute("INSERT INTO music_playlist_items VALUES (?,'legacy-song',0,1)", (playlist_id,))
    return published


def select_playlists(playlists):
    harness, headers, account, signer, _, _, _, principal = playlists
    response = harness.client.post(PREPARE, headers={**headers, "Idempotency-Key": "playlists-only"},
        json={"job_ids": [], "include_credits": False, "playlist_ids": ["playlist-public", "playlist-private"]})
    assert response.status_code == 201, response.json
    snapshot = response.json
    response = challenge_request(harness, headers, snapshot)
    assert response.status_code == 201, response.json
    challenge = response.json
    signature = signer.sign_message(encode_defunct(text=challenge["message"])).signature.hex()
    return harness, headers, account, signer, snapshot, challenge, signature, principal


def test_playlist_only_import_preserves_sharing_tracks_and_survives_wallet_unlink(playlists, keys):
    harness, headers, account, signer, _, _, _, _ = playlists
    preview = harness.client.get("/v2/account/wallet-links/link-import/import-preview?limit=2", headers=headers).json
    assert preview["playlist_total"] == 3 and len(preview["playlists"]) == 2
    proof = select_playlists(playlists)
    assert proof[4]["scope"] == ["music_playlists"]
    assert proof[4]["jobs"] == [] and proof[4]["credits"] is None
    assert proof[4]["playlists"][0]["publication_ids"] == ["legacy-song"]
    assert 'playlist_ids: ["playlist-private","playlist-public"]' in proof[5]["message"]
    receipt = execute(proof)
    assert receipt["playlist_ids"] == ["playlist-private", "playlist-public"]
    assert receipt["jobs"] == [] and receipt["credit_units"] == 0
    assert execute(proof) == receipt
    with app.app.app_context():
        conn = app.get_db()
        assert conn.execute("SELECT owner_account_id FROM jobs WHERE id='01-ready'").fetchone()[0] is None
        assert conn.execute("SELECT owner_account_id FROM music_publications WHERE id='legacy-song'").fetchone()[0] is None
        assert conn.execute("SELECT owner_account_id FROM music_playlists WHERE id='playlist-unselected'").fetchone()[0] is None
        assert tuple(conn.execute("SELECT publication_id,position,added_at FROM music_playlist_items WHERE playlist_id='playlist-private'").fetchone()) == ("legacy-song", 0, 1)
        assert not app.music_discover._owner_playlist_row("playlist-private", signer.address.lower())
        with conn:
            conn.execute("UPDATE wallet_links SET unlinked_at=3")
    owned = harness.client.get("/v2/music/playlists", headers=headers).json["playlists"]
    assert {playlist["id"] for playlist in owned} == {"playlist-private", "playlist-public"}
    assert harness.client.get("/music/playlists/playlist-private").status_code == 404
    assert harness.client.get("/music/playlists/playlist-public").status_code == 200
    other = {"Authorization": token(keys, sub="another", sid="another-session")}
    path = "/v2/music/playlists/playlist-private"
    assert harness.client.patch(path, headers=other, json={"title": "Hijack"}).status_code == 404
    changed = harness.client.patch(path, headers=headers, json={"title": "Account playlist"})
    assert changed.status_code == 200, changed.json


@pytest.mark.parametrize("mutation", [
    "UPDATE music_playlists SET title='New title' WHERE id='playlist-private'",
    "UPDATE music_playlists SET is_public=1 WHERE id='playlist-private'",
    "UPDATE music_playlists SET owner_wallet='0x2222222222222222222222222222222222222222' WHERE id='playlist-private'",
    "UPDATE music_playlist_items SET position=2 WHERE playlist_id='playlist-private'",
    "DELETE FROM music_playlist_items WHERE playlist_id='playlist-private'",
])
def test_changed_playlist_cannot_use_previous_signature(playlists, mutation):
    proof = select_playlists(playlists)
    with app.app.app_context():
        conn = app.get_db()
        with conn:
            conn.execute(mutation)
    with pytest.raises(account_import.MigrationError):
        execute(proof)
    with app.app.app_context():
        conn = app.get_db()
        assert conn.execute("SELECT owner_account_id FROM music_playlists WHERE id='playlist-public'").fetchone()[0] is None
        assert conn.execute("SELECT used_at FROM account_import_challenges WHERE id=?", (proof[5]["challenge_id"],)).fetchone()[0] is None


def test_playlist_selection_rejects_missing_and_other_wallets(playlists):
    harness, headers, _, _, _, _, _, _ = playlists
    with app.app.app_context():
        conn = app.get_db()
        with conn:
            conn.execute("UPDATE music_playlists SET owner_wallet='0x2222222222222222222222222222222222222222' WHERE id='playlist-private'")
    for playlist_id in ("missing", "playlist-private"):
        response = harness.client.post(PREPARE, headers={**headers, "Idempotency-Key": playlist_id},
            json={"job_ids": [], "include_credits": False, "playlist_ids": [playlist_id]})
        assert response.status_code == 409 and response.json["error"]["code"] == "import_playlist_unavailable"
