import base64
import json
import time

import pytest

from tests.test_account_auth import keys, token
from tests.test_account_music import music
import app
import artifact_lifecycle as lifecycle


def seed(conn, account, count, prefix="recovery"):
    now = time.time()
    for index in range(count):
        job_id = f"{prefix}-{index:04d}"
        conn.execute("""INSERT INTO jobs
            (id,wallet,model,data,task_type,weight,status,timestamp,owner_account_id,creator_account_id)
            VALUES (?,'','fixture-model',?,'MUSIC_GEN',1,'succeeded',?,?,?)""",
            (job_id, json.dumps({"prompt": "private fixture prompt", "media_url": "/private/output"}), now, account, account))
        # Multiple jobs share deletion timestamps to exercise the tie-breaker.
        deleted_at = now - index // 4
        conn.execute("INSERT INTO artifact_lifecycle VALUES (?,?,?,NULL,NULL)",
            (job_id, deleted_at, deleted_at + lifecycle.RECOVERY_SECONDS))
    conn.commit()


def test_pages_reach_every_owned_deletion_past_the_old_100_row_limit(music, keys):
    harness, headers, account = music
    conn = app.get_db()
    bob_headers = {"Authorization": token(keys, sub="user_bob", sid="sess_bob")}
    bob = harness.client.get("/v2/account", headers=bob_headers).json["id"]
    seed(conn, account, 137)
    seed(conn, bob, 3, "private-bob")
    conn.execute("UPDATE artifact_lifecycle SET restored_at=? WHERE job_id='recovery-0002'", (time.time(),))
    conn.commit()
    expected = [row[0] for row in conn.execute("""SELECT d.job_id FROM artifact_lifecycle d
        JOIN jobs j ON j.id=d.job_id WHERE j.owner_account_id=? AND d.restored_at IS NULL
        ORDER BY deleted_at DESC,job_id DESC""", (account,))]
    seen = []
    cursor = None
    for _ in range(10):
        response = harness.client.get("/v2/account/deleted-generations", headers=headers,
            query_string={"before": cursor} if cursor else {})
        assert response.status_code == 200, response.json
        assert response.cache_control.no_store
        assert "private" in response.headers["Cache-Control"]
        rows = response.json["generations"]
        assert len(rows) <= 25
        for row in rows:
            assert set(row) == {"job_id", "deleted_at", "recover_until", "purged_at"}
        assert "private fixture prompt" not in response.get_data(as_text=True)
        assert "private-bob" not in response.get_data(as_text=True)
        seen.extend(row["job_id"] for row in rows)
        cursor = response.json["next_cursor"]
        if cursor is None:
            break
    assert cursor is None
    assert seen == expected
    assert len(seen) == 136 == len(set(seen))
    assert harness.client.get("/v2/account/deleted-generations").status_code == 401


def test_restoring_page_boundary_does_not_skip_remaining_creations(music, keys):
    harness, headers, account = music
    conn = app.get_db()
    seed(conn, account, 4)
    first = harness.client.get("/v2/account/deleted-generations?limit=2", headers=headers).json
    cursor = first["next_cursor"]
    for row in first["generations"]:
        assert harness.client.post(f"/v2/jobs/{row['job_id']}/restore", headers=headers).status_code == 200
    seed(conn, account, 1, "newer-deletion")
    second = harness.client.get("/v2/account/deleted-generations", headers=headers,
        query_string={"before": cursor, "limit": 2}).json
    assert [row["job_id"] for row in second["generations"]] == ["recovery-0001", "recovery-0000"]
    assert second["next_cursor"] is None
    bob = {"Authorization": token(keys, sub="user_bob", sid="sess_bob")}
    rejected = harness.client.get("/v2/account/deleted-generations", headers=bob, query_string={"before": cursor})
    assert rejected.status_code == 422
    assert rejected.json["error"]["code"] == "invalid_recovery_cursor"


@pytest.mark.parametrize("limit", ["0", "101", "-1", "1.5", "abc", "9999999999999999999999"])
def test_rejects_invalid_page_sizes(music, limit):
    harness, headers, _ = music
    response = harness.client.get("/v2/account/deleted-generations", headers=headers, query_string={"limit": limit})
    assert response.status_code == 422
    assert response.json["error"]["code"] == "invalid_recovery_limit"


@pytest.mark.parametrize("position", ["", "invalid!", "z" * 1025, [], [1, "wrong-owner", 1, "job"],
    [1, "owner", float("nan"), "job"], [1, "owner", float("inf"), "job"], [1, "owner", -1, "job"],
    [1, "owner", True, "job"], [1, "owner", 10, ""], [1, "owner", 10, "x" * 201]])
def test_rejects_malformed_cursors_without_revealing_other_accounts(music, position):
    harness, headers, account = music
    if isinstance(position, list):
        position = [account if value == "owner" else value for value in position]
        position = base64.urlsafe_b64encode(json.dumps(position).encode()).decode()
    response = harness.client.get("/v2/account/deleted-generations", headers=headers, query_string={"before": position})
    assert response.status_code == 422
    assert response.json["error"]["code"] == "invalid_recovery_cursor"
