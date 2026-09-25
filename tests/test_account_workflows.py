import pytest
from tests.test_account_jobs import platform, keys, token
import app


@pytest.fixture
def setup(platform, keys):
    harness, headers, account = platform
    with app.app.app_context():
        app.workflows.init_workflow_tables(app.get_db())
    return harness, {**headers, "Idempotency-Key": "workflow-create"}, account, {"Authorization": token(keys, sub="other", sid="other-session")}


def test_create_retry_update_publish_delete_without_wallet(setup):
    harness, headers, account, other = setup
    path = "/v2/account/workflows"
    body = {"name": "My workflow", "config": {"prompt": "private"}, "tags": ["image"]}
    first = harness.client.post(path, headers=headers, json=body)
    assert first.status_code == 201
    item = path + "/" + str(first.json["id"])
    assert harness.client.post(path, headers=headers, json=body).json == first.json
    assert harness.client.post(path, headers=headers, json={**body, "name": "Changed"}).status_code == 409
    assert harness.client.get(path, headers=headers).json["total"] == 1
    assert harness.client.get(path, headers=other).json["total"] == 0
    assert harness.client.get(item, headers=other).status_code == 404
    assert harness.client.patch(item, headers=other, json={"published": True}).status_code == 404
    assert harness.client.delete(item, headers=other).status_code == 404
    updated = harness.client.patch(item, headers=headers, json={"published": True, "name": "Public template"})
    assert updated.status_code == 200 and updated.json["published"] is True
    with app.app.app_context():
        assert app.get_db().execute("SELECT creator_account_id,owner_account_id,creator_wallet FROM workflow_registry WHERE id=?", (first.json["id"],)).fetchone()[:] == (account, account, "")
    assert harness.client.delete(item, headers=headers).status_code == 204
    assert harness.client.get(item, headers=headers).status_code == 404
    # A late creation retry cannot resurrect a deleted workflow.
    assert harness.client.post(path, headers=headers, json=body).json == first.json
    assert harness.client.get(path, headers=headers).json["total"] == 0


def test_legacy_routes_cannot_read_or_mutate_account_workflows(setup):
    harness, headers, _, _ = setup
    created = harness.client.post("/v2/account/workflows", headers=headers, json={"name": "Secret", "published": True}).json
    identifier = created["id"]
    assert harness.client.get(f"/workflows/{identifier}").status_code == 404
    assert all(row["id"] != identifier for row in harness.client.get("/workflows").json["workflows"])
    assert all(row["id"] != identifier for row in harness.client.get("/marketplace/browse").json["workflows"])
    with app.app.app_context():
        assert app.workflows.update_workflow(identifier, "", name="Hijacked") is None
        assert app.workflows.publish_workflow(identifier, "") is None
        app.workflows.increment_usage(identifier)
        assert app.get_db().execute("SELECT name,usage_count FROM workflow_registry WHERE id=?", (identifier,)).fetchone()[:] == ("Secret", 0)


@pytest.mark.parametrize("body", [{"name": ""}, {"name": "x", "wallet": "spoof"}, {"name": "x", "owner_account_id": "spoof"},
    {"name": "x", "published": "false"}, {"name": "x", "config": []}, {"name": "x", "tags": [False]},
    {"name": "x", "config": {"prompt": "x" * 65536}}])
def test_rejects_invalid_or_spoofed_payloads(setup, body):
    harness, headers, _, _ = setup
    assert harness.client.post("/v2/account/workflows", headers=headers, json=body).status_code == 422
    assert harness.client.get("/v2/account/workflows", headers=headers).json["total"] == 0


def test_requires_account_not_static_owner_or_wallet(setup):
    harness, headers, _, _ = setup
    for auth in ({}, harness.owner_headers):
        assert harness.client.get("/v2/account/workflows", headers=auth).status_code == 401
        assert harness.client.post("/v2/account/workflows", headers=auth, json={"name": "x"}).status_code == 401
    assert harness.client.get("/v2/account/workflows?limit=101", headers=headers).status_code == 422


def test_publication_is_explicit_revocable_and_hides_identity(setup):
    harness, headers, account, _ = setup
    created = harness.client.post("/v2/account/workflows", headers=headers, json={"name": "Template", "config": {"steps": 20}}).json
    private = f'/v2/account/workflows/{created["id"]}'
    public = f'/v2/workflows/{created["id"]}'
    assert harness.client.get(public).status_code == 404
    assert harness.client.get("/v2/workflows").json["total"] == 0
    harness.client.patch(private, headers=headers, json={"published": True})
    result = harness.client.get(public)
    assert result.status_code == 200 and result.json["config"] == {"steps": 20}
    assert not {"creator_wallet", "creator_account_id", "owner_account_id"} & result.json.keys()
    assert result.headers["Cache-Control"] == "private, no-store"
    assert harness.client.get("/v2/workflows").json["total"] == 1
    harness.client.patch(private, headers=headers, json={"published": False})
    assert harness.client.get(public).status_code == 404
    harness.client.patch(private, headers=headers, json={"published": True})
    with app.app.app_context():
        conn = app.get_db()
        with conn:
            conn.execute("UPDATE accounts SET status='suspended' WHERE id=?", (account,))
    assert harness.client.get(public).status_code == 404
    assert harness.client.get("/v2/workflows").json["total"] == 0


def test_additive_schema_preserves_legacy_workflows(setup):
    harness, headers, _, _ = setup
    with app.app.app_context():
        wallet = "0x" + "1" * 40
        legacy = app.workflows.create_workflow(wallet, "Legacy", config={"steps": 10})
        app.workflows.init_workflow_tables(app.get_db())
        assert app.workflows.get_workflow(legacy["id"])["config"] == {"steps": 10}
        assert app.workflows.update_workflow(legacy["id"], wallet, name="Updated")["name"] == "Updated"
        assert app.workflows.publish_workflow(legacy["id"], wallet)["published"] is True
        assert any(row["id"] == legacy["id"] for row in app.workflows.browse_marketplace()["workflows"])
    assert harness.client.get("/v2/account/workflows", headers=headers).json["total"] == 0
    assert harness.client.get(f'/v2/account/workflows/{legacy["id"]}', headers=headers).status_code == 404
    assert harness.client.get("/v2/workflows?search=Updated").json["total"] == 1
    assert harness.client.get("/v2/workflows?search=missing").json["total"] == 0
    assert harness.client.get("/v2/workflows?category=Video").json["total"] == 0
    assert harness.client.get(f'/v2/workflows/{legacy["id"]}').status_code == 200
    assert harness.client.get("/v2/workflows?search=" + "x" * 257).status_code == 422
