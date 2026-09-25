import io
import sqlite3
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import pytest

from tests.test_account_jobs import platform, keys, token, create
import app
import account_marketplace as market
import account_ledger
import job_helpers


@pytest.fixture
def market_case(platform, keys):
    harness, seller_headers, seller = platform
    buyer_headers = {"Authorization": token(keys, sub="buyer", sid="buyer-session"), "Idempotency-Key": "buy-one"}
    buyer = harness.client.get("/v2/account", headers=buyer_headers).json["id"]
    with app.app.app_context():
        app.gallery.init_gallery_tables(app.get_db())
        conn = app.get_db()
        conn.execute("BEGIN IMMEDIATE")
        with conn:
            account_ledger.fund_in_transaction(conn, buyer, 10000, payment_id="buyer-paid")
    response = create(harness, seller_headers)
    assert response.status_code == 202, response.json
    job = response.json
    with app.app.app_context():
        attempt = job_helpers.assign_job_to_node(job["id"], "node-test")
    uploaded = harness.client.post(f"/v1/node/jobs/{job['id']}/artifacts", headers=harness.node_headers,
        data={"node_id": "node-test", "attempt_id": attempt, "kind": "image",
              "file": (io.BytesIO(b"private-artwork"), "out.png")})
    assert uploaded.status_code == 201, uploaded.json
    with app.app.app_context():
        assert job_helpers.complete_job(job["id"], "node-test", "succeeded", attempt)
    detail = harness.client.get(f"/v2/jobs/{job['id']}", headers=seller_headers).json
    body = {"job_id": job["id"], "artifact_id": detail["artifacts"][0]["id"], "title": "Blue sky", "price_units": 3000}
    return harness, seller_headers, buyer_headers, seller, buyer, body, detail["artifacts"][0]["url"]


def listing(case):
    harness, headers, _, _, _, body, _ = case
    response = harness.client.post("/v2/marketplace/listings", headers=headers, json=body)
    assert response.status_code == 201, response.json
    return response.json["listing_id"]


def test_sale_transfers_private_access_once_and_preserves_creator(market_case):
    harness, seller_headers, buyer_headers, seller, buyer, body, url = market_case
    listing_id = listing(market_case)
    assert listing(market_case) == listing_id
    assert harness.client.get(url, headers=buyer_headers).status_code == 404
    purchase_url = f"/v2/marketplace/listings/{listing_id}/purchase"
    responses = [harness.client.post(purchase_url, headers=buyer_headers, json={"expected_price_units": 3000}) for _ in range(2)]
    assert responses[0].status_code == 200, responses[0].json
    assert responses[0].json == responses[1].json
    assert responses[0].headers["Cache-Control"] == "private, no-store"
    assert harness.client.get(url, headers=buyer_headers).data == b"private-artwork"
    assert harness.client.get(url, headers=seller_headers).status_code == 404
    assert harness.client.get(f"/v2/jobs/{body['job_id']}", headers=seller_headers).status_code == 404
    assert harness.client.get("/v2/account/credits", headers=buyer_headers).json["available_units"] == 7000
    assert harness.client.get("/v2/account/credits", headers=seller_headers).json["available_units"] == 12000
    with app.app.app_context():
        conn = app.get_db()
        assert tuple(conn.execute("SELECT creator_account_id,owner_account_id FROM jobs WHERE id=?", (body["job_id"],)).fetchone()) == (seller, buyer)
        assert conn.execute("SELECT COUNT(*) FROM account_credit_sales").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM gallery_sales").fetchone()[0] == 1
    assert harness.client.get(f"/gallery/listings/{listing_id}").status_code == 404
    assert harness.client.post("/v2/marketplace/listings", headers={**seller_headers, "Idempotency-Key": "reclaim"}, json=body).status_code == 404
    relisted = harness.client.post("/v2/marketplace/listings", headers={**buyer_headers, "Idempotency-Key": "resale"}, json=body)
    assert relisted.status_code == 201, relisted.json
    assert relisted.json["listing_id"] != listing_id
    with app.app.app_context():
        conn = app.get_db()
        with conn:
            conn.execute("INSERT INTO account_collection_hidden VALUES (?,?,0)", (seller, body["job_id"]))
    repurchased = harness.client.post(f"/v2/marketplace/listings/{relisted.json['listing_id']}/purchase",
        headers={**seller_headers, "Idempotency-Key": "buy-back"}, json={"expected_price_units": 3000})
    assert repurchased.status_code == 200, repurchased.json
    assert harness.client.get(url, headers=seller_headers).data == b"private-artwork"
    assert harness.client.get(url, headers=buyer_headers).status_code == 404
    with app.app.app_context():
        assert app.get_db().execute("SELECT COUNT(*) FROM account_collection_hidden WHERE account_id=? AND job_id=?",
                                   (seller, body["job_id"])).fetchone()[0] == 0


def test_ownership_failure_rolls_back_sale_and_can_retry(market_case):
    harness, _, buyer_headers, seller, buyer, body, _ = market_case
    listing_id = listing(market_case)
    with app.app.app_context():
        conn = app.get_db()
        conn.execute("CREATE TRIGGER reject_owner BEFORE UPDATE OF owner_account_id ON jobs BEGIN SELECT RAISE(ABORT,'ownership failure'); END")
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError, match="ownership failure"):
            market.purchase(conn, buyer, listing_id, "retry", {"expected_price_units": 3000})
        assert account_ledger.balance(conn, buyer)["available_units"] == 10000
        assert account_ledger.balance(conn, seller)["available_units"] == 9000
        assert conn.execute("SELECT COUNT(*) FROM account_credit_sales").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM gallery_sales").fetchone()[0] == 0
        conn.execute("DROP TRIGGER reject_owner")
        conn.commit()
        assert market.purchase(conn, buyer, listing_id, "retry", {"expected_price_units": 3000})["listing_id"] == listing_id


def test_price_conflicts_self_purchase_and_foreign_delist_do_not_charge(market_case):
    harness, seller_headers, buyer_headers, _, _, _, _ = market_case
    listing_id = listing(market_case)
    route = f"/v2/marketplace/listings/{listing_id}"
    assert harness.client.post(route + "/purchase", headers=buyer_headers, json={"expected_price_units": 2999}).status_code == 409
    assert harness.client.post(route + "/purchase", headers=seller_headers, json={"expected_price_units": 3000}).status_code == 409
    assert harness.client.delete(route, headers=buyer_headers).status_code == 404
    for _ in range(2):
        assert harness.client.delete(route, headers=seller_headers).status_code == 204
    assert harness.client.post(route + "/purchase", headers=buyer_headers, json={"expected_price_units": 3000}).status_code == 404
    assert harness.client.get("/v2/account/credits", headers=buyer_headers).json["available_units"] == 10000


@pytest.mark.parametrize("change", [{"price_units": True}, {"price_units": 0}, {"price_units": 1.5},
                                  {"wallet": "spoof"}, {"title": ""}, {"artifact_id": "foreign"}])
def test_invalid_listings_do_not_publish(market_case, change):
    harness, headers, _, _, _, body, _ = market_case
    response = harness.client.post("/v2/marketplace/listings", headers=headers, json={**body, **change})
    assert response.status_code in {409, 422}, response.json
    with app.app.app_context():
        assert app.get_db().execute("SELECT COUNT(*) FROM gallery_listings").fetchone()[0] == 0


def test_duplicate_concurrent_purchase_returns_one_sale(market_case):
    _, _, _, seller, buyer, _, _ = market_case
    listing_id = listing(market_case)

    def purchase(_):
        with app.app.app_context():
            return market.purchase(app.get_db(), buyer, listing_id, "same", {"expected_price_units": 3000})

    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(purchase, range(4)))
    assert all(result == results[0] for result in results)
    with app.app.app_context():
        assert account_ledger.balance(app.get_db(), buyer)["available_units"] == 7000
        assert account_ledger.balance(app.get_db(), seller)["available_units"] == 12000


def test_different_buyers_cannot_purchase_the_same_asset(market_case, keys):
    harness, _, _, seller, buyer, _, _ = market_case
    listing_id = listing(market_case)
    third = harness.client.get("/v2/account", headers={"Authorization": token(keys, sub="third", sid="third-session")}).json["id"]
    with app.app.app_context():
        conn = app.get_db()
        conn.execute("BEGIN IMMEDIATE")
        with conn:
            account_ledger.fund_in_transaction(conn, third, 10000, payment_id="third-paid")

    def purchase(account):
        with app.app.app_context():
            try:
                return market.purchase(app.get_db(), account, listing_id, "same-key", {"expected_price_units": 3000})
            except market.MarketplaceError as exc:
                return str(exc)

    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(purchase, [buyer, third]))
    assert sum(isinstance(result, dict) for result in results) == 1
    assert results.count("listing_not_found") == 1
    with app.app.app_context():
        conn = app.get_db()
        assert sum(account_ledger.balance(conn, account)["available_units"] for account in [buyer, third]) == 17000
        assert account_ledger.balance(conn, seller)["available_units"] == 12000


def test_missing_original_cannot_be_sold(market_case):
    harness, _, headers, _, _, body, _ = market_case
    listing_id = listing(market_case)
    with app.app.app_context():
        path = app.get_db().execute("SELECT path FROM artifacts WHERE id=?", (body["artifact_id"],)).fetchone()[0]
    Path(path).unlink()
    response = harness.client.post(f"/v2/marketplace/listings/{listing_id}/purchase", headers=headers,
                                   json={"expected_price_units": 3000})
    assert response.status_code == 409
    assert response.json["error"]["code"] == "marketplace_artifact_unavailable"
    assert harness.client.get("/v2/account/credits", headers=headers).json["available_units"] == 10000


@pytest.mark.parametrize("state,task", [("running", "IMAGE_GEN"), ("succeeded", "FACE_SWAP"), ("succeeded", "MUSIC_GEN")])
def test_only_completed_eligible_outputs_can_be_listed(market_case, state, task):
    harness, headers, _, _, _, body, _ = market_case
    with app.app.app_context():
        conn = app.get_db()
        with conn:
            conn.execute("UPDATE jobs SET status=?,task_type=? WHERE id=?", (state, task, body["job_id"]))
    response = harness.client.post("/v2/marketplace/listings", headers=headers, json=body)
    assert response.status_code == 409
    assert response.json["error"]["code"] == "marketplace_ineligible"


def test_account_identity_is_required_and_payload_cannot_replace_it(market_case):
    harness, headers, buyer_headers, _, _, body, _ = market_case
    assert harness.client.post("/v2/marketplace/listings", json=body).status_code == 401
    assert harness.client.post("/v2/marketplace/listings", headers=harness.owner_headers, json=body).status_code == 401
    assert harness.client.post("/v2/marketplace/listings", headers=buyer_headers, json=body).status_code == 404
    listing_id = listing(market_case)
    response = harness.client.post("/v2/marketplace/listings", headers=headers, json={**body, "price_units": 4000})
    assert response.status_code == 409
    assert response.json["error"]["code"] == "idempotency_conflict"
    url = f"/v2/marketplace/listings/{listing_id}/purchase"
    assert harness.client.post(url, headers=buyer_headers, json={"expected_price_units": 3000}).status_code == 200
    conflict = harness.client.post(url, headers=buyer_headers, json={"expected_price_units": 4000})
    assert conflict.status_code == 409
    assert conflict.json["error"]["code"] == "idempotency_conflict"
