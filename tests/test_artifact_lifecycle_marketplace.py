from tests.test_account_jobs import platform, keys
from tests.test_account_marketplace import market_case, listing
import app
import account_marketplace as market


def test_delete_delists_and_retains_account_ledger(market_case):
    harness, seller_headers, buyer_headers, seller, buyer, body, url = market_case
    listing_id = listing(market_case)
    with app.app.app_context():
        conn = app.get_db()
        app.music_discover.init_music_discover_tables(conn)
        conn.commit()
        before = [tuple(row) for row in conn.execute("SELECT * FROM account_credit_ledger")]
    job_url = f"/v2/jobs/{body['job_id']}"
    response = harness.client.delete(job_url, headers=seller_headers)
    assert response.status_code == 200, response.json
    assert harness.client.get(url, headers=seller_headers).status_code == 404
    response = harness.client.post(f"/v2/marketplace/listings/{listing_id}/purchase", headers=buyer_headers, json={"expected_price_units": 3000})
    assert response.status_code == 404
    with app.app.app_context():
        conn = app.get_db()
        assert market.account_listings(conn, seller)["listings"] == []
        assert [tuple(row) for row in conn.execute("SELECT * FROM account_credit_ledger")] == before
    assert harness.client.post(job_url + "/restore", headers=seller_headers).status_code == 200
    with app.app.app_context():
        assert app.get_db().execute("SELECT listed FROM gallery_listings WHERE id=?", (listing_id,)).fetchone()[0] == 0
