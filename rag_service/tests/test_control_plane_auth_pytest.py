from __future__ import annotations


def test_product_api_requires_admin_token(api_client):
    response = api_client.get("/api/threads")
    assert response.status_code == 200

    api_client.headers.pop("Authorization")
    response = api_client.get("/api/threads")
    assert response.status_code == 401


def test_health_endpoints_remain_public(api_client):
    api_client.headers.pop("Authorization")
    assert api_client.get("/health").status_code in {200, 503}
    assert api_client.get("/ready").status_code in {200, 503}
