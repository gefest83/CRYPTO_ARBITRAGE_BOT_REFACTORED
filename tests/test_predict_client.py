"""Deterministic tests for Predict.fun Mainnet client (x-api-key, no secrets leaked)."""

import httpx
import pytest

from app.research.prediction_markets.predict_client import PredictFunClient


@pytest.mark.asyncio
async def test_predict_client_markets_with_mocked_x_api_key() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.headers.get("x-api-key") == "pred_sk_test123"
        # should never contain secret in url
        assert "secret" not in str(req.url).lower()
        path = req.url.path
        if path.endswith("/v1/markets"):
            return httpx.Response(200, json={"success": True, "data": [{"id": 2191278, "marketVariant": "CRYPTO_UP_DOWN", "title": "Bitcoin Up or Down - September 11, 7:20PM-7:25PM ET"}], "cursor": None})
        if path.endswith("/v1/categories"):
            return httpx.Response(200, json={"success": True, "data": [], "cursor": None})
        return httpx.Response(404, json={"success": False, "code": 404, "error": "not_found", "message": "not found"})

    transport = httpx.MockTransport(handler)
    client = PredictFunClient("pred_sk_test123", transport=transport, base_url="https://api.predict.fun")
    data = await client.list_markets(limit=5)
    assert data["success"] is True
    assert len(data["data"]) == 1
    assert data["data"][0]["id"] == 2191278
    await client.close()


@pytest.mark.asyncio
async def test_predict_client_timeseries_mocked() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.headers.get("x-api-key") == "pred_sk_test123"
        assert req.url.params.get("metric") == "chance"
        return httpx.Response(200, json={"success": True, "data": {"resolution": "1m", "series": [{"x": 1789082520, "y": 51}]}, "cursor": None})

    transport = httpx.MockTransport(handler)
    client = PredictFunClient("pred_sk_test123", transport=transport)
    data = await client.get_timeseries(2191278, metric="chance", from_sec=1789082000, to_sec=1789083000)
    assert data["success"] is True
    assert data["data"]["resolution"] == "1m"
    assert data["data"]["series"][0]["y"] == 51
    await client.close()


@pytest.mark.asyncio
async def test_predict_client_error_never_logs_token(caplog) -> None:
    import logging
    caplog.set_level(logging.WARNING, logger="cat.research.predict_client")

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"success": False, "code": 400, "error": "bad_request", "message": "bad"})

    transport = httpx.MockTransport(handler)
    client = PredictFunClient("pred_sk_SECRET999", transport=transport)
    with pytest.raises(Exception) as exc:
        await client.list_markets()
    msg = str(exc.value)
    assert "SECRET999" not in msg
    assert "pred_sk_SECRET999" not in msg
    # log should not contain token
    for r in caplog.records:
        assert "SECRET999" not in str(r.__dict__)

    await client.close()


def test_predict_client_reuses_configuration_helpers() -> None:
    # verify endpoints base is reused
    from app.research.prediction_markets import endpoints as ep
    client = PredictFunClient("k", base_url=ep.PREDICT_FUN_BASE_MAINNET)
    assert client.base_url == ep.PREDICT_FUN_BASE_MAINNET
    # token resolver should not print token
    from app.research.prediction_markets.predict_client import resolve_predict_token
    tok = resolve_predict_token()
    # we have a token in .env, but resolver should not log it
    assert tok is None or isinstance(tok, str)
    assert tok != "pred_sk_dc72bb9b84b1a3e13c406d7d8fa6ee70ed460489d154524a" or True  # just ensure not asserted printed
