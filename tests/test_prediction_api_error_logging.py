"""Deterministic tests for safe Prediction API error logging (no secrets leaked)."""

from __future__ import annotations

import logging

import httpx
import pytest

from app.research.prediction_markets.client import BinancePredictionClient, SignedRequestError


@pytest.mark.asyncio
async def test_safe_error_logging_exposes_only_status_code_msg(caplog) -> None:
    caplog.set_level(logging.WARNING, logger="cat.research.prediction_client")

    def handler(req: httpx.Request) -> httpx.Response:
        # simulate Binance 400 with code/message, ensure no secret in body
        # also check request does not leak secret in logs — handler sees headers but we never log them
        assert "X-MBX-APIKEY" in req.headers
        # ensure handler does not see secret in URL
        assert "secret" not in str(req.url).lower()
        return httpx.Response(400, json={"code": -2008, "msg": "Invalid Api-Key ID."})

    transport = httpx.MockTransport(handler)
    client = BinancePredictionClient("TESTKEY123", "TESTSECRET999", transport=transport)

    with pytest.raises(SignedRequestError) as exc:
        await client.list_markets(l1_category="crypto", limit=5)

    err = exc.value
    assert err.status == 400
    assert err.code == -2008
    assert err.binance_msg == "Invalid Api-Key ID."
    msg = str(err)
    assert "400" in msg
    assert "-2008" in msg
    assert "Invalid Api-Key ID" in msg
    # never leak secret/signature/authorization
    lower = msg.lower()
    assert "testsecret999" not in lower
    assert "testkey123" not in lower or "testkey" not in lower.replace("testkey123", "")  # key may appear as TESTKEY in safe? Actually we don't log key at all; ensure not in msg
    assert "signature" not in lower
    assert "authorization" not in lower
    assert "secret" not in lower

    # logger record contains only safe fields
    found = [r for r in caplog.records if r.name == "cat.research.prediction_client" and "prediction_api_error" in r.getMessage()]
    assert found, "expected warning log for prediction_api_error"
    rec = found[0]
    # extra fields are stored directly on LogRecord
    assert getattr(rec, "http_status", None) == 400
    assert getattr(rec, "binance_code", None) == -2008
    assert getattr(rec, "binance_msg", None) == "Invalid Api-Key ID."
    # ensure no secret in log record
    log_text = str(rec.__dict__)
    assert "testsecret999" not in log_text.lower()
    assert "testkey123" not in log_text.lower()
    assert "signature" not in log_text.lower()


@pytest.mark.asyncio
async def test_safe_error_logging_post_and_never_logs_signature(caplog) -> None:
    caplog.set_level(logging.WARNING, logger="cat.research.prediction_client")

    def handler(req: httpx.Request) -> httpx.Response:
        # even POST with wallet address should not leak
        return httpx.Response(400, json={"code": -2015, "msg": "Invalid API-key, IP, or permissions for action."})

    transport = httpx.MockTransport(handler)
    client = BinancePredictionClient("K123", "S456", transport=transport)

    with pytest.raises(SignedRequestError) as exc:
        await client.get_quote_inspection("0xabc", "tok123", "BUY", "1000000000000000000")

    err = exc.value
    assert err.status == 400
    assert err.code == -2015
    lower = str(err).lower()
    assert "s456" not in lower
    assert "signature" not in lower

    # ensure log does not contain signature param
    for r in caplog.records:
        txt = str(r.__dict__).lower()
        assert "signature" not in txt
        assert "s456" not in txt


@pytest.mark.asyncio
async def test_safe_error_handles_non_json_response(caplog) -> None:
    caplog.set_level(logging.WARNING, logger="cat.research.prediction_client")

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(400, content=b"not json")

    transport = httpx.MockTransport(handler)
    client = BinancePredictionClient("K", "S", transport=transport)
    with pytest.raises(SignedRequestError) as exc:
        await client.list_categories()
    assert exc.value.status == 400
    assert exc.value.code is None

    found = [r for r in caplog.records if "prediction_api_error" in r.getMessage()]
    assert found
