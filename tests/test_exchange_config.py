"""H-11: the ccxt adapter's configuration structure must never leak secrets.

The config dict handed to the ccxt constructor carries real credentials
(ccxt needs them), but any textual rendering of the structure itself —
``str()``, ``repr()``, f-strings, ``%``-formatting, logging — must show
``<redacted>`` instead.  Authentication behaviour must be preserved.
"""

from __future__ import annotations

import logging

import pytest

from app.exchanges.base import AdapterOptions
from app.exchanges.ccxt_adapter import CCXTAdapter
from app.exchanges.credentials import ExchangeCredentials
from app.models.exchange import Exchange

API_KEY = "binance-api-key-123456"
SECRET = "the-real-secret-abcdef7890"
PASSWORD = "okx-passphrase-654321"


def _adapter() -> CCXTAdapter:
    exchange = Exchange(id="okx", name="OKX", adapter="ccxt")
    credentials = ExchangeCredentials(
        api_key=API_KEY, secret=SECRET, password=PASSWORD, uid="uid-42"
    )
    return CCXTAdapter(exchange, credentials=credentials, options=AdapterOptions())


def test_base_config_redacts_secrets_in_every_text_form():
    adapter = _adapter()
    config = adapter._base_config()
    texts = [
        str(config),
        repr(config),
        f"{config}",
        f"{config!r}",
        "%s" % config,
        "{}".format(config),
    ]
    for text in texts:
        assert API_KEY not in text, text
        assert SECRET not in text, text
        assert PASSWORD not in text, text
        assert "<redacted>" in text


def test_base_config_logging_cannot_expose_secrets(caplog):
    adapter = _adapter()
    config = adapter._base_config()
    with caplog.at_level(logging.DEBUG):
        logging.getLogger("test.redaction").warning("built config %s", config)
        logging.getLogger("test.redaction").warning(f"built config {config}")
    joined = caplog.text
    assert API_KEY not in joined
    assert SECRET not in joined
    assert PASSWORD not in joined


def test_base_config_still_carries_real_values_for_ccxt():
    """Authentication behaviour preserved: the dict API returns the real
    credentials exactly as ccxt's constructor consumes them."""
    adapter = _adapter()
    config = adapter._base_config()
    assert config["apiKey"] == API_KEY
    assert config["secret"] == SECRET
    assert config["password"] == PASSWORD
    assert config["uid"] == "uid-42"
    assert config["enableRateLimit"] is True
    # plain-dict semantics are untouched
    assert dict(config) == {
        "enableRateLimit": True,
        "timeout": 10_000,
        "apiKey": API_KEY,
        "secret": SECRET,
        "password": PASSWORD,
        "uid": "uid-42",
    }


def test_config_authenticates_a_real_ccxt_client():
    """End to end: a real ccxt client built from the redacting config still
    receives the credentials (deep_extend copies the raw values out)."""
    ccxt = pytest.importorskip("ccxt")
    adapter = _adapter()
    config = dict(adapter._base_config())
    client = ccxt.binance(config)
    assert client.apiKey == API_KEY
    assert client.secret == SECRET
