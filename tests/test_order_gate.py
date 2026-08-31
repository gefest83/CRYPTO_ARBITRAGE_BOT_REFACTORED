"""Order gates and mode policy: PAPER never places, DEMO sandbox-only, LIVE gated."""

from types import SimpleNamespace

import pytest
from app.config.settings import Settings
from app.errors import ConfigurationError
from app.execution.order_gate import live_session_gate, never_place_orders, sandbox_only
from pydantic import ValidationError


class _Adapter(SimpleNamespace):
    pass


def test_paper_gate_never_places():
    assert never_place_orders(_Adapter()) is False


def test_sandbox_only_gate_fails_closed_without_sandbox():
    adapter = _Adapter(options=_Adapter(sandbox=False))
    assert sandbox_only(adapter) is False
    sandboxed = _Adapter(options=_Adapter(sandbox=True))
    assert sandbox_only(sandboxed) is True
    # no options at all -> refuse
    assert sandbox_only(_Adapter()) is False


def test_live_gate_requires_enabled_guard():
    guard = _Adapter(trading_enabled=True, is_halted=False)
    gate = live_session_gate(guard)
    assert gate(_Adapter()) is True
    halted = _Adapter(trading_enabled=True, is_halted=True)
    assert live_session_gate(halted)(_Adapter()) is False
    disabled = _Adapter(trading_enabled=False, is_halted=False)
    assert live_session_gate(disabled)(_Adapter()) is False

    # a broken guard fails closed
    class Broken:
        @property
        def trading_enabled(self):
            raise RuntimeError("boom")

    assert live_session_gate(Broken())(_Adapter()) is False


def test_live_mode_requires_triple_opt_in():
    import os

    keys = (
        "CAT_TRADING__MODE",
        "CAT_TRADING__ALLOW_LIVE",
        "CAT_TRADING__LIVE_CONFIRMATION",
    )
    saved = {k: os.environ.get(k) for k in keys}
    try:
        # 1. allow_live missing -> configuration refused
        os.environ["CAT_TRADING__MODE"] = "LIVE"
        with pytest.raises(ValidationError):
            Settings(_env_file=None)
        # 2. allow_live without confirmation -> refused
        os.environ["CAT_TRADING__ALLOW_LIVE"] = "true"
        with pytest.raises(ValidationError):
            Settings(_env_file=None)
        # 3. all three flags -> accepted
        os.environ["CAT_TRADING__LIVE_CONFIRMATION"] = "I UNDERSTAND THE RISK"
        settings = Settings(_env_file=None)
        assert settings.mode.value == "LIVE"
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_allow_live_without_live_mode_is_refused():
    import os

    key = "CAT_TRADING__ALLOW_LIVE"
    saved = os.environ.get(key)
    try:
        os.environ[key] = "true"
        with pytest.raises(ValidationError):
            Settings(_env_file=None)
    finally:
        if saved is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = saved


def test_paper_mode_is_the_default():
    assert Settings(_env_file=None).mode.value == "PAPER"


def test_unknown_venue_is_refused():
    from app.exchanges.profiles import resolve_profile

    try:
        resolve_profile("kraken")
        raise AssertionError("kraken must not be a supported venue")
    except ConfigurationError as exc:
        assert "binance" in str(exc) and "okx" in str(exc) and "bybit" in str(exc)
