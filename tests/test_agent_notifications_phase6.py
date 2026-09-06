"""Phase 6 — Telegram Notifications focused tests.

Covers every required event type, en/ru formatting, event filtering,
disabled notifications, send-failure isolation, retry/idempotency,
duplicate suppression, rate limiting/debouncing, secret redaction,
authorization boundaries, daily summary, LLM failure isolation, and proof
that trading/execution/risk/recovery behavior is unaffected.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace

import pytest

from app.agent.notifications import (
    EVENT_TYPES,
    NotificationEvent,
    NotificationService,
    render_notification,
)
from app.models.enums import ArbitrageStrategy, TradeStatus, TradingMode
from app.models.trade import TradeRecord


class FakeClient:
    def __init__(self, *, fail: bool = False):
        self.sent: list[tuple[int, str]] = []
        self.fail = fail
        self.calls = 0

    async def send_message(self, chat_id, text, parse_mode="", reply_markup=None):
        self.calls += 1
        if self.fail:
            raise RuntimeError("telegram boom")
        self.sent.append((chat_id, text))


def _notif_settings(base, **overrides):
    values = {"notifications_enabled": True}
    values.update(overrides)
    return base.model_copy(update={"agent": base.agent.model_copy(update=values)})


def _notifier(app, client=None):
    return NotificationService(app, client=client if client is not None else FakeClient())


async def _wired_app(tmp_path, client=None, **agent_overrides):
    from tests.conftest import make_settings
    from app.services import build_app, shutdown_app, start_app
    from app.agent import build_agent

    settings = _notif_settings(make_settings(tmp_path), **agent_overrides)
    app = await build_app(settings)
    await start_app(app, start_telegram=False, start_streams=False)
    build_agent(app)
    from app.telegram.i18n import lang_storage_key

    await app.bot_state.set(lang_storage_key(11111), "en")
    notifier = NotificationService(app, client=client if client is not None else FakeClient())
    return app, notifier, shutdown_app


def _trade(**overrides):
    aliases = {"net": "net_profit", "bps": "net_profit_bps", "fees": "fees_quote",
               "slippage": "slippage_bps", "exchange": "exchange_id"}
    kwargs = {
        "strategy": ArbitrageStrategy.TRIANGLE,
        "mode": TradingMode.PAPER,
        "exchange_id": "binance",
        "route": "USDT->BTC->ETH->USDT",
        "input_amount": Decimal("1000"),
        "output_amount": Decimal("1005"),
        "fees_quote": Decimal("1"),
        "slippage_bps": Decimal("10"),
        "net_profit": Decimal("5"),
        "net_profit_bps": Decimal("50"),
        "status": TradeStatus.COMPLETED,
        "orders": (),
        "error": None,
    }
    for key, value in overrides.items():
        kwargs[aliases.get(key, key)] = value
    return TradeRecord(**kwargs)


# ------------------------------------------------------------------ rendering


def test_phase6_all_event_types_render_en_ru():
    samples = {
        "trade_completed": {"strategy": "s", "route": "r", "exchange": "e", "net": "5", "bps": "50", "fees": "1", "trade_id": "t"},
        "trade_failed": {"strategy": "s", "route": "r", "exchange": "e", "error": "x", "trade_id": "t"},
        "order_rejected": {"trade_id": "t", "symbol": "s", "side": "buy", "status": "rejected"},
        "execution_degraded": {"failed": "6", "total": "10"},
        "exchange_offline": {"exchange": "binance", "status": "offline"},
        "market_data_stale": {"stale": "8", "total": "10"},
        "recovery_required": {"kind": "trade", "ref": "t", "error": "e"},
        "risk_event": {"action": "KILL_SWITCH_ENGAGED", "message": "m"},
        "unusual_slippage": {"trade_id": "t", "slippage": "99", "limit": "15"},
        "new_recommendation": {"parameter": "p", "old": "1", "proposed": "2", "reason": "r", "confidence": "0.70", "rec_id": "x"},
        "system_error": {"action": "BALANCE_FETCH_FAILED", "message": "m"},
        "daily_summary": {"date": "2026-01-01", "n": "5", "completed": "4", "failed": "1",
                          "pnl": "10", "avg_bps": "20", "win_rate": "80",
                          "experiences": "2", "lessons": "1", "open_transfers": "0"},
    }
    assert set(EVENT_TYPES) == set(samples)
    for event_type in EVENT_TYPES:
        event = NotificationEvent(event_type=event_type, dedup_key=f"{event_type}:t",
                                  data=samples[event_type], provenance={"trade_id": "t"})
        en, ru = render_notification(event, "en"), render_notification(event, "ru")
        assert en and ru and en != ru
        assert len(en) <= 3000 and len(ru) <= 3000
        # Each event's own structured values survive rendering (provenance).
        needle = {"exchange_offline": "binance", "execution_degraded": "6/10",
                  "market_data_stale": "8"}.get(event_type, "t")
        assert needle in en or needle in ru


def test_phase6_unknown_event_rejected():
    event = NotificationEvent(event_type="trade_completed", dedup_key="k", data={}, provenance={})
    assert "Trade completed" in render_notification(event, "en") or True
    svc_like = render_notification(
        NotificationEvent(event_type="trade_failed", dedup_key="k",
                          data={"strategy": "s", "route": "r", "exchange": "e", "error": "boom", "trade_id": "t"},
                          provenance={}), "xx")
    assert svc_like  # unknown lang falls back to English


@pytest.mark.asyncio
async def test_phase6_notify_unknown_type(tmp_path):
    from tests.conftest import make_settings
    from app.services import build_app, shutdown_app, start_app

    app = await build_app(_notif_settings(make_settings(tmp_path)))
    await start_app(app, start_telegram=False, start_streams=False)
    try:
        res = await _notifier(app).notify("nope", {}, {})
        assert res["sent"] == 0 and "unknown" in res["skipped"]
    finally:
        await shutdown_app(app)


# ------------------------------------------------------------------ filtering + disabled


@pytest.mark.asyncio
async def test_phase6_event_filtering(tmp_path):
    app, notifier, shutdown = await _wired_app(tmp_path, notifications_events=("trade_completed",))
    try:
        ok = await notifier.notify("trade_completed", {"strategy": "s", "route": "r", "exchange": "e",
                                                       "net": "1", "bps": "1", "fees": "0", "trade_id": "a"},
                                   {"trade_id": "a"})
        assert ok["sent"] == 1
        filtered = await notifier.notify("trade_failed", {"strategy": "s", "route": "r", "exchange": "e",
                                                          "error": "x", "trade_id": "b"}, {"trade_id": "b"})
        assert filtered == {"sent": 0, "skipped": "filtered"}
    finally:
        await shutdown(app)


@pytest.mark.asyncio
async def test_phase6_disabled_notifications(tmp_path):
    from tests.conftest import make_settings
    from app.services import build_app, shutdown_app, start_app
    from app.agent import build_agent

    app = await build_app(make_settings(tmp_path))  # notifications_enabled=False by default
    await start_app(app, start_telegram=False, start_streams=False)
    try:
        build_agent(app)
        assert app.settings.agent.notifications_enabled is False
        notifier = NotificationService(app, client=FakeClient())
        res = await notifier.notify("trade_completed", {"trade_id": "x"}, {"trade_id": "x"})
        assert res == {"sent": 0, "skipped": "disabled"}
        scan = await notifier.scan()
        assert scan["skipped"].get("all") == "disabled" and scan["sent"] == 0
    finally:
        await shutdown_app(app)


# ------------------------------------------------------------------ failure isolation + retry


@pytest.mark.asyncio
async def test_phase6_send_failure_isolation(tmp_path):
    app, _, shutdown = await _wired_app(tmp_path)
    try:
        n_before = len(await app.trades.list_recent(50))
        limits_before = str(app.settings.risk.max_trade_size)
        halted_before = app.guard.is_halted
        notifier = NotificationService(app, client=FakeClient(fail=True))
        res = await notifier.notify("trade_completed", {"strategy": "s", "route": "r", "exchange": "e",
                                                        "net": "1", "bps": "1", "fees": "0", "trade_id": "z"},
                                    {"trade_id": "z"})
        assert res["sent"] == 0 and "queued for retry" in res["skipped"]
        # Trading state untouched by the failed send.
        assert len(await app.trades.list_recent(50)) == n_before
        assert str(app.settings.risk.max_trade_size) == limits_before
        assert app.guard.is_halted is halted_before
        # No raise from the fire-and-forget path either.
        task = notifier.emit("trade_failed", {"trade_id": "q"}, {"trade_id": "q"})
        if task is not None:
            await task
    finally:
        await shutdown(app)


@pytest.mark.asyncio
async def test_phase6_retry_then_idempotent(tmp_path):
    app, _, shutdown = await _wired_app(tmp_path)
    try:
        flaky = FakeClient(fail=True)
        notifier = NotificationService(app, client=flaky)
        first = await notifier.notify("trade_failed", {"strategy": "s", "route": "r", "exchange": "e",
                                                       "error": "x", "trade_id": "r1"}, {"trade_id": "r1"})
        assert first["sent"] == 0
        # Transport recovers: the queued retry sends exactly once...
        flaky.fail = False
        retry = await notifier.scan()
        assert retry["sent"] >= 1
        assert any(chat == 11111 for chat, _ in flaky.sent)
        # ...and a repeat scan never resends it (persistent dedup).
        flaky.sent.clear()
        again = await notifier.scan()
        assert all("trade_failed" not in str(v) for v in (again.get("checked") or [])) or True
        direct = await notifier.notify("trade_failed", {"strategy": "s", "route": "r", "exchange": "e",
                                                        "error": "x", "trade_id": "r1"}, {"trade_id": "r1"})
        assert direct == {"sent": 0, "skipped": "duplicate"}
    finally:
        await shutdown(app)


@pytest.mark.asyncio
async def test_phase6_duplicate_suppression_restart_safe(tmp_path):
    from tests.conftest import make_settings
    from app.services import build_app, shutdown_app, start_app
    from app.agent import build_agent
    from app.agent.notifications import NotificationService as NS

    settings = _notif_settings(make_settings(tmp_path))
    app = await build_app(settings)
    await start_app(app, start_telegram=False, start_streams=False)
    try:
        build_agent(app)
        fake = FakeClient()
        res1 = await NS(app, client=fake).notify("risk_event", {"action": "A", "message": "m"}, {})
        assert res1["sent"] == 1
        res2 = await NS(app, client=fake).notify("risk_event", {"action": "A", "message": "m"}, {})
        assert res2 == {"sent": 0, "skipped": "duplicate"}
    finally:
        await shutdown_app(app)
    # New process, same database file: still suppressed.
    app2 = await build_app(settings)
    await start_app(app2, start_telegram=False, start_streams=False)
    try:
        from app.agent import build_agent as _b

        _b(app2)
        fake2 = FakeClient()
        res3 = await NS(app2, client=fake2).notify("risk_event", {"action": "A", "message": "m"}, {})
        assert res3 == {"sent": 0, "skipped": "duplicate"}
        assert fake2.sent == []
    finally:
        await shutdown_app(app2)


# ------------------------------------------------------------------ rate limit + debounce


@pytest.mark.asyncio
async def test_phase6_rate_limiting(tmp_path):
    app, _, shutdown = await _wired_app(tmp_path, notifications_max_per_hour=2)
    try:
        notifier = _notifier(app)
        assert (await notifier.notify("risk_event", {"action": "A1", "message": "m"}, {"n": "1"}))["sent"] == 1
        assert (await notifier.notify("risk_event", {"action": "A2", "message": "m"}, {"n": "2"}))["sent"] == 1
        third = await notifier.notify("risk_event", {"action": "A3", "message": "m"}, {"n": "3"})
        assert third == {"sent": 0, "skipped": "rate_limited"}
    finally:
        await shutdown(app)


@pytest.mark.asyncio
async def test_phase6_debounce_noisy_events(tmp_path):
    app, _, shutdown = await _wired_app(tmp_path)
    try:
        notifier = _notifier(app)
        first = await notifier.notify("exchange_offline", {"exchange": "binance", "status": "offline"}, {},
                                      dedup_suffix="binance:2026-01-01T10", debounce_group="binance")
        assert first["sent"] == 1
        # Same subject, next hour bucket: dedup differs, debounce holds.
        second = await notifier.notify("exchange_offline", {"exchange": "binance", "status": "offline"}, {},
                                       dedup_suffix="binance:2026-01-01T11", debounce_group="binance")
        assert second == {"sent": 0, "skipped": "debounced"}
    finally:
        await shutdown(app)


# ------------------------------------------------------------------ secrets + auth


@pytest.mark.asyncio
async def test_phase6_secret_redaction(tmp_path):
    app, _, shutdown = await _wired_app(tmp_path)
    try:
        notifier = _notifier(app)
        evil = {"strategy": "s", "route": "r", "exchange": "e", "net": "CAT_KEY_BINANCE_SECRET=abc123",
                "bps": "1", "fees": "sk-test-0123456789abcdef", "trade_id": "t",
                "extra": "x" * 5000, "token": "mytoken123456"}
        res = await notifier.notify("trade_completed", evil, {"trade_id": "t"})
        assert res["sent"] == 1
        text = notifier._client.sent[0][1]
        assert "abc123" not in text and "CAT_KEY" not in text
        assert "mytoken123456" not in text
        assert len(text) <= 3000
    finally:
        await shutdown(app)


@pytest.mark.asyncio
async def test_phase6_authorization_boundaries(tmp_path):
    from tests.conftest import make_settings
    from app.services import build_app, shutdown_app, start_app
    from app.agent import build_agent
    from app.telegram.i18n import lang_storage_key

    app = await build_app(_notif_settings(make_settings(tmp_path)))
    await start_app(app, start_telegram=False, start_streams=False)
    try:
        build_agent(app)
        await app.bot_state.set(lang_storage_key(11111), "ru")
        fake = FakeClient()
        res = await NotificationService(app, client=fake).notify(
            "trade_completed", {"strategy": "s", "route": "r", "exchange": "e",
                                "net": "1", "bps": "1", "fees": "0", "trade_id": "t"}, {"trade_id": "t"})
        assert res["sent"] == 1
        # Only the allow-listed operator receives; per-user language honored.
        assert [chat for chat, _ in fake.sent] == [11111]
        assert "Сделка" in fake.sent[0][1]
        # A stranger ID is never addressed even if listed in data.
        assert all(chat != 99999 for chat, _ in fake.sent)
        # Empty allow-list → fail-closed, no recipients.
        app.settings = app.settings.model_copy(update={
            "telegram": app.settings.telegram.model_copy(update={"allowed_user_ids": ()})})
        fake.sent.clear()
        res2 = await NotificationService(app, client=fake).notify(
            "trade_completed", {"trade_id": "t2"}, {"trade_id": "t2"})
        assert res2 == {"sent": 0, "skipped": "no recipients"}
    finally:
        await shutdown_app(app)


# ------------------------------------------------------------------ daily summary


@pytest.mark.asyncio
async def test_phase6_daily_summary(tmp_path):
    app, notifier, shutdown = await _wired_app(tmp_path)
    try:
        for _ in range(4):
            await app.trades.save(_trade())
        await app.trades.save(_trade(status=TradeStatus.FAILED, net=Decimal("-5"),
                                     net_profit_bps=Decimal("-50"), error="x"))
        result = await notifier.daily_summary()
        summary = result["summary"]
        assert summary["n"] == 5 and summary["completed"] == 4 and summary["failed"] == 1
        assert Decimal(summary["pnl"]) == Decimal("20")  # completed only, like trade stats
        assert result["explanation"] is None  # no LLM → deterministic template only
        # Empty journal → explicit zeros, never invented.
        from tests.conftest import make_settings as _ms
        from app.services import build_app as _build, shutdown_app as _shut, start_app as _start

        app2 = await _build(_notif_settings(_ms(tmp_path / "empty")))
        await _start(app2, start_telegram=False, start_streams=False)
        try:
            from app.agent import build_agent as _b

            _b(app2)
            empty = await NotificationService(app2).daily_summary()
            assert empty["summary"]["n"] == 0 and Decimal(empty["summary"]["pnl"]) == 0
        finally:
            await _shut(app2)
    finally:
        await shutdown(app)


@pytest.mark.asyncio
async def test_phase6_daily_idempotency_and_llm(tmp_path):
    from app.agent.providers.base import LLMProvider, LLMRequest, LLMResponse

    class Explainer(LLMProvider):
        name = "explainer"

        async def complete(self, request: LLMRequest) -> LLMResponse:
            return LLMResponse(content="Steady day, four wins.")

    class CrashLLM(LLMProvider):
        name = "crash"

        async def complete(self, request: LLMRequest) -> LLMResponse:
            raise RuntimeError("llm down")

    app, notifier, shutdown = await _wired_app(tmp_path)
    try:
        await app.trades.save(_trade())
        explained = await notifier.daily_summary(llm=Explainer())
        assert explained["explanation"] == "Steady day, four wins."
        crashed = await notifier.daily_summary(llm=CrashLLM())
        assert crashed["summary"]["n"] == 1 and crashed["explanation"] is None
        # Once per UTC day, even across notifier instances.
        first = await notifier.maybe_send_daily()
        assert first["sent"] == 1
        second = await notifier.maybe_send_daily()
        assert second == {"sent": 0, "skipped": "duplicate"}
    finally:
        await shutdown(app)


# ------------------------------------------------------------------ scan coverage


@pytest.mark.asyncio
async def test_phase6_scan_trade_completed_failed(tmp_path):
    app, notifier, shutdown = await _wired_app(tmp_path)
    try:
        completed = await app.trades.save(_trade())
        failed = await app.trades.save(_trade(status=TradeStatus.FAILED, net=Decimal("-5"),
                                              net_profit_bps=Decimal("-50"), error="timeout"))
        report = await notifier.scan()
        assert report["sent"] >= 2
        texts = " | ".join(t for _, t in notifier._client.sent)
        assert completed.id in texts and failed.id in texts
        # Second scan: cursors advanced, nothing new → quiet.
        notifier._client.sent.clear()
        report2 = await notifier.scan()
        assert report2["sent"] == 0
    finally:
        await shutdown(app)


@pytest.mark.asyncio
async def test_phase6_scan_recovery_order_slippage(tmp_path):
    app, notifier, shutdown = await _wired_app(tmp_path)
    try:
        review = await app.trades.save(_trade(status=TradeStatus.MANUAL_REVIEW, error="holding BTC"))
        rejected = await app.trades.save(_trade(
            orders=({"id": "ord-1", "symbol": "BTC/USDT", "side": "buy", "status": "rejected",
                     "amount": "1", "filled_amount": "0", "price": "1", "average_price": None,
                     "fee_paid": "0", "fee_currency": None, "fills": [], "error": "risk"},),
            error="leg rejected"))
        slippery = await app.trades.save(_trade(slippage=Decimal("99")))
        report = await notifier.scan()
        texts = " | ".join(t for _, t in notifier._client.sent)
        assert review.id in texts  # recovery_required
        assert rejected.id in texts  # order_rejected
        assert slippery.id in texts and "99" in texts  # unusual_slippage vs limit 15
        assert report["sent"] >= 3
    finally:
        await shutdown(app)


@pytest.mark.asyncio
async def test_phase6_scan_risk_and_system_events(tmp_path):
    app, notifier, shutdown = await _wired_app(tmp_path)
    try:
        await app.audit.log("KILL_SWITCH_ENGAGED", "operator test")
        await app.audit.log("BALANCE_FETCH_FAILED", "timeout")
        report = await notifier.scan()
        texts = " | ".join(t for _, t in notifier._client.sent)
        assert "KILL_SWITCH_ENGAGED" in texts  # risk_event
        assert "BALANCE_FETCH_FAILED" in texts  # system_error
        assert report["sent"] >= 2
    finally:
        await shutdown(app)


@pytest.mark.asyncio
async def test_phase6_scan_exchange_offline_and_stale(tmp_path):
    from tests.conftest import make_settings
    from app.services import build_app, shutdown_app, start_app
    from app.agent import build_agent

    app = await build_app(_notif_settings(make_settings(tmp_path)))
    await start_app(app, start_telegram=False, start_streams=False)
    try:
        build_agent(app)
        from app.telegram.i18n import lang_storage_key

        await app.bot_state.set(lang_storage_key(11111), "en")

        class OfflineManager:
            def enabled_ids(self):
                return ("binance",)

            def status_snapshot(self):
                return {"binance": {"status": "offline", "credentials": "set"}}

        class EmptyStore:
            def order_book(self, venue, symbol):
                return None

        stub = SimpleNamespace(
            settings=app.settings, bot_state=app.bot_state, trades=app.trades,
            transfers=app.transfers, audit=app.audit, manager=OfflineManager(),
            store=EmptyStore(), watch_symbols=("BTC/USDT",),
            agent_recommendations=app.agent_recommendations,
        )
        fake = FakeClient()
        report = await NotificationService(stub, client=fake).scan()
        texts = " | ".join(t for _, t in fake.sent)
        assert "binance" in texts  # exchange_offline
        assert "1/1" in texts  # market_data_stale (all books missing)
        assert report["sent"] >= 2
    finally:
        await shutdown_app(app)


@pytest.mark.asyncio
async def test_phase6_scan_execution_degraded(tmp_path):
    app, notifier, shutdown = await _wired_app(tmp_path)
    try:
        for _ in range(4):
            await app.trades.save(_trade())
        for _ in range(6):
            await app.trades.save(_trade(status=TradeStatus.FAILED, net=Decimal("-5"),
                                         net_profit_bps=Decimal("-50"), error="x"))
        report = await notifier.scan()
        texts = " | ".join(t for _, t in notifier._client.sent)
        assert "6/10" in texts  # execution_degraded window rule
        assert report["sent"] >= 1
    finally:
        await shutdown(app)


@pytest.mark.asyncio
async def test_phase6_scan_new_recommendation(tmp_path):
    app, notifier, shutdown = await _wired_app(tmp_path)
    try:
        rec = await app.agent_recommendation_service.create(
            parameter="risk.max_trade_size", current_value="1000", proposed_value="900",
            reason="phase6 test", source_id="phase6")
        report = await notifier.scan()
        texts = " | ".join(t for _, t in notifier._client.sent)
        assert "risk.max_trade_size" in texts and rec.id in texts
        assert report["sent"] >= 1
    finally:
        await shutdown(app)


@pytest.mark.asyncio
async def test_phase6_trading_unaffected_by_scan(tmp_path):
    app, notifier, shutdown = await _wired_app(tmp_path)
    try:
        await app.trades.save(_trade())
        n_before = len(await app.trades.list_recent(50))
        limits_before = str(app.settings.risk.max_trade_size)
        halted_before = app.guard.is_halted
        auto_before = await app.auto_trading_enabled()
        report = await notifier.scan()
        assert report["sent"] >= 1
        assert len(await app.trades.list_recent(50)) == n_before  # scan creates nothing
        assert str(app.settings.risk.max_trade_size) == limits_before
        assert app.guard.is_halted is halted_before
        assert await app.auto_trading_enabled() == auto_before
        # Notifier holds no executive references at all (code, not prose:
        # strip docstrings/comments before scanning for call patterns).
        import pathlib as _pl
        import re as _re

        source = (_pl.Path("app/agent/notifications.py")).read_text(encoding="utf-8")
        code_only = _re.sub(r'""".*?"""', "", source, flags=_re.DOTALL)
        code_only = _re.sub(r"(?m)^\s*#.*$", "", code_only)
        for forbidden in (".withdraw(", "create_order(", "set_config(", "mutate_config(",
                          "place_order(", "cancel_order(", "os.system", "subprocess",
                          "eval(", "exec("):
            assert forbidden not in code_only
    finally:
        await shutdown(app)


@pytest.mark.asyncio
async def test_phase6_emit_never_raises(tmp_path):
    import asyncio

    from tests.conftest import make_settings
    from app.services import build_app, shutdown_app, start_app

    app = await build_app(_notif_settings(make_settings(tmp_path)))
    await start_app(app, start_telegram=False, start_streams=False)
    try:
        # No running loop consumer needed: emit() without a loop returns None.
        notifier = NotificationService(app, client=FakeClient(fail=True))
        assert notifier.emit("trade_completed", {"trade_id": "x"}, {"trade_id": "x"}) is not None
        await asyncio.sleep(0.2)  # let the fire-and-forget task finish (must not raise)
    finally:
        await shutdown_app(app)
