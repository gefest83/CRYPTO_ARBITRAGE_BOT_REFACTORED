"""DEMO-only preflight: fail-closed checks before any real DEMO trading.

For each enabled venue in DEMO mode:
- credentials exist
- fetch_balances succeeds and free USDT >= minimum notional
- fetch_trading_fees succeeds (or fallback logged)
- ticker + order book for BTC/USDT available

Missing/invalid credentials → mark DATA_ONLY (manager.record_auth_failure) and
fail closed for trading on that venue only — never crash the whole app.

No orders are placed in preflight.
"""

from __future__ import annotations

from decimal import Decimal

from app.config.logging_config import get_logger
from app.models.enums import TradingMode
from app.models.symbol import Symbol

logger = get_logger("exchanges.preflight")


async def run_demo_preflight(manager, settings, store=None) -> dict[str, dict]:
    """DEMO-only, best-effort preflight. Returns per-venue results.

    Never raises — one venue failing never blocks the others.
    """
    if settings.mode is not TradingMode.DEMO:
        return {}

    results: dict[str, dict] = {}
    # Minimum USDT required to be considered fundable for triangular
    # Use the most restrictive of the configured notionals
    min_notional = min(
        settings.arbitrage.triangle_max_notional_quote,
        settings.arbitrage.default_notional_quote,
        settings.execution.auto_notional_quote,
        settings.risk.max_trade_size,
    )
    base_ccy = settings.trading.base_currency
    probe_symbol = Symbol(base="BTC", quote=base_ccy)

    for venue in manager.enabled_ids():
        venue_result: dict = {
            "credentials": "unknown",
            "balance": "unknown",
            "free_usdt": None,
            "fees": "unknown",
            "ticker": "unknown",
            "order_book": "unknown",
            "ready": False,
            "reason": None,
        }
        try:
            # 1. credentials
            has_creds = False
            try:
                has_creds = manager._credentials.has(venue)  # type: ignore[attr-defined]
            except Exception:
                has_creds = False
            # Also check adapter's view
            try:
                adapter = manager.adapter(venue)
                if getattr(adapter, "has_credentials", False):
                    has_creds = True
            except Exception:
                pass
            if not has_creds:
                venue_result["credentials"] = "missing"
                venue_result["reason"] = "missing credentials"
                manager.record_auth_failure(venue, "demo preflight: missing credentials")
                logger.warning(
                    "demo_preflight_missing_credentials", extra={"exchange_id": venue}
                )
                results[venue] = venue_result
                continue
            venue_result["credentials"] = "ok"

            adapter = manager.adapter(venue)

            # 2. balances
            try:
                snapshot = await adapter.fetch_balances()
                venue_result["balance"] = "ok"
                free = snapshot.free_of(base_ccy)
                venue_result["free_usdt"] = str(free)
                if free < min_notional:
                    venue_result["balance"] = f"insufficient ({free} < {min_notional})"
                    venue_result["reason"] = f"insufficient {base_ccy}: {free} < {min_notional}"
                    logger.warning(
                        "demo_preflight_insufficient_balance",
                        extra={"exchange_id": venue, "free": str(free), "min_notional": str(min_notional)},
                    )
                    # Not marking DATA_ONLY for insufficient funds — just not ready
                else:
                    manager.record_private_success(venue)
            except Exception as exc:
                from app.errors import VenueAuthError

                if isinstance(exc, VenueAuthError) or "auth" in str(type(exc).__name__).lower():
                    venue_result["balance"] = f"auth_failed: {exc}"
                    venue_result["reason"] = str(exc)[:200]
                    manager.record_auth_failure(venue, f"demo preflight balance auth failed: {exc}")
                else:
                    venue_result["balance"] = f"failed: {type(exc).__name__}"
                    venue_result["reason"] = str(exc)[:200]
                    logger.warning(
                        "demo_preflight_balance_failed",
                        extra={"exchange_id": venue, "error": str(exc)[:200]},
                    )
                results[venue] = venue_result
                continue

            # 3. fees
            try:
                fees = await adapter.fetch_trading_fees(probe_symbol)
                venue_result["fees"] = f"ok taker {fees.taker_bps}bps"
                if fees.is_account_specific:
                    venue_result["fees"] += " (account)"
                else:
                    venue_result["fees"] += " (market)"
            except Exception as exc:
                venue_result["fees"] = f"fallback 10bps ({type(exc).__name__})"
                logger.warning(
                    "demo_preflight_fee_fallback",
                    extra={"exchange_id": venue, "error": str(exc)[:160], "fallback_bps": "10"},
                )

            # 4. ticker
            try:
                ticker = await adapter.fetch_ticker(probe_symbol)
                if ticker.bid is None and ticker.ask is None:
                    venue_result["ticker"] = "empty"
                else:
                    venue_result["ticker"] = "ok"
            except Exception as exc:
                venue_result["ticker"] = f"failed: {type(exc).__name__}"
                venue_result["reason"] = str(exc)[:200]

            # 5. order book
            try:
                book = await adapter.fetch_order_book(probe_symbol)
                if not book.bids or not book.asks:
                    venue_result["order_book"] = "empty"
                else:
                    venue_result["order_book"] = "ok"
            except Exception as exc:
                venue_result["order_book"] = f"failed: {type(exc).__name__}"
                venue_result["reason"] = (venue_result["reason"] or "") + f" | book {exc}"

            # Ready if core checks passed
            venue_result["ready"] = (
                venue_result["credentials"] == "ok"
                and venue_result["balance"] == "ok"
                and venue_result["ticker"] == "ok"
                and venue_result["order_book"] == "ok"
            )
            if venue_result["ready"]:
                logger.info("demo_preflight_ready", extra={"exchange_id": venue})
            else:
                logger.warning("demo_preflight_not_ready", extra={"exchange_id": venue, "result": str(venue_result)})

        except Exception as exc:  # noqa: BLE001 - isolate per venue
            venue_result["reason"] = f"unexpected {type(exc).__name__}: {exc}"[:300]
            logger.warning("demo_preflight_venue_failed", extra={"exchange_id": venue, "error": str(exc)[:300]})
        results[venue] = venue_result

    return results
