"""Transfer arbitrage orchestrator: a persisted, restart-safe workflow.

This is NOT a blocking function.  The lifecycle is driven step by step:

    CREATED -> BUY_SUBMITTED -> BUY_FILLED -> WITHDRAW_SUBMITTED ->
    WITHDRAW_PENDING -> TRANSFER_IN_PROGRESS -> DEPOSIT_DETECTED ->
    SELL_SUBMITTED -> COMPLETED

Every step persists its result before the next one runs, so the bot can be
restarted mid-transfer and :meth:`TransferOrchestrator.resume` picks the
workflow up exactly where it stopped (``tick`` keeps advancing open records).

Mode behaviour:

* PAPER — buy/sell legs are simulated against live books; the blockchain leg
  is simulated with a deterministic delay; funds move between per-venue paper
  wallets.
* DEMO — buy/sell legs are real orders on the venue's demo/testnet
  environment; the blockchain leg is simulated (testnets cannot move real
  funds between exchanges) and clearly flagged as such.
* LIVE — every step is real, including the withdrawal (which itself requires
  the separate ``CAT_TRADING__ALLOW_LIVE_WITHDRAWALS`` opt-in on the adapter).

Failure policy: understood errors (rejected buy, disabled network, plan below
minimum) fail the record with the reason; uncertain states (withdrawal stuck
past the deposit timeout, sell leg unrecoverable) escalate to MANUAL_REVIEW.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from decimal import Decimal

from app.config.logging_config import get_logger
from app.config.settings import Settings
from app.errors import ExecutionDisabledError
from app.execution.fill_simulator import FillSimulator, SimulatedFill
from app.execution.guard import ExecutionGuard
from app.execution.paper_wallet import PaperWallet
from app.models.base import DEC0, utc_now
from app.models.enums import OrderSide, OrderStatus, OrderType, TradingMode, TransferState
from app.models.order import Order, OrderRequest
from app.models.risk import RiskAssessment
from app.models.symbol import Symbol
from app.models.transfer import TransferPlan, TransferRecord
from app.recovery import ExecutionRecovery
from app.storage.repositories import (
    AuditLogRepository,
    TradeRepository,
    TransferRepository,
)
from app.strategies.transfer.networks import NetworkMismatchError, NetworkSelector
from app.strategies.transfer.planner import TransferPlanner, price_pair

__all__ = ["TransferOrchestrator"]

logger = get_logger("strategies.transfer.orchestrator")

_QUANTUM = Decimal("0.00000001")

#: Prefix of a persisted withdrawal submission intent (H-2).  A restart that
#: finds this id knows a withdrawal was ATTEMPTED but its acceptance could
#: not be confirmed — it fails closed instead of resubmitting.
_WITHDRAWAL_INTENT_PREFIX = "wd-intent-"

#: Assets whose on-chain transfers always require a destination tag / memo.
#: Withdrawals to an address on these networks WITHOUT the tag are sent to an
#: uncredited address and the funds are lost (the venue credits them, but
#: they cannot be routed to a user account).  The set is intentionally
#: conservative: it covers the well-known tag-required assets the bot may
#: encounter.  When in doubt, fail closed.
_MEMO_REQUIRED_ASSETS: frozenset[str] = frozenset(
    {
        "XRP",
        "XLM",
        "EOS",
        "HBAR",
        "XIN",
        "RVN",
        "ATOM",
        "BNB",
        "LUNA",
        "LUNC",
        "KAVA",
        "CRO",
        "BTS",
    }
)


class TransferOrchestrator:
    """Plans, starts and advances transfer arbitrage workflows."""

    def __init__(
        self,
        *,
        settings: Settings,
        manager,
        store,
        guard: ExecutionGuard,
        recovery: ExecutionRecovery,
        fill_simulator: FillSimulator,
        planner: TransferPlanner,
        transfer_repo: TransferRepository,
        trade_repo: TradeRepository,
        audit: AuditLogRepository,
        risk_check: Callable[[TransferPlan], RiskAssessment] | None = None,
        paper_wallets: dict[str, PaperWallet] | None = None,
        market=None,
    ) -> None:
        self._settings = settings
        self._manager = manager
        self._store = store
        self._guard = guard
        self._recovery = recovery
        self._sim = fill_simulator
        self._planner = planner
        self._transfers = transfer_repo
        self._trades = trade_repo
        self._audit = audit
        self._risk_check = risk_check
        self._paper_wallets = paper_wallets or {}
        self._selector = NetworkSelector()
        self._mode = settings.mode
        self._market = market

    # ---------------------------------------------------------------- planning
    async def plan(
        self,
        *,
        asset: str | None = None,
        amount: Decimal | None = None,
        source: str | None = None,
        dest: str | None = None,
    ) -> list[TransferPlan]:
        """Scan venues for profitable transfer plans (no execution)."""
        assets = [asset.strip().upper()] if asset else list(self._settings.transfer.assets)
        venues = list(self._manager.enabled_ids())
        sources = [source.strip().lower()] if source else venues
        dests = [dest.strip().lower()] if dest else venues
        plans: list[TransferPlan] = []
        for asset_code in assets:
            symbol = Symbol(base=asset_code, quote=self._settings.trading.base_currency)
            for source_id in sources:
                for dest_id in dests:
                    if source_id == dest_id:
                        continue
                    plan = await self._plan_pair(symbol, source_id, dest_id, amount)
                    if plan is not None:
                        plans.append(plan)
        plans.sort(key=lambda p: p.net_profit_bps, reverse=True)
        return plans

    async def _plan_pair(
        self,
        symbol: Symbol,
        source_id: str,
        dest_id: str,
        amount: Decimal | None,
    ) -> TransferPlan | None:
        asset = symbol.base
        try:
            buy_book = await self._fresh_book(source_id, symbol)
            sell_book = await self._fresh_book(dest_id, symbol)
        except ValueError:
            return None
        try:
            source_networks = await self._manager.adapter(source_id).fetch_withdrawal_networks(
                asset
            )
            dest_networks = await self._manager.adapter(dest_id).fetch_withdrawal_networks(asset)
            route = self._selector.select(
                asset=asset, source_networks=source_networks, dest_networks=dest_networks
            )
        except NetworkMismatchError as exc:
            logger.debug(
                "transfer_network_mismatch",
                extra={"asset": asset, "from": source_id, "to": dest_id, "reason": str(exc)},
            )
            return None
        except Exception as exc:  # noqa: BLE001 - unknown fees -> uncertain profit
            logger.debug(
                "transfer_network_lookup_failed",
                extra={"asset": asset, "from": source_id, "to": dest_id, "error": str(exc)[:160]},
            )
            return None

        # Layer A+B market-data sanity guard (pure, no I/O beyond books)
        from app.strategies.transfer.planner import validate_transfer_books

        guard_reason = validate_transfer_books(
            buy_book,
            sell_book,
            max_gross_divergence_bps=self._settings.transfer.max_transfer_gross_divergence_bps,
        )
        if guard_reason is not None:
            logger.debug(
                "transfer_sanity_rejected",
                extra={
                    "asset": asset,
                    "from": source_id,
                    "to": dest_id,
                    "reason": guard_reason,
                    "buy_ask": str(buy_book.best_ask) if buy_book.best_ask is not None else None,
                    "sell_bid": str(sell_book.best_bid) if sell_book.best_bid is not None else None,
                },
            )
            return None

        available_quote = await self._available_quote(source_id)
        buy_fees = await self._taker_fees(source_id, symbol)
        sell_fees = await self._taker_fees(dest_id, symbol)

        # Sizing: explicit `amount` (coin) overrides notional; otherwise
        # derive coin amount from USDT notional $100-500 capped by risk and
        # available quote.
        if amount is not None:
            requested = amount
            max_notional = min(
                self._settings.transfer.max_notional_quote,
                self._settings.risk.max_trade_size,
            )
            probe = max(requested, route.withdrawal_min)
            prices = price_pair(buy_book, sell_book, probe)
            executable = self._planner.executable_amount(
                requested_amount=requested,
                withdrawal_min=route.withdrawal_min,
                available_quote=available_quote,
                buy_price=prices.buy_price,
                max_amount=self._settings.transfer.max_amount,
                max_notional=max_notional,
            )
            if executable <= DEC0:
                return None
            if executable != probe:
                prices = price_pair(buy_book, sell_book, executable)
        else:
            min_notional = self._settings.transfer.min_notional_quote
            max_notional = min(
                self._settings.transfer.max_notional_quote,
                self._settings.risk.max_trade_size,
            )
            # Cap by available quote before deriving amount
            max_notional = min(max_notional, available_quote)
            if max_notional < min_notional:
                return None
            # Estimate amount from max notional using best ask as proxy
            est_price = buy_book.asks[0].price if buy_book.asks else DEC0
            if est_price <= DEC0:
                return None
            est_amount = (max_notional / est_price).quantize(_QUANTUM)
            est_amount = max(est_amount, route.withdrawal_min)
            prices = price_pair(buy_book, sell_book, est_amount)
            executable = self._planner.executable_amount_from_notional(
                min_notional=min_notional,
                max_notional=max_notional,
                available_quote=available_quote,
                buy_price=prices.buy_price,
                withdrawal_min=route.withdrawal_min,
                max_amount=self._settings.transfer.max_amount,
            )
            if executable <= DEC0:
                return None
            if executable != est_amount:
                prices = price_pair(buy_book, sell_book, executable)
                # Re-validate after repricing (buy_price may have moved)
                executable = self._planner.executable_amount_from_notional(
                    min_notional=min_notional,
                    max_notional=max_notional,
                    available_quote=available_quote,
                    buy_price=prices.buy_price,
                    withdrawal_min=route.withdrawal_min,
                    max_amount=self._settings.transfer.max_amount,
                )
                if executable <= DEC0:
                    return None
                if executable != est_amount:
                    prices = price_pair(buy_book, sell_book, executable)
        plan = self._planner.build(
            source_exchange=source_id,
            dest_exchange=dest_id,
            asset=asset,
            network=route.network,
            amount=executable,
            prices=prices,
            buy_fee_bps=buy_fees,
            sell_fee_bps=sell_fees,
            withdrawal_fee=route.withdrawal_fee,
        )
        # H-7: carry the real age of the pricing data into risk validation
        # (the freshest of the two books that priced the plan).
        plan = plan.model_copy(
            update={
                "data_age_ms": max(
                    self._store.age_ms(buy_book),
                    self._store.age_ms(sell_book),
                )
            }
        )
        return self._planner.evaluate(plan)

    # ---------------------------------------------------------------- start
    async def start(self, plan: TransferPlan) -> TransferRecord:
        """Validate and start a transfer workflow: buy leg + withdrawal.

        Cross-process safety (TEST GAP #7): the open-transfer count and
        the initial insert are wrapped in a single atomic claim at the
        database level (``BEGIN IMMEDIATE``), so independent OS processes
        racing on the same database file are serialised by the SQLite
        write lock.  At most one of them can observe ``count < max_open``
        and insert a new open row, making ``MAX_OPEN_TRANSFERS`` a real
        database-level invariant rather than an in-process Python lock.
        """
        self._guard.ensure_can_trade()
        if self._risk_check is not None:
            assessment = self._risk_check(plan)
            if not assessment.approved:
                reasons = "; ".join(f"{v.rule}: {v.message}" for v in assessment.violations)
                raise ExecutionDisabledError(
                    f"transfer rejected by risk validation: {reasons}",
                    mode=str(self._mode),
                )

        record = TransferRecord(
            source_exchange=plan.source_exchange,
            dest_exchange=plan.dest_exchange,
            asset=plan.asset,
            network=plan.network,
            amount=plan.amount,
            plan=plan,
            mode=str(self._mode),
        )
        # Atomic claim: the open-transfer count and the insert happen
        # inside one BEGIN IMMEDIATE transaction, so two concurrent
        # processes cannot both observe count < max_open and both
        # insert.  The MaxOpenTransfersRule above already enforced the
        # limit at validation time, but that check is in-process; the
        # atomic claim is the database-level guarantee.
        max_open = int(self._settings.risk.max_open_transfers)
        claimed = await self._transfers.try_claim_new_open_slot(
            record, max_open=max_open
        )
        if not claimed:
            raise ExecutionDisabledError(
                f"transfer rejected by risk validation: "
                f"max_open_transfers: {max_open} open transfers already "
                f"in flight (cross-process slot full)",
                mode=str(self._mode),
            )
        await self._audit.log(
            "TRANSFER_STARTED",
            f"{plan.source_exchange} -> {plan.dest_exchange} {plan.asset} "
            f"{plan.amount} via {plan.network}",
            {"transfer_id": record.id, "expected_net_bps": str(plan.net_profit_bps)},
        )
        record = await self._execute_buy(record)
        if record.state is TransferState.BUY_FILLED:
            record = await self._submit_withdrawal(record)
        return record

    # ---------------------------------------------------------------- ticking
    async def tick(self) -> list[TransferRecord]:
        """Advance every open transfer workflow by one step."""
        advanced: list[TransferRecord] = []
        for record in await self._transfers.list_open():
            try:
                updated = await self._advance(record)
            except Exception as exc:  # noqa: BLE001 - isolate per transfer
                logger.error(
                    "transfer_tick_error",
                    extra={"transfer_id": record.id, "error": str(exc)[:300]},
                )
                updated = record.with_state(
                    TransferState.MANUAL_REVIEW,
                    error=f"tick error: {type(exc).__name__}: {exc}"[:500],
                )
                await self._transfers.save(updated)
            if updated is not record:
                advanced.append(updated)
        return advanced

    async def resume(self) -> int:
        """Load open transfers on startup; returns how many resumed."""
        open_records = await self._transfers.list_open()
        if open_records:
            logger.info(
                "transfers_resumed",
                extra={"count": len(open_records)},
            )
        return len(open_records)

    async def _advance(self, record: TransferRecord) -> TransferRecord:
        state = record.state
        if state is TransferState.CREATED:
            return await self._execute_buy(record)
        if state is TransferState.BUY_SUBMITTED:
            return await self._execute_buy(record)
        if state is TransferState.BUY_FILLED:
            return await self._submit_withdrawal(record)
        if state is TransferState.WITHDRAW_SUBMITTED:
            return await self._check_withdrawal(record)
        if state is TransferState.WITHDRAW_PENDING:
            return await self._check_withdrawal(record)
        if state is TransferState.TRANSFER_IN_PROGRESS:
            return await self._await_deposit(record)
        if state is TransferState.DEPOSIT_DETECTED:
            return await self._execute_sell(record)
        if state is TransferState.SELL_SUBMITTED:
            return await self._finish(record)
        return record

    # ---------------------------------------------------------------- steps
    async def _execute_buy(self, record: TransferRecord) -> TransferRecord:
        """BUY leg on the source venue.  CREATED/BUY_SUBMITTED -> BUY_FILLED|FAILED.

        H-2 idempotency: if a previous attempt already persisted a buy order
        (or a pre-submission intent), it is resolved against the venue — a
        second BUY order is never created merely because the process
        restarted.
        """
        self._guard.ensure_can_trade()
        plan = record.plan
        symbol = Symbol(base=plan.asset, quote=self._settings.trading.base_currency)
        if record.buy_order is not None:
            return await self._resolve_existing_buy(record, symbol)

        async def _note(request: OrderRequest) -> None:
            nonlocal record
            # Persist the submission intent (client_order_id) BEFORE the
            # order exists on the venue: a crash between submission and the
            # outcome save is then recoverable by client-id lookup.
            placeholder = Order(
                exchange_id=request.exchange_id,
                symbol=request.symbol,
                side=request.side,
                order_type=request.order_type,
                amount=request.amount,
                client_order_id=request.client_order_id,
                status=OrderStatus.PENDING,
            )
            record = record.with_state(
                TransferState.BUY_SUBMITTED,
                buy_order=_order_to_json(placeholder),
            )
            await self._transfers.save(record)

        try:
            fill, order = await self._fill(
                plan.source_exchange,
                symbol,
                OrderSide.BUY,
                quote_amount=plan.buy_cost_quote,
                on_submit=_note,
            )
        except ExecutionDisabledError:
            raise
        except Exception as exc:  # noqa: BLE001
            if record.buy_order is not None:
                # The submission intent was persisted, so an order may exist
                # on the venue — never fail as a plain rejection.
                return await self._fail(
                    record,
                    f"buy leg error after submission attempt: {exc} — order may "
                    "exist on venue; manual review required",
                    state=TransferState.MANUAL_REVIEW,
                )
            return await self._fail(record, f"buy leg error: {exc}")
        record = record.with_state(
            TransferState.BUY_SUBMITTED,
            buy_order=_order_to_json(order),
            buy_filled_amount=order.filled_amount,
        )
        await self._transfers.save(record)
        if order.confirmed_not_filled:
            return await self._fail(record, f"buy leg rejected: {order.error or 'no fill'}")
        if order.outcome_unconfirmed:
            return await self._fail(
                record,
                f"buy leg outcome unconfirmed ({order.status.value}"
                f"{f': {order.error}' if order.error else ''}) — order may have "
                "filled; manual review required",
                state=TransferState.MANUAL_REVIEW,
            )
        # Paper fills resolve synchronously; venue orders above resolved
        # through recovery already.
        filled = order.filled_amount
        if filled <= DEC0 and fill is not None:
            filled = fill.filled_amount
        if filled <= DEC0:
            return await self._fail(record, "buy leg produced no fill")
        record = record.with_state(
            TransferState.BUY_FILLED,
            buy_filled_amount=filled.quantize(_QUANTUM),
        )
        await self._transfers.save(record)
        return record

    async def _resolve_existing_buy(
        self, record: TransferRecord, symbol: Symbol
    ) -> TransferRecord:
        """Resolve a buy order (or intent) persisted by a previous attempt."""
        order = _order_from_json(record.buy_order)
        if order is None:
            return await self._fail(
                record,
                "persisted buy order could not be parsed — manual review required",
                state=TransferState.MANUAL_REVIEW,
            )
        if order.is_confirmed_fill:
            # Idempotent replay: the previous attempt already bought the asset.
            record = record.with_state(
                TransferState.BUY_FILLED,
                buy_filled_amount=order.filled_amount.quantize(_QUANTUM),
            )
            await self._transfers.save(record)
            return record
        if order.confirmed_not_filled:
            # The venue established the order will not fill — no asset was
            # bought, so failing closed here loses nothing.
            return await self._fail(
                record,
                f"buy leg confirmed not filled ({order.status.value}) on a "
                "previous attempt",
            )
        # Unresolved (pre-submission intent, timeout, still open, ...): query
        # the venue through recovery — never place a second order.
        adapter = self._manager.adapter(record.plan.source_exchange)
        recovered = await self._recovery.recover(order, adapter)
        record = record.with_state(
            TransferState.BUY_SUBMITTED,
            buy_order=_order_to_json(recovered),
            buy_filled_amount=recovered.filled_amount,
        )
        await self._transfers.save(record)
        if recovered.is_confirmed_fill:
            record = record.with_state(
                TransferState.BUY_FILLED,
                buy_filled_amount=recovered.filled_amount.quantize(_QUANTUM),
            )
            await self._transfers.save(record)
            return record
        if recovered.confirmed_not_filled:
            return await self._fail(
                record,
                f"buy leg confirmed not filled after recovery "
                f"({recovered.status.value})",
            )
        return await self._fail(
            record,
            f"buy order state could not be established ({recovered.status.value}) "
            "— a second buy would risk duplication; manual review required",
            state=TransferState.MANUAL_REVIEW,
        )

    async def _submit_withdrawal(self, record: TransferRecord) -> TransferRecord:
        """Withdraw the bought asset to the destination venue's address.

        H-2 idempotency: a withdrawal that was already submitted (or whose
        submission intent was persisted) is never duplicated.  C-3: once the
        buy leg has filled, the purchased asset is tracked explicitly — a
        withdrawal that cannot proceed escalates to MANUAL_REVIEW instead of
        silently abandoning the asset.
        """
        self._guard.ensure_can_trade()
        plan = record.plan
        if record.withdrawal_id is not None:
            # A previous attempt already submitted (or attempted) the
            # withdrawal — never submit a second one.  The state machine
            # normally routes WITHDRAW_SUBMITTED to _check_withdrawal; this
            # guards the BUY_FILLED re-entry path.
            logger.warning(
                "transfer_withdrawal_already_submitted",
                extra={"transfer_id": record.id, "withdrawal_id": record.withdrawal_id},
            )
            return record
        adapter = self._manager.adapter(plan.source_exchange)
        dest_adapter = self._manager.adapter(plan.dest_exchange)

        def _asset_note() -> str:
            return (
                f"purchased {record.buy_filled_amount} {plan.asset} remains on "
                f"{plan.source_exchange}"
            )

        # Re-validate the network right before moving funds (venues suspend
        # networks; the world may have changed since planning).
        try:
            source_networks = await adapter.fetch_withdrawal_networks(plan.asset)
            dest_networks = await dest_adapter.fetch_withdrawal_networks(plan.asset)
            route = self._selector.select(
                asset=plan.asset, source_networks=source_networks, dest_networks=dest_networks
            )
        except NetworkMismatchError as exc:
            return await self._fail(
                record,
                f"network validation failed before withdrawal: {exc}; {_asset_note()} "
                "— manual review required",
                state=TransferState.MANUAL_REVIEW,
            )
        if (
            route.code.upper()
            not in {
                plan.network.upper(),
            }
            and route.network.upper() != plan.network.upper()
        ):
            return await self._fail(
                record,
                f"planned network {plan.network} no longer matches validated "
                f"network {route.network} — refusing to withdraw; {_asset_note()} "
                "— manual review required",
                state=TransferState.MANUAL_REVIEW,
            )

        try:
            address = await dest_adapter.fetch_deposit_address(plan.asset, network=plan.network)
        except Exception as exc:  # noqa: BLE001
            return await self._fail(
                record,
                f"deposit address unavailable: {exc}; {_asset_note()} — manual "
                "review required",
                state=TransferState.MANUAL_REVIEW,
            )

        # Memo/tag safety: a withdrawal to a tag-required asset WITHOUT a
        # tag is sent to an uncredited address and the funds are lost.
        # Fail closed BEFORE the external withdrawal call when the venue
        # returned no memo for an asset we know requires one.  A
        # post-withdrawal MANUAL_REVIEW is not sufficient — the funds
        # are already gone.
        if plan.asset.upper() in _MEMO_REQUIRED_ASSETS and not (
            address.memo and str(address.memo).strip()
        ):
            return await self._fail(
                record,
                f"destination venue {plan.dest_exchange} returned no memo/tag "
                f"for the {plan.asset} deposit address on {plan.network} — "
                f"withdrawals to a {plan.asset} address without a tag lose "
                f"funds; {_asset_note()} — manual review required",
                state=TransferState.MANUAL_REVIEW,
            )

        send_amount = record.buy_filled_amount - route.withdrawal_fee
        if send_amount <= DEC0:
            return await self._fail(
                record,
                f"bought amount {record.buy_filled_amount} does not cover the "
                f"withdrawal fee {route.withdrawal_fee}; {_asset_note()} — "
                "manual review required",
                state=TransferState.MANUAL_REVIEW,
            )

        # H-2: persist the withdrawal intent BEFORE submitting, so a crash
        # between venue acceptance and the outcome save can never lead to a
        # second withdrawal.  In LIVE mode an unmatched intent fails closed
        # through the deposit timeout into MANUAL_REVIEW.
        record = record.with_state(
            TransferState.WITHDRAW_SUBMITTED,
            withdrawal_id=f"{_WITHDRAWAL_INTENT_PREFIX}{record.id[-12:]}",
            deposit_address=address.address,
            withdrawal_amount=send_amount.quantize(_QUANTUM),
        )
        await self._transfers.save(record)

        try:
            if self._mode is TradingMode.PAPER:
                tx = await adapter.withdraw(
                    plan.asset,
                    f"{send_amount}",
                    address.address,
                    memo=address.memo,
                    network=plan.network,
                )
                withdrawal_id, txid = tx.txid, tx.txid
            elif self._mode is TradingMode.DEMO:
                # Demo/testnet venues cannot move real funds between
                # exchanges: the blockchain leg is simulated (flagged).
                withdrawal_id, txid = f"sim-demo-{record.id[-6:]}", f"sim-demo-{record.id[-6:]}"
            else:  # LIVE: a real withdrawal (adapter enforces its own opt-in)
                tx = await adapter.withdraw(
                    plan.asset,
                    f"{send_amount}",
                    address.address,
                    memo=address.memo,
                    network=plan.network,
                )
                withdrawal_id, txid = (tx.txid or tx.id) if hasattr(tx, "id") else tx.txid, tx.txid
        except ExecutionDisabledError:
            raise
        except Exception as exc:  # noqa: BLE001
            return await self._fail(
                record,
                f"withdrawal failed: {exc}; {_asset_note()} — manual review "
                "required",
                state=TransferState.MANUAL_REVIEW,
            )

        if self._mode is TradingMode.PAPER:
            wallet = self._paper_wallets.get(plan.source_exchange)
            if wallet is not None:
                wallet.debit(plan.asset, record.buy_filled_amount)

        record = record.with_state(
            TransferState.WITHDRAW_SUBMITTED,
            withdrawal_id=withdrawal_id,
            withdrawal_txid=txid,
            deposit_address=address.address,
            withdrawal_amount=send_amount.quantize(_QUANTUM),
        )
        await self._transfers.save(record)
        await self._audit.log(
            "TRANSFER_WITHDRAW_SUBMITTED",
            f"{plan.asset} {send_amount} via {plan.network}",
            {"transfer_id": record.id, "withdrawal_id": withdrawal_id},
        )
        return record

    async def _check_withdrawal(self, record: TransferRecord) -> TransferRecord:
        """WITHDRAW_SUBMITTED/WITHDRAW_PENDING -> TRANSFER_IN_PROGRESS|..."""
        plan = record.plan
        if self._mode in (TradingMode.PAPER, TradingMode.DEMO):
            # Simulated blockchain: the transfer is in progress immediately
            # after the (simulated) withdrawal was accepted.
            elapsed = (utc_now() - record.updated_at).total_seconds()
            delay = self._settings.transfer.simulated_transfer_seconds
            # For DEMO make lifecycle faster (10s total) so withdraw_pending
            # does not appear stuck; PAPER keeps 30s for realism.
            if self._mode is TradingMode.DEMO:
                delay = min(delay, Decimal("10"))
            delay_f = float(delay)
            if elapsed < delay_f / 2 and record.state is TransferState.WITHDRAW_SUBMITTED:
                record = record.with_state(TransferState.WITHDRAW_PENDING)
                await self._transfers.save(record)
                return record
            if elapsed < delay_f:
                return record
            record = record.with_state(TransferState.TRANSFER_IN_PROGRESS)
            await self._transfers.save(record)
            return record

        # LIVE: poll the source venue's withdrawal history for our id/txid.
        try:
            withdrawals = await self._manager.adapter(plan.source_exchange).fetch_withdrawals(
                plan.asset, limit=20
            )
        except Exception as exc:  # noqa: BLE001
            return await self._timeout_guard(record, f"withdrawal poll failed: {exc}")
        for tx in withdrawals:
            matches = (record.withdrawal_id and tx.txid == record.withdrawal_id) or (
                record.withdrawal_txid and tx.txid == record.withdrawal_txid
            )
            if not matches:
                continue
            if tx.is_confirmed:
                record = record.with_state(TransferState.TRANSFER_IN_PROGRESS)
                await self._transfers.save(record)
                return record
            if tx.status in ("failed", "canceled", "cancelled", "reject"):
                # The venue reports the withdrawal failed: the purchased
                # asset did NOT leave the source venue — track it
                # explicitly instead of abandoning it (C-3).
                return await self._fail(
                    record,
                    f"withdrawal failed on venue: {tx.status}; purchased "
                    f"{record.buy_filled_amount} {plan.asset} remains on "
                    f"{plan.source_exchange} — manual review required",
                    state=TransferState.MANUAL_REVIEW,
                )
            return record.with_state(TransferState.WITHDRAW_PENDING)
        return await self._timeout_guard(record, "withdrawal not visible in history yet")

    async def _await_deposit(self, record: TransferRecord) -> TransferRecord:
        """TRANSFER_IN_PROGRESS -> DEPOSIT_DETECTED (or timeout escalation)."""
        plan = record.plan
        if self._mode in (TradingMode.PAPER, TradingMode.DEMO):
            elapsed = (utc_now() - record.updated_at).total_seconds()
            delay = self._settings.transfer.simulated_transfer_seconds
            if self._mode is TradingMode.DEMO:
                delay = min(delay, Decimal("10"))
            delay_f = float(delay)
            if elapsed < delay_f:
                return record
            deposit_amount = record.withdrawal_amount
            # For both PAPER and DEMO the blockchain leg is simulated, so
            # credit the destination paper wallet so the simulated sell has
            # funds. Real DEMO buy/sell legs still go through the venue, but
            # the deposited asset is synthetic.
            wallet = self._paper_wallets.get(plan.dest_exchange)
            if wallet is not None:
                wallet.credit(plan.asset, deposit_amount)
            record = record.with_state(
                TransferState.DEPOSIT_DETECTED,
                deposit_txid=f"sim-{record.withdrawal_txid}",
                deposit_amount=deposit_amount.quantize(_QUANTUM),
            )
            await self._transfers.save(record)
            return record

        try:
            deposits = await self._manager.adapter(plan.dest_exchange).fetch_deposits(
                plan.asset, limit=20
            )
        except Exception as exc:  # noqa: BLE001
            return await self._timeout_guard(record, f"deposit poll failed: {exc}")
        for tx in deposits:
            if tx.status in ("pending", "processing"):
                continue
            tolerance = record.withdrawal_amount * Decimal("0.005")
            amount_close = abs(tx.amount - record.withdrawal_amount) <= tolerance
            address_match = record.deposit_address and tx.address == record.deposit_address
            if tx.is_confirmed and (amount_close or address_match):
                record = record.with_state(
                    TransferState.DEPOSIT_DETECTED,
                    deposit_txid=tx.txid,
                    deposit_amount=tx.amount.quantize(_QUANTUM),
                )
                await self._transfers.save(record)
                return record
        return await self._timeout_guard(record, "deposit not detected yet")

    async def _execute_sell(self, record: TransferRecord) -> TransferRecord:
        """SELL leg on the destination venue.

        C-4: the destination asset is tracked explicitly — a sell that cannot
        be confirmed or completed escalates to MANUAL_REVIEW instead of
        marking the whole operation FAILED while holding the asset.
        """
        self._guard.ensure_can_trade()
        plan = record.plan
        symbol = Symbol(base=plan.asset, quote=self._settings.trading.base_currency)
        sell_amount = record.deposit_amount
        if sell_amount <= DEC0:
            return await self._fail(
                record,
                "nothing to sell: deposit amount is zero",
                state=TransferState.MANUAL_REVIEW,
            )
        # For DEMO the deposit is simulated and lives in the paper wallet.
        # On a fresh process the in-memory wallet is empty, so ensure it
        # holds the deposited asset before the simulated sell.
        if self._mode is TradingMode.DEMO:
            wallet = self._paper_wallets.get(plan.dest_exchange)
            if wallet is not None and wallet.free(plan.asset) < sell_amount:
                # Credit the exact deposited amount (idempotent if already credited)
                try:
                    wallet.credit(plan.asset, sell_amount - wallet.free(plan.asset))
                except Exception:
                    wallet.credit(plan.asset, sell_amount)

        def _asset_note() -> str:
            return (
                f"{record.deposit_amount} {plan.asset} deposited on "
                f"{plan.dest_exchange}"
            )

        try:
            fill, order = await self._fill(
                plan.dest_exchange, symbol, OrderSide.SELL, base_amount=sell_amount
            )
        except ExecutionDisabledError:
            raise
        except Exception as exc:  # noqa: BLE001
            return await self._fail(
                record,
                f"sell leg error: {exc}; {_asset_note()} — manual review required",
                state=TransferState.MANUAL_REVIEW,
            )
        # Net-of-fee proceeds in quote currency (paper: simulator net; live:
        # venue average price x fill minus the quote-denominated fee).
        if fill is not None:
            proceeds = fill.quote_amount
        elif order.average_price is not None:
            proceeds = order.filled_amount * order.average_price
            if order.fee_currency == symbol.quote:
                proceeds -= order.fee_paid
        else:
            proceeds = DEC0
        record = record.with_state(
            TransferState.SELL_SUBMITTED,
            sell_order=_order_to_json(order),
            sell_filled_amount=order.filled_amount,
            sell_proceeds_quote=proceeds.quantize(_QUANTUM) if proceeds > DEC0 else DEC0,
        )
        await self._transfers.save(record)
        if order.outcome_unconfirmed:
            return await self._fail(
                record,
                f"sell leg outcome unconfirmed ({order.status.value}"
                f"{f': {order.error}' if order.error else ''}) — the deposit may "
                f"have been (partially) sold; {_asset_note()} — manual review "
                "required",
                state=TransferState.MANUAL_REVIEW,
            )
        if order.filled_amount <= DEC0:
            return await self._fail(
                record,
                f"sell leg rejected: {order.error or 'no fill'}; {_asset_note()} "
                "— manual review required",
                state=TransferState.MANUAL_REVIEW,
            )
        return await self._finish(record)

    async def _finish(self, record: TransferRecord) -> TransferRecord:
        """SELL_SUBMITTED -> COMPLETED with realised P&L and a trade record."""
        plan = record.plan
        sell_order = record.sell_order or {}
        proceeds = record.sell_proceeds_quote
        filled = record.sell_filled_amount
        if proceeds <= DEC0 and filled > DEC0:
            average = sell_order.get("average_price")
            if average is not None:
                proceeds = filled * Decimal(str(average))
        if proceeds <= DEC0:
            return await self._fail(
                record, "cannot compute sell proceeds", state=TransferState.MANUAL_REVIEW
            )

        if self._mode is TradingMode.PAPER:
            wallet = self._paper_wallets.get(plan.dest_exchange)
            if wallet is not None:
                wallet.debit(plan.asset, filled)
                wallet.credit(self._settings.trading.base_currency, proceeds)

        # Use actual buy cost from the filled buy order when available
        buy_avg = None
        try:
            buy_avg = Decimal(str((record.buy_order or {}).get("average_price"))) if record.buy_order else None
        except Exception:
            buy_avg = None
        if buy_avg is not None and buy_avg > DEC0:
            buy_cost = record.buy_filled_amount * buy_avg
        else:
            buy_cost = plan.buy_cost_quote
        # Realized is proceeds (already net of sell fee) minus actual buy cost.
        # Fees are already embedded via reduced amounts, so realized is net.
        # For reporting, compute fees from actual averages when available.
        sell_avg = None
        try:
            sell_avg = Decimal(str(sell_order.get("average_price"))) if sell_order.get("average_price") else None
        except Exception:
            sell_avg = None
        if sell_avg is None or sell_avg <= DEC0:
            sell_avg = plan.sell_price
        buy_avg_for_fee = buy_avg if buy_avg is not None and buy_avg > DEC0 else plan.buy_price
        realized = proceeds - buy_cost
        fees = (
            record.buy_filled_amount * buy_avg_for_fee * plan.buy_fee_bps / Decimal("10000")
            + filled * sell_avg * plan.sell_fee_bps / Decimal("10000")
            + plan.withdrawal_fee * sell_avg
        )
        record = record.with_state(
            TransferState.COMPLETED,
            sell_proceeds_quote=proceeds.quantize(_QUANTUM),
            fees_quote=fees.quantize(_QUANTUM),
            realized_profit_quote=realized.quantize(_QUANTUM),
        )
        await self._transfers.save(record)
        await self._audit.log(
            "TRANSFER_COMPLETED",
            f"{plan.source_exchange} -> {plan.dest_exchange} {plan.asset}",
            {
                "transfer_id": record.id,
                "realized_profit": str(realized),
                "planned_profit": str(plan.net_profit_quote),
            },
        )
        return record

    # ---------------------------------------------------------------- internals
    async def _timeout_guard(self, record: TransferRecord, note: str) -> TransferRecord:
        """Escalate to MANUAL_REVIEW once the deposit timeout is exhausted."""
        elapsed = (utc_now() - record.created_at).total_seconds()
        if elapsed > self._settings.transfer.deposit_timeout_seconds:
            updated = record.with_state(
                TransferState.MANUAL_REVIEW,
                error=f"timeout: {note} after {elapsed:.0f}s",
            )
            await self._transfers.save(updated)
            return updated
        return record

    async def _fill(
        self,
        venue: str,
        symbol: Symbol,
        side: OrderSide,
        *,
        base_amount: Decimal | None = None,
        quote_amount: Decimal | None = None,
        on_submit: Callable[[OrderRequest], Awaitable[None]] | None = None,
    ) -> tuple[SimulatedFill | None, Order]:
        """One order leg in the current mode (paper simulation vs real order).

        ``on_submit`` (venue legs only) is fired with the final request right
        before submission so the caller can persist a PENDING intent
        identified by its ``client_order_id``.
        """
        book = await self._fresh_book(venue, symbol)
        # For DEMO the sell leg's funds are simulated (deposit is simulated),
        # so use the paper-wallet path for SELL in DEMO. Buy remains real.
        is_demo_sell_via_paper = (
            self._mode is TradingMode.DEMO
            and side is OrderSide.SELL
            and venue in self._paper_wallets
        )
        if self._mode is TradingMode.PAPER or is_demo_sell_via_paper:
            fill = self._sim.simulate(
                book, side, base_amount=base_amount, quote_amount=quote_amount
            )
            if fill.is_rejected:
                return None, Order(
                    exchange_id=venue,
                    symbol=symbol,
                    side=side,
                    order_type=OrderType.MARKET,
                    status=OrderStatus.REJECTED,
                    amount=base_amount or DEC0,
                    error=fill.rejected_reason,
                )
            self._apply_wallet(venue, side, symbol, fill)
            order = Order(
                exchange_id=venue,
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,
                status=OrderStatus.FILLED,
                amount=fill.filled_amount,
                filled_amount=fill.filled_amount,
                average_price=fill.average_price,
                fee_paid=fill.fee,
                fee_currency=symbol.base if side is OrderSide.BUY else symbol.quote,
            )
            return fill, order

        # DEMO buy / LIVE: real market order with timeout + recovery.
        amount = base_amount
        if amount is None:
            if not book.asks:
                return None, Order(
                    exchange_id=venue,
                    symbol=symbol,
                    side=side,
                    order_type=OrderType.MARKET,
                    status=OrderStatus.REJECTED,
                    error="cannot size buy from empty book",
                )
            amount = (quote_amount / book.asks[0].price).quantize(_QUANTUM)
        request = OrderRequest(
            exchange_id=venue,
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            amount=amount,
        )
        adapter = self._manager.adapter(venue)
        if on_submit is not None:
            # H-2: persist the intent BEFORE the order exists on the venue.
            await on_submit(request)
        try:
            order = await asyncio.wait_for(
                adapter.create_order(request),
                timeout=self._settings.execution.leg_timeout_seconds,
            )
        except TimeoutError:
            order = Order(
                exchange_id=venue,
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,
                amount=amount,
                client_order_id=request.client_order_id,
                status=OrderStatus.TIMEOUT,
                error="placement timed out",
            )
        except Exception as exc:  # noqa: BLE001
            order = Order(
                exchange_id=venue,
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,
                amount=amount,
                client_order_id=request.client_order_id,
                status=OrderStatus.UNKNOWN,
                error=f"{type(exc).__name__}: {exc}"[:500],
            )
        if not order.client_order_id:
            # Some venues do not echo the client id; keep the intent identity
            # so the persisted outcome replaces the PENDING placeholder.
            order = order.model_copy(update={"client_order_id": request.client_order_id})
        if order.status in (
            OrderStatus.TIMEOUT,
            OrderStatus.UNKNOWN,
            OrderStatus.PARTIALLY_FILLED,
        ):
            order = await self._recovery.recover(order, adapter)
        return None, order

    def _apply_wallet(
        self, venue: str, side: OrderSide, symbol: Symbol, fill: SimulatedFill
    ) -> None:
        wallet = self._paper_wallets.get(venue)
        if wallet is None:
            return
        if side is OrderSide.BUY:
            wallet.debit(symbol.quote, fill.quote_amount)
            wallet.credit(symbol.base, fill.filled_amount)
        else:
            wallet.debit(symbol.base, fill.filled_amount)
            wallet.credit(symbol.quote, fill.quote_amount)

    async def _fresh_book(self, venue: str, symbol: Symbol):
        book = self._store.order_book(venue, symbol)
        if book is not None and not self._store.is_stale(self._store.age_ms(book)):
            return book
        # Stale or missing -> one on-demand refresh in DEMO (fail-closed if still stale)
        if self._mode is TradingMode.DEMO and getattr(self, "_market", None) is not None:
            try:
                await self._market.refresh_order_books([symbol], exchange_ids=[venue])
            except Exception:
                pass
            book = self._store.order_book(venue, symbol)
            if book is not None and not self._store.is_stale(self._store.age_ms(book)):
                return book
        if book is None:
            raise ValueError(f"no order book cached for {venue} {symbol.name}")
        raise ValueError(
            f"stale order book for {venue} {symbol.name} "
            f"({self._store.age_ms(book):.0f} ms) — refusing to plan/execute"
        )

    async def _available_quote(self, venue: str) -> Decimal:
        quote = self._settings.trading.base_currency
        if self._mode is TradingMode.PAPER:
            wallet = self._paper_wallets.get(venue)
            if wallet is not None:
                return wallet.free(quote)
        try:
            snapshot = await self._manager.adapter(venue).fetch_balances()
            return snapshot.free_of(quote)
        except Exception:  # noqa: BLE001 - unknown balance -> zero budget
            return DEC0

    async def _taker_fees(self, venue: str, symbol: Symbol) -> Decimal:
        try:
            fees = await self._manager.adapter(venue).fetch_trading_fees(symbol)
            return fees.taker_bps
        except Exception:  # noqa: BLE001 - unknown fee -> default taker
            return Decimal("10")

    async def _fail(
        self,
        record: TransferRecord,
        reason: str,
        *,
        state: TransferState = TransferState.FAILED,
    ) -> TransferRecord:
        logger.warning(
            "transfer_failed",
            extra={"transfer_id": record.id, "reason": reason, "state": state.value},
        )
        updated = record.with_state(state, error=reason)
        await self._transfers.save(updated)
        await self._audit.log(
            "TRANSFER_FAILED" if state is TransferState.FAILED else "TRANSFER_MANUAL_REVIEW",
            reason,
            {"transfer_id": record.id},
        )
        return updated


def _order_to_json(order: Order) -> dict:
    import json

    computed = set(order.model_computed_fields)
    return json.loads(order.model_dump_json(exclude=computed))


def _order_from_json(data: dict | None) -> Order | None:
    if data is None:
        return None
    try:
        return Order.model_validate(data)
    except Exception:  # noqa: BLE001 - unparsable records must fail closed
        return None
