"""Centralized i18n layer for Telegram interface.

Supported languages: en, ru.  No silent default on first /start.
All user-facing Telegram text must go through :func:`t`.
"""

from __future__ import annotations

__all__ = [
    "SUPPORTED_LANGUAGES",
    "LANGUAGE_PICKER_PROMPT",
    "LANGUAGE_PICKER_KEYBOARD",
    "is_supported_lang",
    "lang_storage_key",
    "t",
    "TRANSLATIONS",
]

SUPPORTED_LANGUAGES: tuple[str, ...] = ("en", "ru")

LANGUAGE_PICKER_PROMPT = "Choose language / Выберите язык:"

LANGUAGE_PICKER_KEYBOARD = {
    "inline_keyboard": [
        [
            {"text": "🇬🇧 English", "callback_data": "lang:en"},
            {"text": "🇷🇺 Русский", "callback_data": "lang:ru"},
        ]
    ]
}


def is_supported_lang(lang: str | None) -> bool:
    return lang in SUPPORTED_LANGUAGES


def lang_storage_key(user_id: int) -> str:
    return f"tg_lang:{user_id}"


# fmt: off
TRANSLATIONS: dict[str, dict[str, str]] = {
    "en": {
        "help_text": (
            "crypto-arbitrage-bot — operator commands:\n"
            "/start        - welcome / help\n"
            "/help         - this message\n"
            "/status       - mode, kill switch, exchanges, transfers, recent trades\n"
            "/reconcile    - list MANUAL_REVIEW transfers (read-only)\n"
            "/opportunities - scan triangles (no execution)\n"
            "/pause        - engage the persistent kill switch (safe)\n"
            "/resume       - release the kill switch (safe)\n"
            "/start_trading - start the DEMO auto-trading loop (DEMO only)\n"
            "/stop_trading  - stop the auto-trading loop (idempotent)\n"
            "/language     - choose language (English / Русский)\n"
            "\n"
            "Order placement, transfers and withdrawals are NOT exposed here — "
            "use the CLI for any execution that moves funds."
        ),
        "unauthorized": "unauthorized",
        "internal_error": "internal error",
        "unknown_command": "unknown command\n\n{help_text}",

        "language_picker_prompt": "Choose language / Выберите язык:",
        "language_selected": "Language changed to English.",
        "language_picker_chosen": "Language selected: English",

        "common_on": "on",
        "common_off": "off",
        "common_running": "running",
        "common_stopped": "stopped",

        "status_mode": "mode: {mode}",
        "status_uptime": "uptime: {sec}s",
        "status_trading_enabled": "trading enabled: {val}",
        "status_kill_switch_engaged": "kill switch: ENGAGED ({reason})",
        "status_kill_switch_released": "kill switch: released",
        "status_auto_trading_flag": "auto trading (flag): {flag}",
        "status_auto_loop": "auto loop: {state}",
        "status_exchanges_header": "exchanges:",
        "status_exchange_line": "  {venue}: {status} (keys {credentials})",
        "status_market_data": "market data: {order_books} books / {tickers} tickers across {exchanges} venues",
        "status_daily_pnl": "daily pnl: {pnl}",
        "status_open_transfers_count": "open transfers: {count}",
        "status_open_transfers_header": "open transfers:",
        "status_open_transfer_line": "  {route} {asset} {amount}: {state}",
        "status_recent_trades_header": "recent trades:",
        "status_recent_trade_line": "  {strategy} {route} {status} net {net_profit}",

        "reconcile_empty": "no transfers require manual review",
        "reconcile_header": "{count} transfer(s) require manual review:",
        "reconcile_id": "  id        : {id}",
        "reconcile_state": "  state     : {state}",
        "reconcile_route": "  route     : {route}",
        "reconcile_asset": "  asset     : {asset} (planned {amount})",
        "reconcile_held": "  held      : {amount} {asset} on {exchange}",
        "reconcile_withdrawal": "  withdrawal: {amount} {asset} on {exchange} ({details})",
        "reconcile_withdrawal_no_details": "  withdrawal: {amount} {asset} on {exchange}",
        "reconcile_created": "  created   : {ts}",
        "reconcile_updated": "  updated   : {ts}",
        "reconcile_reason": "  reason    : {reason}",
        "reconcile_more": "... and {remaining} more (CLI has the full list)",

        "opportunities_triangle": "triangle opportunities: {count}",
        "opportunities_triangle_line": "  {direction} net {net_profit_bps} bps notional {size_notional_quote}",
        "opportunities_triangle_none": "  (none above configured minimum)",
        "opportunities_transfer": "transfer plans: {count}",
        "opportunities_transfer_line": "  {source}->{dest} {asset} {amount} via {network}: net {net_profit_bps:.1f} bps",
        "opportunities_transfer_none": "  (none above configured minimum)",
        "opportunities_view_only": "(no execution — view-only)",

        "pause_already_engaged": "kill switch already engaged: {reason}",
        "pause_engaged": "kill switch engaged (persisted across restart)",

        "resume_already_released": "kill switch already released",
        "resume_released": "kill switch released — auto trading is still OFF; start it from the CLI when ready",

        "start_trading_refused_mode": "refused: /start_trading is only allowed in DEMO mode (current: {mode})",
        "start_trading_no_controller": "refused: auto-trading controller is not initialised",
        "start_trading_started": "auto trading started: {msg}",
        "start_trading_not_started": "auto trading not started: {msg}",

        "stop_trading_no_controller": "auto trading controller is not initialised",
        "stop_trading_stopped": "auto trading stopped: {msg}",
        "stop_trading_already_stopped": "auto trading already stopped: {msg}",

        "callback_language_changed": "Language changed",
    },
    "ru": {
        "help_text": (
            "crypto-arbitrage-bot — команды оператора:\n"
            "/start        - приветствие / помощь\n"
            "/help         - это сообщение\n"
            "/status       - режим, стоп-кран, биржи, переводы, последние сделки\n"
            "/reconcile    - список переводов MANUAL_REVIEW (только чтение)\n"
            "/opportunities - поиск треугольников (без исполнения)\n"
            "/pause        - включить стоп-кран (безопасно)\n"
            "/resume       - выключить стоп-кран (безопасно)\n"
            "/start_trading - запустить DEMO автоторговлю (только DEMO)\n"
            "/stop_trading  - остановить автоторговлю (идемпотентно)\n"
            "/language     - выбрать язык (English / Русский)\n"
            "\n"
            "Размещение ордеров, переводы и выводы НЕ доступны здесь — "
            "используйте CLI для любых операций с движением средств."
        ),
        "unauthorized": "доступ запрещён",
        "internal_error": "внутренняя ошибка",
        "unknown_command": "неизвестная команда\n\n{help_text}",

        "language_picker_prompt": "Choose language / Выберите язык:",
        "language_selected": "Язык изменён на русский.",
        "language_picker_chosen": "Язык выбран: Русский",

        "common_on": "вкл",
        "common_off": "выкл",
        "common_running": "запущен",
        "common_stopped": "остановлен",

        "status_mode": "режим: {mode}",
        "status_uptime": "аптайм: {sec}с",
        "status_trading_enabled": "торговля включена: {val}",
        "status_kill_switch_engaged": "стоп-кран: СРАБОТАЛ ({reason})",
        "status_kill_switch_released": "стоп-кран: снят",
        "status_auto_trading_flag": "автоторговля (флаг): {flag}",
        "status_auto_loop": "автоцикл: {state}",
        "status_exchanges_header": "биржи:",
        "status_exchange_line": "  {venue}: {status} (ключи {credentials})",
        "status_market_data": "рыночные данные: {order_books} стаканов / {tickers} тикеров на {exchanges} площадках",
        "status_daily_pnl": "дневной PnL: {pnl}",
        "status_open_transfers_count": "открытые переводы: {count}",
        "status_open_transfers_header": "открытые переводы:",
        "status_open_transfer_line": "  {route} {asset} {amount}: {state}",
        "status_recent_trades_header": "последние сделки:",
        "status_recent_trade_line": "  {strategy} {route} {status} net {net_profit}",

        "reconcile_empty": "нет переводов, требующих ручной проверки",
        "reconcile_header": "{count} перевод(ов) требуют ручной проверки:",
        "reconcile_id": "  id        : {id}",
        "reconcile_state": "  статус    : {state}",
        "reconcile_route": "  маршрут   : {route}",
        "reconcile_asset": "  актив     : {asset} (запланировано {amount})",
        "reconcile_held": "  хранится  : {amount} {asset} на {exchange}",
        "reconcile_withdrawal": "  вывод     : {amount} {asset} на {exchange} ({details})",
        "reconcile_withdrawal_no_details": "  вывод     : {amount} {asset} на {exchange}",
        "reconcile_created": "  создан    : {ts}",
        "reconcile_updated": "  обновлён  : {ts}",
        "reconcile_reason": "  причина   : {reason}",
        "reconcile_more": "... и ещё {remaining} (полный список в CLI)",

        "opportunities_triangle": "треугольные возможности: {count}",
        "opportunities_triangle_line": "  {direction} net {net_profit_bps} bps номинал {size_notional_quote}",
        "opportunities_triangle_none": "  (нет подходящих выше минимума)",
        "opportunities_transfer": "планы переводов: {count}",
        "opportunities_transfer_line": "  {source}->{dest} {asset} {amount} через {network}: net {net_profit_bps:.1f} bps",
        "opportunities_transfer_none": "  (нет подходящих выше минимума)",
        "opportunities_view_only": "(без исполнения — только просмотр)",

        "pause_already_engaged": "стоп-кран уже сработал: {reason}",
        "pause_engaged": "стоп-кран включён (сохранится после перезапуска)",

        "resume_already_released": "стоп-кран уже снят",
        "resume_released": "стоп-кран снят — автоторговля всё ещё ВЫКЛ; запустите её из CLI, когда будете готовы",

        "start_trading_refused_mode": "отклонено: /start_trading разрешён только в режиме DEMO (текущий: {mode})",
        "start_trading_no_controller": "отклонено: контроллер автоторговли не инициализирован",
        "start_trading_started": "автоторговля запущена: {msg}",
        "start_trading_not_started": "автоторговля не запущена: {msg}",

        "stop_trading_no_controller": "контроллер автоторговли не инициализирован",
        "stop_trading_stopped": "автоторговля остановлена: {msg}",
        "stop_trading_already_stopped": "автоторговля уже остановлена: {msg}",

        "callback_language_changed": "Язык изменён",
    },
}
# fmt: on


def t(key: str, lang: str | None, **kwargs) -> str:
    """Translate *key* into *lang*.

    Falls back to English when *lang* is missing or unsupported, and to
    the key itself when the translation is absent in both languages.
    """
    effective = lang if lang in SUPPORTED_LANGUAGES else "en"
    table = TRANSLATIONS.get(effective, TRANSLATIONS["en"])
    template = table.get(key)
    if template is None:
        # Fallback to English or to the key
        template = TRANSLATIONS["en"].get(key, key)
    if kwargs:
        try:
            return template.format(**kwargs)
        except Exception:
            return template
    return template
