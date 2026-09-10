"""Binance Wallet Prediction Markets — official API endpoints.

Sources (2026-06):
  - https://developers.binance.com/en/docs/catalog/web3-wallet-prediction-trading
  - https://developers.binance.com/legacy-docs/w3w_prediction/*
  - Binance announcement 2026-06-08: Prediction Markets API launch
  - Predict.fun as upstream vendor (PREDICT_FUN on BSC chainId 56)
  - WebSocket: w3w Prediction orderbook push (SAPI WSS)

All endpoints are SAPI signed (apiKey + HMAC) — even market data.
Base URL: https://api.binance.com
"""

from __future__ import annotations

BASE_URL = "https://api.binance.com"

# Market data (signed GET)
ENDPOINT_CATEGORY_LIST = "/sapi/v1/w3w/wallet/prediction/category/list"
ENDPOINT_MARKET_LIST = "/sapi/v1/w3w/wallet/prediction/market/list"
ENDPOINT_MARKET_SEARCH = "/sapi/v1/w3w/wallet/prediction/market/search"
ENDPOINT_MARKET_DETAIL = "/sapi/v1/w3w/wallet/prediction/market/detail"
ENDPOINT_ORDER_BOOK = "/sapi/v1/w3w/wallet/prediction/order-book"
ENDPOINT_LAST_TRADE_PRICE = "/sapi/v1/w3w/wallet/prediction/order-book/last-trade-price"

# Trading (signed) — inspected only, never invoked to place orders
ENDPOINT_GET_QUOTE = "/sapi/v1/w3w/wallet/prediction/trade/get-quote"
ENDPOINT_WALLET_LIST = "/sapi/v1/w3w/wallet/prediction/wallet/list"

# Account / position / redemption / transfers (documented for completeness)
ENDPOINT_POSITION_LIST = "/sapi/v1/w3w/wallet/prediction/position/list"
ENDPOINT_BATCH_REDEEM = "/sapi/v1/w3w/wallet/prediction/batch-redeem"

# WebSocket (SAPI WSS, signed)
WS_BASE_URL = "wss://api.binance.com/sapi/wss"
WS_TOPIC_ORDERBOOK_AGG = "web3_prediction_orderbook_data"
WS_TOPIC_ORDERBOOK_SINGLE_TPL = "web3_prediction_orderbook_{marketId}"

# Predict.fun direct (upstream) — used only to assess historical data availability
PREDICT_FUN_BASE_MAINNET = "https://api.predict.fun"
PREDICT_FUN_BASE_TESTNET = "https://api-testnet.predict.fun"

# Human-readable registry
ALL_ENDPOINTS: dict[str, str] = {
    "category_list": ENDPOINT_CATEGORY_LIST,
    "market_list": ENDPOINT_MARKET_LIST,
    "market_search": ENDPOINT_MARKET_SEARCH,
    "market_detail": ENDPOINT_MARKET_DETAIL,
    "order_book": ENDPOINT_ORDER_BOOK,
    "last_trade_price": ENDPOINT_LAST_TRADE_PRICE,
    "get_quote": ENDPOINT_GET_QUOTE,
    "wallet_list": ENDPOINT_WALLET_LIST,
    "position_list": ENDPOINT_POSITION_LIST,
    "batch_redeem": ENDPOINT_BATCH_REDEEM,
}

WEIGHTS: dict[str, int] = {
    ENDPOINT_CATEGORY_LIST: 200,
    ENDPOINT_MARKET_LIST: 200,
    ENDPOINT_MARKET_SEARCH: 200,
    ENDPOINT_MARKET_DETAIL: 200,
    ENDPOINT_ORDER_BOOK: 200,
    ENDPOINT_LAST_TRADE_PRICE: 200,
    ENDPOINT_GET_QUOTE: 200,
}
