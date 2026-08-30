"""Venue profiles: the bot's exchange ids ↔ ccxt exchange classes.

One profile per supported venue (Binance, OKX, Bybit — the only venues this
bot supports).  The profile is the single place that knows

* which internal id maps to which ccxt class — the declared ``ccxt_id`` plus
  renamed/legacy aliases, resolved against the **installed** ``ccxt.exchanges``
  registry by the adapter layer (exact id wins, otherwise an alias, else
  fail-closed);
* how the venue's DEMO/testnet environment is routed (each of the three
  venues does it differently — see the profile flags).

Fail-closed rule: a CCXT adapter for an id without a profile refuses to
instantiate with a :class:`~app.errors.ConfigurationError` that lists every
supported venue — an unmapped id must never silently hit ccxt's generic
"no exchange named ..." error after credentials were sent.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.errors import ConfigurationError

__all__ = [
    "VenueProfile",
    "ccxt_candidate_ids",
    "resolve_profile",
    "supported_ids",
]


@dataclass(frozen=True, slots=True)
class VenueProfile:
    """Static facts about one supported venue."""

    #: Internal bot id (also the id stored in the DB / universe).
    id: str
    #: Class name inside the ccxt module.
    ccxt_id: str
    display_name: str
    #: Extra internal ids that resolve to this venue.
    aliases: tuple[str, ...] = ()
    #: DEMO/testnet routing, per venue:
    #
    #: * binance — dedicated demo endpoints (demo-api.binance.com) served via
    #:   the ccxt ``urls['demo']`` section; ``set_sandbox_mode`` is NOT used
    #:   (it would point at the separate testnet account system).  The demo
    #:   host does not serve ``load_markets`` (times out), so markets are
    #:   loaded from production endpoints.
    #: * okx — demo needs the ``x-simulated-trading: 1`` header on every
    #:   private request; endpoints stay the same.
    #: * bybit — demo trading lives on its own host (api-demo.bybit.com),
    #:   which serves PRIVATE calls only: public data stays on production.
    #:   ``set_sandbox_mode`` is NOT used (demo keys are invalid on testnet).
    demo_base_url: str | None = None
    #: The demo host serves private calls only; public data stays on
    #: production (bybit).  Fail-closed if the URL table is unexpected.
    demo_private_only: bool = False
    #: The venue's DEMO environment uses dedicated endpoints that replace the
    #: sandbox/testnet (binance demo-api.binance.com).
    demo_replaces_sandbox: bool = False
    #: The ccxt client has a dedicated ``demo`` section in its ``urls`` dict
    #: (binance).  DEMO mode copies ``urls['demo']`` -> ``urls['api']``.
    demo_has_url_section: bool = False
    #: Demo headers merged into every request in DEMO mode (okx).
    demo_headers: tuple[tuple[str, str], ...] = ()
    #: The venue's DEMO environment does not support ``load_markets``
    #: (binance demo host times out): precision is sourced from defaults.
    demo_no_load_markets: bool = False


VENUE_PROFILES: tuple[VenueProfile, ...] = (
    VenueProfile(
        "binance",
        "binance",
        "Binance",
        demo_replaces_sandbox=True,
        demo_has_url_section=True,
        demo_no_load_markets=True,
    ),
    VenueProfile(
        "okx",
        "okx",
        "OKX",
        demo_headers=(("x-simulated-trading", "1"),),
    ),
    VenueProfile(
        "bybit",
        "bybit",
        "Bybit",
        demo_base_url="https://api-demo.bybit.com",
        demo_private_only=True,
    ),
)

_BY_KEY: dict[str, VenueProfile] = {}
for _profile in VENUE_PROFILES:
    _BY_KEY[_profile.id] = _profile
    _BY_KEY[_profile.ccxt_id] = _profile
    for _alias in _profile.aliases:
        _BY_KEY[_alias] = _profile


def resolve_profile(exchange_id: str) -> VenueProfile:
    """Profile for an internal/aliased/ccxt id; fail-closed when unknown."""
    key = exchange_id.strip().lower()
    profile = _BY_KEY.get(key)
    if profile is not None:
        return profile
    raise ConfigurationError(
        f"unknown exchange id '{exchange_id}'; supported venues: {', '.join(supported_ids())}",
        exchange_id=exchange_id,
    )


def supported_ids() -> tuple[str, ...]:
    """Canonical internal ids, sorted for deterministic error messages."""
    return tuple(sorted(profile.id for profile in VENUE_PROFILES))


def ccxt_candidate_ids(profile: VenueProfile) -> tuple[str, ...]:
    """ccxt class names for this venue in preference order (exact id first)."""
    return (profile.ccxt_id, *profile.aliases)
