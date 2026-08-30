"""Network validation: only provably-safe networks may carry a transfer."""

from decimal import Decimal

import pytest
from app.models.transfer import WithdrawalNetwork
from app.strategies.transfer.networks import NetworkMismatchError, NetworkSelector

D = Decimal


def _net(network: str, code: str, *, withdraw=True, deposit=True, fee="0.01") -> WithdrawalNetwork:
    return WithdrawalNetwork(
        network=network,
        network_code=code,
        withdraw_enabled=withdraw,
        deposit_enabled=deposit,
        withdrawal_fee=D(fee),
    )


def test_selects_cheapest_common_network():
    selector = NetworkSelector()
    source = (
        _net("ERC20", "ETH", fee="5"),
        _net("TRC20", "TRX", fee="1"),
        _net("SOL", "SOL", fee="0.01"),
    )
    dest = (
        _net("ERC20", "ETH", fee="6"),
        _net("TRC20", "TRX", fee="1.5"),
        _net("BEP20", "BSC", fee="0.5"),
    )
    route = selector.select(asset="USDT", source_networks=source, dest_networks=dest)
    assert route.code == "TRX"  # ETH(5) vs TRX(1): cheapest common wins
    assert route.network == "TRC20"
    assert route.withdrawal_fee == D("1")


def test_matches_by_unified_code_not_display_name():
    """ERC20 on one venue and 'Ethereum' code ETH on the other are the same chain."""
    selector = NetworkSelector()
    source = (_net("ERC20", "ETH"),)
    dest = (_net("ETH", "ETH"),)
    route = selector.select(asset="USDT", source_networks=source, dest_networks=dest)
    assert route.code == "ETH"


def test_no_common_network_refuses():
    selector = NetworkSelector()
    source = (_net("ERC20", "ETH"),)
    dest = (_net("SOL", "SOL"),)
    with pytest.raises(NetworkMismatchError, match="no common enabled network"):
        selector.select(asset="USDT", source_networks=source, dest_networks=dest)


def test_withdrawals_disabled_refuses():
    selector = NetworkSelector()
    source = (_net("ERC20", "ETH", withdraw=False),)
    dest = (_net("ERC20", "ETH"),)
    with pytest.raises(NetworkMismatchError, match="withdrawals are disabled"):
        selector.select(asset="USDT", source_networks=source, dest_networks=dest)


def test_deposits_disabled_refuses():
    selector = NetworkSelector()
    source = (_net("ERC20", "ETH"),)
    dest = (_net("ERC20", "ETH", deposit=False),)
    with pytest.raises(NetworkMismatchError, match="deposits are disabled"):
        selector.select(asset="USDT", source_networks=source, dest_networks=dest)


def test_missing_network_data_refuses():
    selector = NetworkSelector()
    with pytest.raises(NetworkMismatchError):
        selector.select(asset="USDT", source_networks=(), dest_networks=(_net("SOL", "SOL"),))
    with pytest.raises(NetworkMismatchError):
        selector.select(asset="USDT", source_networks=(_net("SOL", "SOL"),), dest_networks=())
