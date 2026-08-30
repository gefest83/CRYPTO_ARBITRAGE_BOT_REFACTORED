"""Triangular arbitrage: depth-aware scanner + sequential 3-leg executor."""

from app.strategies.triangular.executor import TriangleExecutor
from app.strategies.triangular.fees import (
    FeeProvider,
    ScanRequest,
    StaticFeeProvider,
    VenueProvider,
)
from app.strategies.triangular.scanner import TriangleRoute, TriangularScanner

__all__ = [
    "FeeProvider",
    "ScanRequest",
    "StaticFeeProvider",
    "TriangleExecutor",
    "TriangleRoute",
    "TriangularScanner",
    "VenueProvider",
]
