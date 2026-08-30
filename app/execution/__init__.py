"""Execution: order gates, kill-switch guard, fill simulation, precision."""

from app.execution.fill_simulator import FillSimulator, SimulatedFill
from app.execution.guard import ExecutionGuard
from app.execution.order_gate import live_session_gate, never_place_orders, sandbox_only
from app.execution.precision import (
    InstrumentFilters,
    PrecisionProvider,
    StaticPrecisionProvider,
    apply_filters,
    round_step_down,
)

__all__ = [
    "ExecutionGuard",
    "FillSimulator",
    "InstrumentFilters",
    "PrecisionProvider",
    "SimulatedFill",
    "StaticPrecisionProvider",
    "apply_filters",
    "live_session_gate",
    "never_place_orders",
    "round_step_down",
    "sandbox_only",
]
