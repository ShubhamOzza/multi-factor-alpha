"""
attribution.py  --  Per-factor backtest and PnL attribution.

    factor_returns     : Standalone single-factor backtests
    rolling_factor_ic  : Per-factor rolling rank-IC (delegates to analytics.py)
"""
from __future__ import annotations

import pandas as pd

from analytics import rolling_ic
from portfolio import run_backtest


def factor_returns(
    prices: pd.DataFrame,
    factor_signals: dict[str, pd.DataFrame],
) -> dict[str, pd.Series]:
    """Run an independent backtest for each factor signal.

    Single-factor returns isolate the standalone alpha of each signal, using
    the same portfolio construction parameters as the composite strategy.
    """
    return {name: run_backtest(prices, sig) for name, sig in factor_signals.items()}


def rolling_factor_ic(
    factor_signals: dict[str, pd.DataFrame],
    prices: pd.DataFrame,
    window: int = 63,
) -> pd.DataFrame:
    """Rolling mean rank-IC for each factor independently.

    Delegates to analytics.rolling_ic so the IC implementation stays in
    a single canonical location.
    """
    series: dict[str, pd.Series] = {}
    for name, sig in factor_signals.items():
        ic = rolling_ic(sig, prices, window=window, horizon=1)
        if not ic.empty:
            series[name] = ic
    return pd.DataFrame(series).dropna(how="all")
