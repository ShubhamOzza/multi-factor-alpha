"""
regime.py  --  Volatility regime classification and conditional performance.

Tercile thresholds are computed from the full sample once (non-lookahead).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from metrics import (
    annualized_return, annualized_volatility, sharpe_ratio, max_drawdown,
)


def classify_regimes(
    returns: pd.Series,
    window: int = 63,
) -> pd.Series:
    """Classify each date into Low / Mid / High vol regime.

    Rolling realized volatility (annualized) is computed with a `window`-day
    lookback. Tercile thresholds are fixed on the full history to avoid
    any lookahead bias in regime labels.
    """
    rolling_vol = returns.rolling(window).std() * np.sqrt(252)
    rolling_vol = rolling_vol.bfill()   # fill warmup window with first valid

    q33 = rolling_vol.quantile(1 / 3)
    q67 = rolling_vol.quantile(2 / 3)

    regimes = pd.Series("Mid Vol", index=returns.index, dtype=object)
    regimes[rolling_vol <= q33] = "Low Vol"
    regimes[rolling_vol >  q67] = "High Vol"
    return regimes


def regime_stats(returns: pd.Series, regimes: pd.Series) -> pd.DataFrame:
    """Annualized performance metrics conditional on volatility regime."""
    rows: list[dict] = []
    for label in ("Low Vol", "Mid Vol", "High Vol"):
        r = returns[regimes == label].dropna()
        if len(r) < 20:
            continue
        rows.append({
            "Regime":      label,
            "Days":        len(r),
            "Ann. Return": annualized_return(r),
            "Ann. Vol":    annualized_volatility(r),
            "Sharpe":      sharpe_ratio(r),
            "Max DD":      max_drawdown(r),
            "Win Rate":    float((r > 0).mean()),
        })
    return pd.DataFrame(rows).set_index("Regime")
