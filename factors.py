"""
factors.py  —  Multi-factor signal construction.

Factors implemented (all cross-sectionally z-scored before combination):
  1. Risk-adjusted momentum   (Barroso & Santa-Clara 2015)
  2. Short-term mean reversion (Jegadeesh 1990)
  3. Low-volatility anomaly   (Baker, Bradley & Wurgler 2011)

Composite signal: equal-weight average, then re-z-scored cross-sectionally.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import zscore

from config import MOM_LOOKBACK, MOM_SKIP, VOL_WINDOW


def _cross_section_zscore(series: pd.Series) -> pd.Series:
    valid = series.dropna()
    if len(valid) < 5:
        return pd.Series(np.nan, index=series.index)
    result = pd.Series(np.nan, index=series.index)
    result[valid.index] = zscore(valid)
    return result


# ─── Individual factors ────────────────────────────────────────────────────────

def momentum(prices: pd.DataFrame) -> pd.DataFrame:
    """12-month price momentum, skipping most recent month (Jegadeesh & Titman 1993)."""
    return prices.shift(MOM_SKIP) / prices.shift(MOM_LOOKBACK) - 1


def realized_volatility(prices: pd.DataFrame) -> pd.DataFrame:
    """Annualized 21-day realized volatility."""
    return prices.pct_change().rolling(VOL_WINDOW).std() * np.sqrt(252)


def risk_adjusted_momentum(prices: pd.DataFrame) -> pd.DataFrame:
    """Momentum scaled by realized volatility (Barroso & Santa-Clara 2015)."""
    vol = realized_volatility(prices).replace(0, np.nan)
    return momentum(prices) / vol


def short_term_reversal(prices: pd.DataFrame) -> pd.DataFrame:
    """1-month mean-reversion (Jegadeesh 1990). Short the recent winners."""
    return -(prices / prices.shift(MOM_SKIP) - 1)


def low_volatility(prices: pd.DataFrame) -> pd.DataFrame:
    """Low-volatility anomaly (Baker et al. 2011). Prefer low-vol stocks."""
    return -realized_volatility(prices)


# ─── Composite signal ──────────────────────────────────────────────────────────

def build_signal(prices: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight composite of risk-adj momentum, reversal, and low-vol.

    Each factor is z-scored cross-sectionally before averaging, then the
    composite is z-scored again to maintain clean rank ordering.
    """
    raw = {
        "mom": risk_adjusted_momentum(prices),
        "rev": short_term_reversal(prices),
        "lvol": low_volatility(prices),
    }
    zscored = {k: v.apply(_cross_section_zscore, axis=1) for k, v in raw.items()}
    composite = (zscored["mom"] + zscored["rev"] + zscored["lvol"]) / 3
    return composite.apply(_cross_section_zscore, axis=1)


def get_factor_signals(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return individually z-scored factor DataFrames for IC attribution."""
    return {
        "Risk-Adj Momentum": risk_adjusted_momentum(prices).apply(_cross_section_zscore, axis=1),
        "ST Reversal":       short_term_reversal(prices).apply(_cross_section_zscore, axis=1),
        "Low Volatility":    low_volatility(prices).apply(_cross_section_zscore, axis=1),
    }


# Backward-compat alias
def cross_section_zscore(series: pd.Series) -> pd.Series:
    return _cross_section_zscore(series)
