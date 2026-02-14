import numpy as np
import pandas as pd
from scipy.stats import zscore
from config import MOM_LOOKBACK, MOM_SKIP, VOL_WINDOW


def momentum(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.shift(MOM_SKIP) / prices.shift(MOM_LOOKBACK) - 1


def realized_volatility(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().rolling(VOL_WINDOW).std() * np.sqrt(252)


def risk_adjusted_momentum(prices: pd.DataFrame) -> pd.DataFrame:
    mom = momentum(prices)
    vol = realized_volatility(prices).replace(0, np.nan)
    return mom / vol


def cross_section_zscore(series: pd.Series) -> pd.Series:
    valid = series.dropna()
    if len(valid) < 5:
        return pd.Series(np.nan, index=series.index)
    scores = zscore(valid)
    result = pd.Series(np.nan, index=series.index)
    result[valid.index] = scores
    return result


def build_signal(prices: pd.DataFrame) -> pd.DataFrame:
    raw = risk_adjusted_momentum(prices)
    return raw.apply(cross_section_zscore, axis=1)
