import numpy as np
import pandas as pd
from config import LONG_N, SHORT_N, REBALANCE_DAYS, COST_BPS, MIN_STOCKS


def build_weights(signal_row: pd.Series, long_n: int, short_n: int) -> pd.Series:
    valid = signal_row.dropna()
    if len(valid) < MIN_STOCKS:
        return pd.Series(0.0, index=signal_row.index)

    weights = pd.Series(0.0, index=signal_row.index)
    longs  = valid.nlargest(long_n).index
    shorts = valid.nsmallest(short_n).index

    weights[longs]  =  1.0 / long_n
    weights[shorts] = -1.0 / short_n
    return weights


def run_backtest(prices: pd.DataFrame, signal: pd.DataFrame) -> pd.Series:
    cost      = COST_BPS / 10_000
    rets      = prices.pct_change().fillna(0)
    dates     = prices.index
    holdings  = pd.Series(0.0, index=prices.columns)
    port_rets = []

    warmup = max(signal.notna().any(axis=1).idxmax(), dates[0])
    start_idx = dates.get_loc(warmup)

    for i in range(start_idx, len(dates), REBALANCE_DAYS):
        date     = dates[i]
        new_w    = build_weights(signal.loc[date], LONG_N, SHORT_N)
        turnover = (new_w - holdings).abs().sum()
        tc       = turnover * cost

        end_idx    = min(i + REBALANCE_DAYS, len(dates))
        period_ret = rets.iloc[i:end_idx]

        for day, row in period_ret.iterrows():
            daily = (new_w * row).sum() - tc / REBALANCE_DAYS
            port_rets.append((day, daily))

        holdings = new_w

    series = pd.Series(dict(port_rets)).sort_index()
    series.index = pd.to_datetime(series.index)
    return series


def equity_curve(returns: pd.Series, start_value: float = 100.0) -> pd.Series:
    return (1 + returns).cumprod() * start_value
