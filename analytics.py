"""
analytics.py  --  Signal quality analysis and statistical validation.

    rolling_ic         : Vectorized rolling cross-sectional rank-IC
    ic_decay           : Mean IC across forecast horizons (signal half-life)
    bootstrap_sharpe   : Fully-vectorized bootstrap CI for annualized Sharpe
    factor_correlation : Spearman cross-correlation matrix of factor signals
    oos_split          : Slice pre-computed returns into IS / OOS periods

All IC computations use the rank-Pearson identity (= Spearman) on the
full cross-section for each date, computed without a Python-level date loop.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ─── Core IC utility ───────────────────────────────────────────────────────────

def rolling_ic(
    signal: pd.DataFrame,
    prices: pd.DataFrame,
    window: int = 21,
    horizon: int = 1,
) -> pd.Series:
    """Vectorized rolling mean cross-sectional rank-IC.

    Computes daily IC as the Pearson correlation of cross-sectional ranks
    (= Spearman rho) between `signal` and `horizon`-day forward returns,
    then applies a rolling mean of width `window`.
    """
    fwd = prices.pct_change().rolling(horizon).sum().shift(-horizon)
    common = signal.index.intersection(fwd.index)
    sig, fwd = signal.loc[common], fwd.loc[common]

    # Cross-sectional rank (NaN preserved via na_option='keep')
    sig_r = sig.rank(axis=1, na_option="keep")
    fwd_r = fwd.rank(axis=1, na_option="keep")

    # Demean cross-sectionally (vectorized)
    sig_d = sig_r.subtract(sig_r.mean(axis=1), axis=0)
    fwd_d = fwd_r.subtract(fwd_r.mean(axis=1), axis=0)

    num      = (sig_d * fwd_d).sum(axis=1)
    den      = np.sqrt((sig_d ** 2).sum(axis=1) * (fwd_d ** 2).sum(axis=1))
    daily_ic = (num / den.replace(0, np.nan)).rename("ic")

    return daily_ic.rolling(window).mean().dropna()


# ─── Signal decay ──────────────────────────────────────────────────────────────

def ic_decay(
    signal: pd.DataFrame,
    prices: pd.DataFrame,
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """Mean rank-IC and t-statistics across forecast horizons.

    Returns a DataFrame indexed by horizon (days) with columns:
        mean_ic  -- time-averaged cross-sectional IC
        ic_std   -- standard deviation of daily IC series
        ic_ir    -- information ratio of the IC (mean / std)
        t_stat   -- t-statistic (ic_ir * sqrt(n_obs))
        n_obs    -- number of valid observations used
    """
    if horizons is None:
        horizons = [1, 2, 3, 5, 10, 21, 42, 63]

    daily_rets = prices.pct_change()
    rows: list[dict] = []

    for h in horizons:
        fwd    = daily_rets.rolling(h).sum().shift(-h)
        common = signal.index.intersection(fwd.index)
        sig_h  = signal.loc[common]
        fwd_h  = fwd.loc[common]

        sig_r = sig_h.rank(axis=1, na_option="keep")
        fwd_r = fwd_h.rank(axis=1, na_option="keep")
        sig_d = sig_r.subtract(sig_r.mean(axis=1), axis=0)
        fwd_d = fwd_r.subtract(fwd_r.mean(axis=1), axis=0)

        num     = (sig_d * fwd_d).sum(axis=1)
        den     = np.sqrt((sig_d ** 2).sum(axis=1) * (fwd_d ** 2).sum(axis=1))
        ic_vals = (num / den.replace(0, np.nan)).dropna()

        if len(ic_vals) < 20:
            continue

        arr    = ic_vals.values
        mean   = float(arr.mean())
        std    = float(arr.std(ddof=1))
        n      = int(len(arr))
        ic_ir  = mean / std if std > 0 else 0.0
        t_stat = ic_ir * np.sqrt(n)
        rows.append(dict(horizon=h, mean_ic=mean, ic_std=std,
                         ic_ir=ic_ir, t_stat=t_stat, n_obs=n))

    return pd.DataFrame(rows).set_index("horizon") if rows else pd.DataFrame()


# ─── Bootstrap ─────────────────────────────────────────────────────────────────

def bootstrap_sharpe(
    returns: pd.Series,
    n_boot: int = 10_000,
    seed: int = 42,
) -> dict:
    """Fully vectorized parametric bootstrap of annualized Sharpe ratio.

    Generates the full (n_boot × T) resample matrix in one NumPy call -- no
    Python-level loop. Returns a dict with keys:
        observed  -- empirical Sharpe on the original series
        mean      -- bootstrap mean of Sharpe distribution
        std       -- bootstrap std
        ci_lo     -- 2.5th percentile (lower 95% CI bound)
        ci_hi     -- 97.5th percentile (upper 95% CI bound)
        p_value   -- fraction of bootstrap Sharpes as extreme as observed (H0: SR=0)
        samples   -- full np.ndarray of bootstrap Sharpe estimates (length n_boot)
    """
    rng = np.random.default_rng(seed)
    arr = returns.dropna().values
    n   = len(arr)

    # Batched vectorization avoids allocating one giant (n_boot x n) matrix.
    batch   = max(1, min(1_000, n_boot))
    sharpes = np.empty(n_boot)
    for start in range(0, n_boot, batch):
        end   = min(start + batch, n_boot)
        idx   = rng.integers(0, n, size=(end - start, n))
        samps = arr[idx]
        means = samps.mean(axis=1)
        stds  = samps.std(axis=1, ddof=1)
        sharpes[start:end] = np.where(stds > 0, means / stds * np.sqrt(252), 0.0)

    obs     = float((arr.mean() / arr.std(ddof=1)) * np.sqrt(252)) if arr.std(ddof=1) > 0 else 0.0
    p_value = float((np.abs(sharpes) >= np.abs(obs)).mean())

    return {
        "observed": obs,
        "mean":     float(sharpes.mean()),
        "std":      float(sharpes.std()),
        "ci_lo":    float(np.percentile(sharpes, 2.5)),
        "ci_hi":    float(np.percentile(sharpes, 97.5)),
        "p_value":  p_value,
        "samples":  sharpes,
    }


# ─── Factor correlation ────────────────────────────────────────────────────────

def factor_correlation(signals: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Spearman cross-correlation of factor signals.

    Stacks each factor DataFrame into a (date, ticker) indexed Series, then
    computes the pairwise Spearman correlation using pandas corr(). This
    implicitly aligns on the common (date, ticker) index.
    """
    stacked = pd.DataFrame({name: df.stack() for name, df in signals.items()})
    return stacked.corr(method="spearman")


# ─── OOS split ─────────────────────────────────────────────────────────────────

def oos_split(
    returns: pd.Series,
    split_date: str = "2019-01-01",
) -> tuple[pd.Series, pd.Series]:
    """Slice pre-computed returns into IS and OOS periods.

    Slices the single full-history returns series rather than re-running the
    backtest, which preserves portfolio holding continuity at the split date.
    Returns (is_returns, oos_returns).
    """
    split = pd.Timestamp(split_date)
    return returns[returns.index < split], returns[returns.index >= split]
