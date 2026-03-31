"""
metrics.py  —  Performance analytics for strategy returns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def annualized_return(returns: pd.Series) -> float:
    return returns.mean() * 252


def annualized_volatility(returns: pd.Series) -> float:
    return returns.std() * np.sqrt(252)


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
    excess = returns.mean() - risk_free / 252
    vol    = returns.std()
    return (excess / vol * np.sqrt(252)) if vol != 0 else 0.0


def sortino_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
    excess   = returns - risk_free / 252
    downside = excess[excess < 0].std() * np.sqrt(252)
    return (annualized_return(returns) - risk_free) / downside if downside != 0 else 0.0


def max_drawdown(returns: pd.Series) -> float:
    equity = (1 + returns).cumprod()
    return (equity / equity.cummax() - 1).min()


def drawdown_series(returns: pd.Series) -> pd.Series:
    equity = (1 + returns).cumprod()
    return (equity / equity.cummax() - 1) * 100


def calmar_ratio(returns: pd.Series) -> float:
    mdd = abs(max_drawdown(returns))
    return annualized_return(returns) / mdd if mdd != 0 else 0.0


def win_rate(returns: pd.Series) -> float:
    return (returns > 0).sum() / len(returns)


def tail_ratio(returns: pd.Series, cutoff: float = 0.05) -> float:
    """Ratio of right-tail 95th pct to left-tail 5th pct (abs). > 1 = right-skewed P&L."""
    p95 = np.percentile(returns, 100 * (1 - cutoff))
    p05 = np.percentile(returns, 100 * cutoff)
    return abs(p95 / p05) if p05 != 0 else 0.0


def var_95(returns: pd.Series) -> float:
    """Historical 1-day Value at Risk at 95% confidence (negative number)."""
    return float(np.percentile(returns, 5))


def cvar_95(returns: pd.Series) -> float:
    """Conditional VaR (Expected Shortfall) at 95% confidence."""
    threshold = np.percentile(returns, 5)
    tail = returns[returns <= threshold]
    return float(tail.mean()) if len(tail) > 0 else 0.0


def return_skewness(returns: pd.Series) -> float:
    return float(scipy_stats.skew(returns.dropna()))


def return_kurtosis(returns: pd.Series) -> float:
    """Excess kurtosis (normal = 0)."""
    return float(scipy_stats.kurtosis(returns.dropna()))


def summary(returns: pd.Series) -> dict:
    return {
        "Annualized Return":     f"{annualized_return(returns):.2%}",
        "Annualized Volatility": f"{annualized_volatility(returns):.2%}",
        "Sharpe Ratio":          f"{sharpe_ratio(returns):.2f}",
        "Sortino Ratio":         f"{sortino_ratio(returns):.2f}",
        "Calmar Ratio":          f"{calmar_ratio(returns):.2f}",
        "Max Drawdown":          f"{max_drawdown(returns):.2%}",
        "Win Rate":              f"{win_rate(returns):.2%}",
        "Total Return":          f"{(1 + returns).prod() - 1:.2%}",
        "Tail Ratio":            f"{tail_ratio(returns):.2f}",
        "Return Skewness":       f"{return_skewness(returns):.3f}",
        "Excess Kurtosis":       f"{return_kurtosis(returns):.3f}",
        "VaR 95% (daily)":       f"{var_95(returns):.3%}",
        "CVaR 95% (daily)":      f"{cvar_95(returns):.3%}",
    }
