import numpy as np
import pandas as pd


def annualized_return(returns: pd.Series) -> float:
    return returns.mean() * 252


def annualized_volatility(returns: pd.Series) -> float:
    return returns.std() * np.sqrt(252)


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
    excess = returns.mean() - risk_free / 252
    vol    = returns.std()
    return (excess / vol * np.sqrt(252)) if vol != 0 else 0.0


def sortino_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
    excess     = returns - risk_free / 252
    downside   = excess[excess < 0].std() * np.sqrt(252)
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


def summary(returns: pd.Series) -> dict:
    return {
        "Annualized Return":    f"{annualized_return(returns):.2%}",
        "Annualized Volatility":f"{annualized_volatility(returns):.2%}",
        "Sharpe Ratio":         f"{sharpe_ratio(returns):.2f}",
        "Sortino Ratio":        f"{sortino_ratio(returns):.2f}",
        "Calmar Ratio":         f"{calmar_ratio(returns):.2f}",
        "Max Drawdown":         f"{max_drawdown(returns):.2%}",
        "Win Rate":             f"{win_rate(returns):.2%}",
        "Total Return":         f"{(1 + returns).prod() - 1:.2%}",
    }
