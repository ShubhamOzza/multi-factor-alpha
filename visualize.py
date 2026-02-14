import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from metrics import drawdown_series

plt.rcParams.update({
    "figure.facecolor":  "#FAFAF8",
    "axes.facecolor":    "#FAFAF8",
    "axes.edgecolor":    "#D0CBBF",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "grid.color":        "#E8E4DC",
    "grid.linewidth":    0.6,
    "font.family":       "sans-serif",
    "text.color":        "#1A1916",
    "axes.labelcolor":   "#1A1916",
    "xtick.color":       "#6B6760",
    "ytick.color":       "#6B6760",
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
})

GREEN  = "#1D6F4E"
RED    = "#C0392B"
BLUE   = "#2B5BAD"
MUTED  = "#6B6760"


def plot_full_report(returns: pd.Series, signal: pd.DataFrame, prices: pd.DataFrame):
    equity  = (1 + returns).cumprod() * 100
    dd      = drawdown_series(returns)
    rolling_sharpe = (
        returns.rolling(63).mean() / returns.rolling(63).std() * np.sqrt(252)
    ).dropna()
    monthly = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)

    fig = plt.figure(figsize=(14, 11))
    fig.suptitle("Multi-Factor Equity Alpha — Backtest Report", fontsize=14, fontweight="bold", y=0.98)
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.48, wspace=0.32)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(equity.index, equity.values, color=GREEN, linewidth=1.6)
    ax1.fill_between(equity.index, 100, equity.values, where=equity.values > 100,
                     alpha=0.12, color=GREEN)
    ax1.fill_between(equity.index, 100, equity.values, where=equity.values <= 100,
                     alpha=0.12, color=RED)
    ax1.axhline(100, color=MUTED, linewidth=0.8, linestyle="--")
    ax1.set_title("Equity Curve  (starting $100)", fontsize=10, fontweight="semibold", pad=8)
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.yaxis.grid(True)

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.fill_between(dd.index, dd.values, 0, color=RED, alpha=0.45)
    ax2.plot(dd.index, dd.values, color=RED, linewidth=1.2)
    ax2.set_title("Drawdown (%)", fontsize=10, fontweight="semibold", pad=8)
    ax2.set_ylabel("Drawdown (%)")
    ax2.yaxis.grid(True)

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(rolling_sharpe.index, rolling_sharpe.values, color=BLUE, linewidth=1.3)
    ax3.axhline(0, color=MUTED, linewidth=0.8, linestyle="--")
    ax3.fill_between(rolling_sharpe.index, 0, rolling_sharpe.values,
                     where=rolling_sharpe.values > 0, alpha=0.12, color=BLUE)
    ax3.set_title("Rolling 63-Day Sharpe Ratio", fontsize=10, fontweight="semibold", pad=8)
    ax3.set_ylabel("Sharpe")
    ax3.yaxis.grid(True)

    ax4 = fig.add_subplot(gs[2, 0])
    colors = [GREEN if r >= 0 else RED for r in monthly.values]
    ax4.bar(monthly.index, monthly.values * 100, color=colors, width=20, alpha=0.85)
    ax4.axhline(0, color=MUTED, linewidth=0.8)
    ax4.set_title("Monthly Returns (%)", fontsize=10, fontweight="semibold", pad=8)
    ax4.set_ylabel("Return (%)")
    ax4.yaxis.grid(True)

    ax5 = fig.add_subplot(gs[2, 1])
    ret_vals = returns.values * 100
    ax5.hist(ret_vals, bins=60, color=BLUE, alpha=0.75, edgecolor="white", linewidth=0.4)
    ax5.axvline(0, color=RED, linewidth=1.2, linestyle="--")
    ax5.set_title("Daily Return Distribution (%)", fontsize=10, fontweight="semibold", pad=8)
    ax5.set_xlabel("Daily Return (%)")
    ax5.set_ylabel("Frequency")
    ax5.yaxis.grid(True)

    plt.savefig("report.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved → report.png")


def export_equity_json(returns: pd.Series):
    equity = (1 + returns).cumprod() * 100
    sampled = equity.resample("W").last().dropna()
    values  = [round(v, 4) for v in sampled.values.tolist()]
    with open("equity_curve.json", "w") as f:
        f.write("const EQUITY_CURVE = " + str(values) + ";")
    print("  Saved → equity_curve.json  (paste into index.html)")
