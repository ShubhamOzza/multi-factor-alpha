"""
visualize.py  --  Professional quantitative research report generator.

Report 1  report.png           (performance tear-sheet)
  Row 0 (full):  Equity curve with vol-regime shading + metrics card
  Row 1 (L/R):   Underwater chart  |  Rolling Sharpe (63d & 252d)
  Row 2 (full):  Monthly returns heatmap
  Row 3 (L/R):   Return distribution  |  Rolling composite rank-IC

Report 2  report_analytics.png (signal analytics & statistical validation)
  Row 0 (L/R):   IC decay bar chart  |  Factor cross-correlation heatmap
  Row 1 (full):  Rolling per-factor IC (all 3 factors)
  Row 2 (L/R):   Vol-regime Sharpe bars  |  Bootstrap Sharpe distribution
  Row 3 (full):  In-sample vs out-of-sample equity curves
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats as scipy_stats

from analytics import rolling_ic
from metrics import (
    drawdown_series, annualized_return, annualized_volatility,
    sharpe_ratio, sortino_ratio, max_drawdown, win_rate,
)

# ─── Colour palette ────────────────────────────────────────────────────────────
BG      = "#090E1A"
PANEL   = "#0F172A"
GRID    = "#1E293B"
BORDER  = "#1E3A5F"
TEXT    = "#E2E8F0"
DIM     = "#64748B"
GREEN   = "#10B981"
RED     = "#F43F5E"
BLUE    = "#3B82F6"
GOLD    = "#F59E0B"
PURPLE  = "#A78BFA"
TEAL    = "#06B6D4"

plt.rcParams.update({
    "figure.facecolor":    BG,
    "axes.facecolor":      PANEL,
    "axes.edgecolor":      BORDER,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "axes.spines.left":    True,
    "axes.spines.bottom":  True,
    "grid.color":          GRID,
    "grid.linewidth":      0.5,
    "font.family":         "DejaVu Sans",
    "font.size":           9,
    "text.color":          TEXT,
    "axes.labelcolor":     DIM,
    "axes.titlecolor":     TEXT,
    "xtick.color":         DIM,
    "ytick.color":         DIM,
    "xtick.labelsize":     8,
    "ytick.labelsize":     8,
    "legend.facecolor":    PANEL,
    "legend.edgecolor":    BORDER,
    "legend.fontsize":     8,
    "legend.framealpha":   0.88,
})


# ─── Shared helpers ────────────────────────────────────────────────────────────

def _style(ax, title: str, ylabel: str = "", xlabel: str = "") -> None:
    ax.set_title(title, fontsize=9, fontweight="bold", pad=9, color=TEXT, loc="left")
    ax.set_ylabel(ylabel, fontsize=8, color=DIM)
    ax.set_xlabel(xlabel, fontsize=8, color=DIM)
    ax.yaxis.grid(True, alpha=0.35, linestyle=":")
    ax.xaxis.grid(False)
    ax.tick_params(axis="both", length=3, color=BORDER)


def _shade_regimes(ax, regimes: pd.Series) -> None:
    """Draw translucent background bands for Low/High vol regimes."""
    col_map = {"Low Vol": GREEN, "High Vol": RED}
    group_id = (regimes != regimes.shift()).cumsum()
    for _, grp in regimes.groupby(group_id):
        label = grp.iloc[0]
        if label not in col_map:
            continue
        ax.axvspan(grp.index[0], grp.index[-1],
                   alpha=0.10, color=col_map[label], zorder=1, linewidth=0)


# ══════════════════════════════════════════════════════════════════════════════
# REPORT 1 — Performance tear-sheet
# ══════════════════════════════════════════════════════════════════════════════

def _panel_equity(ax, equity: pd.Series, returns: pd.Series,
                  regimes: pd.Series | None = None) -> None:
    if regimes is not None:
        _shade_regimes(ax, regimes.reindex(equity.index, method="nearest"))

    ax.plot(equity.index, equity.values, color=GREEN, linewidth=2.0, zorder=4)
    ax.fill_between(equity.index, 100, equity.values,
                    where=equity.values >= 100, alpha=0.18, color=GREEN, zorder=2)
    ax.fill_between(equity.index, 100, equity.values,
                    where=equity.values < 100,  alpha=0.24, color=RED,   zorder=2)
    ax.axhline(100, color=DIM, linewidth=0.7, linestyle="--", alpha=0.55, zorder=3)

    peak_date = equity.idxmax()
    ax.annotate(f"  Peak: ${equity.max():.1f}",
                xy=(peak_date, equity.max()), fontsize=7.5, color=GOLD,
                xytext=(0, 7), textcoords="offset points", ha="center", zorder=5)

    ann_r = annualized_return(returns)
    ann_v = annualized_volatility(returns)
    sr    = sharpe_ratio(returns)
    so    = sortino_ratio(returns)
    mdd   = max_drawdown(returns)
    wr    = win_rate(returns)
    tot   = (1 + returns).prod() - 1

    card = (
        f"  Ann. Return     {ann_r:+.2%}\n"
        f"  Ann. Volatility  {ann_v:.2%}\n"
        f"  Sharpe Ratio    {sr:.2f}\n"
        f"  Sortino Ratio   {so:.2f}\n"
        f"  Max Drawdown    {mdd:.2%}\n"
        f"  Win Rate        {wr:.1%}\n"
        f"  Total Return    {tot:+.1%}"
    )
    ax.text(0.013, 0.97, card, transform=ax.transAxes,
            fontsize=8, family="monospace", color=TEXT, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.55", facecolor=PANEL,
                      edgecolor=BORDER, alpha=0.92, linewidth=0.8))

    if regimes is not None:
        from matplotlib.patches import Patch
        leg = [Patch(facecolor=GREEN, alpha=0.35, label="Low Vol regime"),
               Patch(facecolor=RED,   alpha=0.35, label="High Vol regime")]
        ax.legend(handles=leg, loc="upper right", fontsize=7.5)

    _style(ax, "EQUITY CURVE  --  Long / Short Momentum  --  Starting NAV = $100",
           ylabel="Portfolio NAV ($)")


def _panel_drawdown(ax, dd: pd.Series) -> None:
    ax.fill_between(dd.index, dd.values, 0, color=RED, alpha=0.30, zorder=2)
    ax.plot(dd.index, dd.values, color=RED, linewidth=1.1, zorder=3)
    ax.axhline(0, color=BORDER, linewidth=0.6)
    peak_dt = dd.idxmin()
    ax.annotate(f"{dd.min():.1f}%",
                xy=(peak_dt, dd.min()), fontsize=7.5, color=RED,
                xytext=(0, -12), textcoords="offset points", ha="center")
    _style(ax, "UNDERWATER CHART", ylabel="Drawdown (%)")


def _panel_sharpe(ax, returns: pd.Series) -> None:
    rs63  = (returns.rolling(63).mean()  / returns.rolling(63).std()  * np.sqrt(252)).dropna()
    rs252 = (returns.rolling(252).mean() / returns.rolling(252).std() * np.sqrt(252)).dropna()

    ax.plot(rs63.index,  rs63.values,  color=BLUE, linewidth=1.1, label="63-day",  alpha=0.85, zorder=3)
    ax.plot(rs252.index, rs252.values, color=GOLD, linewidth=1.6, label="252-day", alpha=0.90, zorder=4)
    ax.fill_between(rs63.index, 0, rs63.values,
                    where=rs63.values > 0, alpha=0.08, color=BLUE, zorder=2)
    ax.axhline(0,   color=DIM,   linewidth=0.7, linestyle="--", alpha=0.55)
    ax.axhline(1.0, color=GREEN, linewidth=0.6, linestyle=":",  alpha=0.40)
    ax.legend(loc="upper left")
    _style(ax, "ROLLING SHARPE RATIO", ylabel="Sharpe")


def _panel_heatmap(ax, returns: pd.Series) -> None:
    mo = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1) * 100
    df = pd.DataFrame({"ret": mo, "year": mo.index.year, "month": mo.index.month})
    pivot = df.pivot(index="year", columns="month", values="ret")

    annual = returns.resample("YE").apply(lambda x: (1 + x).prod() - 1) * 100
    annual.index = annual.index.year
    pivot["Full Year"] = annual

    n_cols = len(pivot.columns)
    col_labels = (["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec","Full Year"][:n_cols])
    pivot.columns = col_labels

    vals   = pivot.values.astype(float)
    finite = vals[np.isfinite(vals)]
    vmax   = max(np.percentile(np.abs(finite), 92), 2.0) if len(finite) else 5.0

    cmap = LinearSegmentedColormap.from_list(
        "rg", ["#7B0D1E", "#1B0A2E", PANEL, "#0A3D24", "#064E3B"], N=512)
    disp = np.where(np.isfinite(vals), vals, 0.0)
    ax.imshow(disp, cmap=cmap, aspect="auto", vmin=-vmax, vmax=vmax, zorder=1)

    for i in range(vals.shape[0] + 1):
        ax.axhline(i - 0.5, color=BG, linewidth=1.2, zorder=3)
    for j in range(vals.shape[1] + 1):
        ax.axvline(j - 0.5, color=BG, linewidth=1.2, zorder=3)
    ax.axvline(11.5, color=GOLD, linewidth=2.0, alpha=0.65, zorder=4)

    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            if np.isfinite(v):
                txt = TEXT if abs(v) / vmax > 0.30 else DIM
                bold = "bold" if j == vals.shape[1] - 1 else "normal"
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center",
                        fontsize=7.2, color=txt, fontweight=bold, zorder=5)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=8, color=TEXT)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8, color=TEXT)
    ax.tick_params(length=0)
    _style(ax, "MONTHLY RETURNS HEATMAP  (%)")


def _panel_distribution(ax, returns: pd.Series) -> None:
    ret_pct = returns.dropna().values * 100
    mu, sigma = ret_pct.mean(), ret_pct.std()

    ax.hist(ret_pct, bins=75, color=BLUE, alpha=0.60,
            edgecolor=PANEL, linewidth=0.3, density=True, zorder=2)
    xs = np.linspace(ret_pct.min(), ret_pct.max(), 400)
    ax.plot(xs, scipy_stats.norm.pdf(xs, mu, sigma),
            color=GOLD, linewidth=1.5, linestyle="--", label="Normal fit", zorder=3)

    var95 = np.percentile(ret_pct, 5)
    ax.axvline(var95, color=RED, linewidth=1.3, linestyle="--", alpha=0.9,
               label=f"VaR 95%  {var95:.2f}%", zorder=4)
    ax.axvline(0, color=DIM, linewidth=0.7, linestyle=":", alpha=0.6)

    skew = scipy_stats.skew(ret_pct)
    kurt = scipy_stats.kurtosis(ret_pct)
    ax.text(0.97, 0.96,
            f"mu     {mu:.3f}%\nsigma  {sigma:.3f}%\nskew   {skew:.3f}\nkurt   {kurt:.3f}",
            transform=ax.transAxes, fontsize=7.5, family="monospace",
            va="top", ha="right", color=TEXT,
            bbox=dict(boxstyle="round,pad=0.45", facecolor=PANEL,
                      edgecolor=BORDER, alpha=0.92))
    ax.legend(loc="upper left", fontsize=7.5)
    _style(ax, "DAILY RETURN DISTRIBUTION", ylabel="Density", xlabel="Daily Return (%)")


def _panel_ic(ax, signal: pd.DataFrame, prices: pd.DataFrame) -> None:
    ic = rolling_ic(signal, prices, window=21, horizon=1)
    if ic.empty:
        ax.text(0.5, 0.5, "Insufficient data for IC",
                transform=ax.transAxes, ha="center", color=DIM)
        _style(ax, "ROLLING RANK IC")
        return

    ax.plot(ic.index, ic.values, color=PURPLE, linewidth=1.2, alpha=0.9, zorder=3)
    ax.fill_between(ic.index, 0, ic.values,
                    where=ic.values >= 0, alpha=0.18, color=GREEN, zorder=2)
    ax.fill_between(ic.index, 0, ic.values,
                    where=ic.values < 0, alpha=0.18, color=RED, zorder=2)
    ax.axhline(0,    color=DIM,  linewidth=0.7, linestyle="--", alpha=0.6)
    ax.axhline(0.05, color=TEAL, linewidth=0.6, linestyle=":",  alpha=0.45)

    mean_ic = float(ic.mean())
    ic_std  = float(ic.std())
    ic_ir   = mean_ic / ic_std if ic_std > 0 else 0.0
    clr     = GREEN if mean_ic > 0 else RED
    ax.text(0.015, 0.95, f"Mean IC = {mean_ic:.4f}   IC-IR = {ic_ir:.2f}",
            transform=ax.transAxes, fontsize=8, family="monospace", color=clr, va="top")
    _style(ax, "ROLLING 21-DAY RANK IC  --  Composite Signal vs Next-Day Returns",
           ylabel="Spearman rho")


def plot_full_report(returns: pd.Series, signal: pd.DataFrame, prices: pd.DataFrame,
                     regimes: pd.Series | None = None) -> None:
    """Generate the performance tear-sheet (report.png)."""
    equity = (1 + returns).cumprod() * 100
    dd     = drawdown_series(returns)

    fig = plt.figure(figsize=(16, 19))
    fig.patch.set_facecolor(BG)

    fig.text(0.50, 0.988, "MULTI-FACTOR EQUITY ALPHA",
             ha="center", va="top", fontsize=22, fontweight="bold", color=TEXT)
    fig.text(0.50, 0.975,
             "Risk-Adjusted Momentum  --  Long/Short Equity  --  S&P 500 Universe  --  2014-2024",
             ha="center", va="top", fontsize=10, color=DIM)

    gs = gridspec.GridSpec(4, 2, figure=fig,
                           height_ratios=[3.0, 1.8, 2.4, 1.8],
                           hspace=0.50, wspace=0.28,
                           top=0.966, bottom=0.033, left=0.057, right=0.972)

    _panel_equity(      fig.add_subplot(gs[0, :]), equity, returns, regimes)
    _panel_drawdown(    fig.add_subplot(gs[1, 0]), dd)
    _panel_sharpe(      fig.add_subplot(gs[1, 1]), returns)
    _panel_heatmap(     fig.add_subplot(gs[2, :]), returns)
    _panel_distribution(fig.add_subplot(gs[3, 0]), returns)
    _panel_ic(          fig.add_subplot(gs[3, 1]), signal, prices)

    fig.text(0.972, 0.010,
             "Multi-Factor Alpha Engine  --  Systematic Long/Short Equity Research",
             ha="right", va="bottom", fontsize=6.5, color=DIM, style="italic")

    plt.savefig("report.png", dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("  Saved -> report.png")


# ══════════════════════════════════════════════════════════════════════════════
# REPORT 2 — Signal analytics & statistical validation
# ══════════════════════════════════════════════════════════════════════════════

def _panel_ic_decay(ax, df: pd.DataFrame) -> None:
    if df.empty:
        ax.text(0.5, 0.5, "No IC data", transform=ax.transAxes,
                ha="center", color=DIM)
        _style(ax, "IC DECAY")
        return

    horizons = df.index.values
    means    = df["mean_ic"].values
    se       = (df["ic_std"] / np.sqrt(df["n_obs"])).values

    colors = [GREEN if m > 0 else RED for m in means]
    ax.bar(range(len(horizons)), means, color=colors, alpha=0.75,
           edgecolor=PANEL, linewidth=0.5, zorder=2)
    ax.errorbar(range(len(horizons)), means, yerr=se * 1.96,
                fmt="none", color=TEXT, linewidth=1.2, capsize=3, zorder=3)
    ax.axhline(0, color=DIM, linewidth=0.7, linestyle="--", alpha=0.6)

    ax.set_xticks(range(len(horizons)))
    ax.set_xticklabels([f"{h}d" for h in horizons], fontsize=8)
    _style(ax, "IC DECAY CURVE  --  Signal Forecast Horizon",
           ylabel="Mean Rank-IC", xlabel="Forward Horizon")

    # t-stat annotations
    for i, (m, t) in enumerate(zip(means, df["t_stat"].values)):
        if not np.isnan(m):
            ax.text(i, m + np.sign(m) * 0.003, f"t={t:.1f}",
                    ha="center", va="bottom" if m >= 0 else "top",
                    fontsize=6.5, color=DIM, zorder=4)


def _panel_factor_corr(ax, corr: pd.DataFrame) -> None:
    names = corr.columns.tolist()
    vals  = corr.values.astype(float)

    cmap = LinearSegmentedColormap.from_list("rg", [RED, PANEL, BLUE], N=256)
    ax.imshow(vals, cmap=cmap, aspect="auto", vmin=-1, vmax=1, zorder=1)

    for i in range(len(names) + 1):
        ax.axhline(i - 0.5, color=BG, linewidth=1.0, zorder=3)
    for j in range(len(names) + 1):
        ax.axvline(j - 0.5, color=BG, linewidth=1.0, zorder=3)

    for i in range(len(names)):
        for j in range(len(names)):
            v = vals[i, j]
            bold = "bold" if i == j else "normal"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=9.5, color=TEXT, fontweight=bold, zorder=4)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=8, rotation=15, ha="right")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.tick_params(length=0)
    _style(ax, "FACTOR CROSS-CORRELATION  (Spearman)")


def _panel_rolling_factor_ic(ax, roll_ic_df: pd.DataFrame) -> None:
    palette = [BLUE, GOLD, PURPLE, TEAL]
    for col, clr in zip(roll_ic_df.columns, palette):
        ax.plot(roll_ic_df.index, roll_ic_df[col].values,
                color=clr, linewidth=1.2, label=col, alpha=0.88, zorder=3)

    ax.axhline(0,    color=DIM,  linewidth=0.7, linestyle="--", alpha=0.6)
    ax.axhline(0.05, color=TEAL, linewidth=0.5, linestyle=":",  alpha=0.40)
    ax.legend(loc="upper right")
    _style(ax, "ROLLING 63-DAY RANK IC  --  Per-Factor", ylabel="Spearman rho")


def _panel_regime_bars(ax, reg_stats: pd.DataFrame) -> None:
    regimes = reg_stats.index.tolist()
    sharpes = reg_stats["Sharpe"].values
    colors  = [GREEN if s > 0 else RED for s in sharpes]

    bars = ax.barh(regimes, sharpes, color=colors, alpha=0.78,
                   edgecolor=PANEL, linewidth=0.5, height=0.55, zorder=2)
    ax.axvline(0, color=DIM, linewidth=0.7, linestyle="--", alpha=0.6)

    for bar, v, idx in zip(bars, sharpes, reg_stats.index):
        days = int(reg_stats.loc[idx, "Days"])
        ann_r = float(reg_stats.loc[idx, "Ann. Return"])
        sign  = 1 if v >= 0 else -1
        ax.text(v + sign * 0.025, bar.get_y() + bar.get_height() / 2,
                f"SR={v:.2f}  Ret={ann_r:+.1%}  n={days}d",
                va="center", ha="left" if v >= 0 else "right",
                fontsize=7.5, color=TEXT, zorder=4)

    _style(ax, "SHARPE RATIO BY VOLATILITY REGIME", xlabel="Annualized Sharpe Ratio")


def _panel_bootstrap(ax, boot: dict) -> None:
    samples = boot["samples"]
    ax.hist(samples, bins=80, color=BLUE, alpha=0.62,
            edgecolor=PANEL, linewidth=0.3, density=True, zorder=2)

    obs = boot["observed"]
    ax.axvline(obs,          color=GREEN, linewidth=2.0, linestyle="-",
               label=f"Observed SR={obs:.2f}", zorder=4)
    ax.axvline(boot["ci_lo"], color=GOLD, linewidth=1.3, linestyle="--",
               label=f"95% CI  [{boot['ci_lo']:.2f}, {boot['ci_hi']:.2f}]", zorder=4)
    ax.axvline(boot["ci_hi"], color=GOLD, linewidth=1.3, linestyle="--", zorder=4)
    ax.axvline(0, color=DIM, linewidth=0.8, linestyle=":", alpha=0.6, zorder=3)

    clr = GREEN if boot["p_value"] < 0.05 else RED
    ax.text(0.97, 0.96, f"p-value = {boot['p_value']:.3f}",
            transform=ax.transAxes, fontsize=8.5, family="monospace",
            va="top", ha="right", color=clr)
    ax.legend(loc="upper left", fontsize=7.5)
    _style(ax, "BOOTSTRAP SHARPE RATIO  (n=10,000 resamples)",
           ylabel="Density", xlabel="Annualized Sharpe")


def _panel_oos(ax, is_ret: pd.Series, oos_ret: pd.Series,
               split_date: pd.Timestamp) -> None:
    # Both curves start at 100 independently (standard in academic factor research)
    is_eq  = (1 + is_ret).cumprod() * 100
    oos_eq = (1 + oos_ret).cumprod() * 100

    ax.plot(is_eq.index,  is_eq.values,  color=BLUE, linewidth=1.8,
            label="In-Sample",      zorder=4)
    ax.plot(oos_eq.index, oos_eq.values, color=GOLD, linewidth=1.8,
            label="Out-of-Sample",  zorder=4)

    ax.fill_between(is_eq.index, 100, is_eq.values,
                    where=is_eq.values >= 100, alpha=0.12, color=BLUE, zorder=2)
    ax.fill_between(oos_eq.index, 100, oos_eq.values,
                    where=oos_eq.values >= 100, alpha=0.12, color=GOLD, zorder=2)
    ax.fill_between(oos_eq.index, 100, oos_eq.values,
                    where=oos_eq.values < 100, alpha=0.12, color=RED, zorder=2)

    ax.axhline(100, color=DIM, linewidth=0.6, linestyle=":", alpha=0.5)
    ax.axvline(split_date, color=DIM, linewidth=1.2, linestyle="--", alpha=0.7, zorder=3)

    is_sr  = sharpe_ratio(is_ret)
    is_r   = annualized_return(is_ret)
    oos_sr = sharpe_ratio(oos_ret)
    oos_r  = annualized_return(oos_ret)

    ax.text(0.02, 0.97,
            f"In-Sample (2014-2018)     Ann.Ret={is_r:+.2%}  Sharpe={is_sr:.2f}",
            transform=ax.transAxes, fontsize=8, family="monospace",
            va="top", color=BLUE)
    ax.text(0.02, 0.89,
            f"Out-of-Sample (2019-2023) Ann.Ret={oos_r:+.2%}  Sharpe={oos_sr:.2f}",
            transform=ax.transAxes, fontsize=8, family="monospace",
            va="top", color=GOLD)

    ax.legend(loc="upper right")
    _style(ax, "IN-SAMPLE vs OUT-OF-SAMPLE  --  Both Normalized to $100",
           ylabel="NAV ($)")


def plot_analytics_report(
    ic_decay_df: pd.DataFrame,
    factor_corr: pd.DataFrame,
    bootstrap: dict,
    reg_stats: pd.DataFrame,
    is_returns: pd.Series,
    oos_returns: pd.Series,
    roll_factor_ic: pd.DataFrame,
    split_date: pd.Timestamp,
) -> None:
    """Generate the signal analytics & statistical validation report (report_analytics.png)."""
    fig = plt.figure(figsize=(16, 19))
    fig.patch.set_facecolor(BG)

    fig.text(0.50, 0.988, "SIGNAL ANALYTICS & STATISTICAL VALIDATION",
             ha="center", va="top", fontsize=20, fontweight="bold", color=TEXT)
    fig.text(0.50, 0.975,
             "IC Decay  --  Factor Correlation  --  Regime Analysis  --  Bootstrap Validation  --  IS / OOS Split",
             ha="center", va="top", fontsize=10, color=DIM)

    gs = gridspec.GridSpec(4, 2, figure=fig,
                           height_ratios=[2.2, 2.0, 2.0, 2.5],
                           hspace=0.50, wspace=0.30,
                           top=0.966, bottom=0.033, left=0.057, right=0.972)

    _panel_ic_decay(          fig.add_subplot(gs[0, 0]), ic_decay_df)
    _panel_factor_corr(       fig.add_subplot(gs[0, 1]), factor_corr)
    _panel_rolling_factor_ic( fig.add_subplot(gs[1, :]), roll_factor_ic)
    _panel_regime_bars(       fig.add_subplot(gs[2, 0]), reg_stats)
    _panel_bootstrap(         fig.add_subplot(gs[2, 1]), bootstrap)
    _panel_oos(               fig.add_subplot(gs[3, :]), is_returns, oos_returns, split_date)

    fig.text(0.972, 0.010,
             "Multi-Factor Alpha Engine  --  Systematic Long/Short Equity Research",
             ha="right", va="bottom", fontsize=6.5, color=DIM, style="italic")

    plt.savefig("report_analytics.png", dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("  Saved -> report_analytics.png")


# ─── Utility ───────────────────────────────────────────────────────────────────

def export_equity_json(returns: pd.Series) -> None:
    equity  = (1 + returns).cumprod() * 100
    sampled = equity.resample("W").last().dropna()
    values  = [round(float(v), 4) for v in sampled.values]
    with open("equity_curve.json", "w") as f:
        f.write("const EQUITY_CURVE = " + str(values) + ";")
    print("  Saved -> equity_curve.json")
