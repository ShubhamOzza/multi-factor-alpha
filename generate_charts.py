"""
generate_charts.py  --  Produce individual chart PNGs for the README.

Saves 12 standalone charts to the charts/ directory. Run after main.py so
prices.parquet is already cached. All charts use the same dark theme as the
full report pages.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats as scipy_stats

# ── project imports ────────────────────────────────────────────────────────────
from data        import fetch_prices
from factors     import build_signal, get_factor_signals
from portfolio   import run_backtest
from metrics     import (drawdown_series, annualized_return, annualized_volatility,
                          sharpe_ratio, sortino_ratio, max_drawdown, win_rate)
from analytics   import (rolling_ic, ic_decay, bootstrap_sharpe,
                          factor_correlation, oos_split)
from regime      import classify_regimes, regime_stats
from attribution import rolling_factor_ic
from config      import OOS_START

# ── theme ──────────────────────────────────────────────────────────────────────
BG     = "#090E1A"; PANEL  = "#0F172A"; GRID   = "#1E293B"; BORDER = "#1E3A5F"
TEXT   = "#E2E8F0"; DIM    = "#64748B"
GREEN  = "#10B981"; RED    = "#F43F5E"; BLUE   = "#3B82F6"
GOLD   = "#F59E0B"; PURPLE = "#A78BFA"; TEAL   = "#06B6D4"

plt.rcParams.update({
    "figure.facecolor": BG,   "axes.facecolor":  PANEL, "axes.edgecolor":    BORDER,
    "axes.spines.top":  False,"axes.spines.right":False,"grid.color":        GRID,
    "grid.linewidth":   0.5,  "font.family":    "DejaVu Sans","font.size":    9,
    "text.color":       TEXT, "axes.labelcolor": DIM,   "axes.titlecolor":   TEXT,
    "xtick.color":      DIM,  "ytick.color":     DIM,   "xtick.labelsize":   8,
    "ytick.labelsize":  8,    "legend.facecolor":PANEL, "legend.edgecolor":  BORDER,
    "legend.framealpha":0.88,
})

OUT = "charts"
os.makedirs(OUT, exist_ok=True)


def _style(ax, title, ylabel="", xlabel=""):
    ax.set_title(title, fontsize=10, fontweight="bold", pad=10, color=TEXT, loc="left")
    ax.set_ylabel(ylabel, fontsize=8.5, color=DIM)
    ax.set_xlabel(xlabel, fontsize=8.5, color=DIM)
    ax.yaxis.grid(True, alpha=0.35, linestyle=":")
    ax.xaxis.grid(False)
    ax.tick_params(length=3, color=BORDER)


def _save(name):
    plt.savefig(f"{OUT}/{name}", dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved -> {OUT}/{name}")


def _shade_regimes(ax, regimes):
    group_id = (regimes != regimes.shift()).cumsum()
    for _, grp in regimes.groupby(group_id):
        label = grp.iloc[0]
        if label == "Low Vol":
            ax.axvspan(grp.index[0], grp.index[-1], alpha=0.10, color=GREEN, zorder=1, linewidth=0)
        elif label == "High Vol":
            ax.axvspan(grp.index[0], grp.index[-1], alpha=0.10, color=RED,   zorder=1, linewidth=0)


# ══════════════════════════════════════════════════════════════════════════════
# Data pipeline (runs once, uses parquet cache)
# ══════════════════════════════════════════════════════════════════════════════
print("\nLoading data and computing signals...")
prices      = fetch_prices()
signal      = build_signal(prices)
factor_sigs = get_factor_signals(prices)
returns     = run_backtest(prices, signal)
equity      = (1 + returns).cumprod() * 100
dd          = drawdown_series(returns)
regimes     = classify_regimes(returns)
reg_stats   = regime_stats(returns, regimes)
is_ret, oos_ret = oos_split(returns, OOS_START)
boot        = bootstrap_sharpe(returns)
ic_df       = ic_decay(signal, prices)
fcorr       = factor_correlation(factor_sigs)
roll_ic_df  = rolling_factor_ic(factor_sigs, prices, window=63)
print("  Done. Generating charts...\n")

# ══════════════════════════════════════════════════════════════════════════════
# Chart 01 — Equity Curve
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(13, 5))
fig.patch.set_facecolor(BG)
_shade_regimes(ax, regimes.reindex(equity.index, method="nearest"))
ax.plot(equity.index, equity.values, color=GREEN, linewidth=2.0, zorder=4)
ax.fill_between(equity.index, 100, equity.values,
                where=equity.values >= 100, alpha=0.18, color=GREEN, zorder=2)
ax.fill_between(equity.index, 100, equity.values,
                where=equity.values < 100,  alpha=0.24, color=RED,   zorder=2)
ax.axhline(100, color=DIM, linewidth=0.7, linestyle="--", alpha=0.55)
peak_date = equity.idxmax()
ax.annotate(f"  Peak: ${equity.max():.1f}", xy=(peak_date, equity.max()),
            fontsize=8.5, color=GOLD, xytext=(0, 8), textcoords="offset points", ha="center")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor=GREEN, alpha=0.35, label="Low Vol regime"),
                   Patch(facecolor=RED,   alpha=0.35, label="High Vol regime")],
          loc="upper right", fontsize=8)
_style(ax, "EQUITY CURVE  --  Starting NAV = $100  |  Green = Low Vol  |  Red = High Vol",
       ylabel="Portfolio NAV ($)")
plt.tight_layout()
_save("01_equity_curve.png")

# ══════════════════════════════════════════════════════════════════════════════
# Chart 02 — Underwater / Drawdown
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(13, 4))
fig.patch.set_facecolor(BG)
ax.fill_between(dd.index, dd.values, 0, color=RED, alpha=0.30, zorder=2)
ax.plot(dd.index, dd.values, color=RED, linewidth=1.2, zorder=3)
ax.axhline(0, color=BORDER, linewidth=0.6)
peak_dt = dd.idxmin()
ax.annotate(f"Max DD: {dd.min():.1f}%", xy=(peak_dt, dd.min()),
            fontsize=8.5, color=RED, xytext=(0, -14), textcoords="offset points", ha="center")
_style(ax, "UNDERWATER CHART  --  Drawdown from Peak", ylabel="Drawdown (%)")
plt.tight_layout()
_save("02_drawdown.png")

# ══════════════════════════════════════════════════════════════════════════════
# Chart 03 — Rolling Sharpe
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(13, 4))
fig.patch.set_facecolor(BG)
rs63  = (returns.rolling(63).mean()  / returns.rolling(63).std()  * np.sqrt(252)).dropna()
rs252 = (returns.rolling(252).mean() / returns.rolling(252).std() * np.sqrt(252)).dropna()
ax.plot(rs63.index,  rs63.values,  color=BLUE, linewidth=1.1, label="63-day (quarterly)", alpha=0.85)
ax.plot(rs252.index, rs252.values, color=GOLD, linewidth=1.6, label="252-day (annual)",   alpha=0.90)
ax.fill_between(rs63.index, 0, rs63.values, where=rs63.values > 0, alpha=0.08, color=BLUE)
ax.axhline(0,   color=DIM,   linewidth=0.7, linestyle="--", alpha=0.55)
ax.axhline(1.0, color=GREEN, linewidth=0.6, linestyle=":",  alpha=0.40, label="Sharpe = 1.0")
ax.legend(loc="upper left")
_style(ax, "ROLLING SHARPE RATIO  --  Strategy Risk-Adjusted Return Over Time", ylabel="Sharpe Ratio")
plt.tight_layout()
_save("03_rolling_sharpe.png")

# ══════════════════════════════════════════════════════════════════════════════
# Chart 04 — Monthly Heatmap
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 5))
fig.patch.set_facecolor(BG)
mo     = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1) * 100
df_mo  = pd.DataFrame({"ret": mo, "year": mo.index.year, "month": mo.index.month})
pivot  = df_mo.pivot(index="year", columns="month", values="ret")
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
cmap   = LinearSegmentedColormap.from_list(
    "rg", ["#7B0D1E", "#1B0A2E", PANEL, "#0A3D24", "#064E3B"], N=512)
disp   = np.where(np.isfinite(vals), vals, 0.0)
ax.imshow(disp, cmap=cmap, aspect="auto", vmin=-vmax, vmax=vmax, zorder=1)
for i in range(vals.shape[0]+1): ax.axhline(i-0.5, color=BG, linewidth=1.2, zorder=3)
for j in range(vals.shape[1]+1): ax.axvline(j-0.5, color=BG, linewidth=1.2, zorder=3)
ax.axvline(11.5, color=GOLD, linewidth=2.0, alpha=0.65, zorder=4)
for i in range(vals.shape[0]):
    for j in range(vals.shape[1]):
        v = vals[i, j]
        if np.isfinite(v):
            ax.text(j, i, f"{v:+.1f}", ha="center", va="center",
                    fontsize=7.5, color=TEXT if abs(v)/vmax > 0.3 else DIM,
                    fontweight="bold" if j == vals.shape[1]-1 else "normal", zorder=5)
ax.set_xticks(range(n_cols)); ax.set_xticklabels(col_labels, fontsize=8.5, color=TEXT)
ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index, fontsize=8.5, color=TEXT)
ax.tick_params(length=0)
_style(ax, "MONTHLY RETURNS HEATMAP  --  Green = Gain  |  Red = Loss  |  Gold line separates Full Year")
plt.tight_layout()
_save("04_monthly_heatmap.png")

# ══════════════════════════════════════════════════════════════════════════════
# Chart 05 — Return Distribution
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor(BG)
ret_pct = returns.dropna().values * 100
mu, sigma = ret_pct.mean(), ret_pct.std()
ax.hist(ret_pct, bins=75, color=BLUE, alpha=0.60, edgecolor=PANEL, linewidth=0.3, density=True)
xs = np.linspace(ret_pct.min(), ret_pct.max(), 400)
ax.plot(xs, scipy_stats.norm.pdf(xs, mu, sigma), color=GOLD, linewidth=1.8,
        linestyle="--", label="Normal distribution fit")
var95 = np.percentile(ret_pct, 5)
ax.axvline(var95, color=RED, linewidth=1.5, linestyle="--",
           label=f"95% VaR = {var95:.2f}%")
ax.axvline(0, color=DIM, linewidth=0.8, linestyle=":", alpha=0.7)
skew = scipy_stats.skew(ret_pct); kurt = scipy_stats.kurtosis(ret_pct)
ax.text(0.97, 0.96,
        f"Mean  {mu:.3f}%\nStd   {sigma:.3f}%\nSkew  {skew:.3f}\nKurt  {kurt:.3f}",
        transform=ax.transAxes, fontsize=8.5, family="monospace", va="top", ha="right",
        color=TEXT, bbox=dict(boxstyle="round,pad=0.5", facecolor=PANEL,
                              edgecolor=BORDER, alpha=0.92))
ax.legend(loc="upper left")
_style(ax, "DAILY RETURN DISTRIBUTION", ylabel="Density", xlabel="Daily Return (%)")
plt.tight_layout()
_save("05_return_distribution.png")

# ══════════════════════════════════════════════════════════════════════════════
# Chart 06 — Rolling Composite IC
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(13, 4))
fig.patch.set_facecolor(BG)
ic = rolling_ic(signal, prices, window=21, horizon=1)
if not ic.empty:
    ax.plot(ic.index, ic.values, color=PURPLE, linewidth=1.2, alpha=0.9)
    ax.fill_between(ic.index, 0, ic.values, where=ic.values >= 0, alpha=0.18, color=GREEN)
    ax.fill_between(ic.index, 0, ic.values, where=ic.values < 0,  alpha=0.18, color=RED)
    ax.axhline(0,    color=DIM,  linewidth=0.7, linestyle="--", alpha=0.6)
    ax.axhline(0.05, color=TEAL, linewidth=0.6, linestyle=":",  alpha=0.45)
    mean_ic = float(ic.mean()); ic_std = float(ic.std())
    ic_ir = mean_ic / ic_std if ic_std > 0 else 0.0
    clr = GREEN if mean_ic > 0 else RED
    ax.text(0.015, 0.95, f"Mean IC = {mean_ic:.4f}   IC-IR = {ic_ir:.2f}",
            transform=ax.transAxes, fontsize=9, family="monospace", color=clr, va="top")
_style(ax, "ROLLING 21-DAY RANK IC  --  Composite Signal vs Next-Day Returns",
       ylabel="Spearman rho")
plt.tight_layout()
_save("06_rolling_ic.png")

# ══════════════════════════════════════════════════════════════════════════════
# Chart 07 — IC Decay
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor(BG)
if not ic_df.empty:
    horizons = ic_df.index.values
    means    = ic_df["mean_ic"].values
    se       = (ic_df["ic_std"] / np.sqrt(ic_df["n_obs"])).values
    colors   = [GREEN if m > 0 else RED for m in means]
    ax.bar(range(len(horizons)), means, color=colors, alpha=0.75,
           edgecolor=PANEL, linewidth=0.5)
    ax.errorbar(range(len(horizons)), means, yerr=se*1.96,
                fmt="none", color=TEXT, linewidth=1.2, capsize=4)
    ax.axhline(0, color=DIM, linewidth=0.7, linestyle="--", alpha=0.6)
    ax.set_xticks(range(len(horizons)))
    ax.set_xticklabels([f"{h}d" for h in horizons], fontsize=9)
    for i, (m, t) in enumerate(zip(means, ic_df["t_stat"].values)):
        if np.isfinite(m):
            ax.text(i, m + np.sign(m)*0.003, f"t={t:.1f}",
                    ha="center", va="bottom" if m >= 0 else "top",
                    fontsize=7, color=DIM)
_style(ax, "IC DECAY CURVE  --  How Far Into the Future Does the Signal Predict?",
       ylabel="Mean Rank-IC", xlabel="Forecast Horizon (trading days)")
plt.tight_layout()
_save("07_ic_decay.png")

# ══════════════════════════════════════════════════════════════════════════════
# Chart 08 — Factor Cross-Correlation
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 6))
fig.patch.set_facecolor(BG)
names = fcorr.columns.tolist()
vals2 = fcorr.values.astype(float)
cmap2 = LinearSegmentedColormap.from_list("rg", [RED, PANEL, BLUE], N=256)
ax.imshow(vals2, cmap=cmap2, aspect="auto", vmin=-1, vmax=1)
for i in range(len(names)+1): ax.axhline(i-0.5, color=BG, linewidth=1.0)
for j in range(len(names)+1): ax.axvline(j-0.5, color=BG, linewidth=1.0)
for i in range(len(names)):
    for j in range(len(names)):
        ax.text(j, i, f"{vals2[i,j]:.2f}", ha="center", va="center",
                fontsize=11, color=TEXT, fontweight="bold" if i==j else "normal")
ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=9, rotation=15, ha="right")
ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=9)
ax.tick_params(length=0)
_style(ax, "FACTOR CROSS-CORRELATION  --  Are the 3 Signals Independent?")
plt.tight_layout()
_save("08_factor_correlation.png")

# ══════════════════════════════════════════════════════════════════════════════
# Chart 09 — Rolling Per-Factor IC
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(13, 4.5))
fig.patch.set_facecolor(BG)
palette = [BLUE, GOLD, PURPLE]
for col, clr in zip(roll_ic_df.columns, palette):
    ax.plot(roll_ic_df.index, roll_ic_df[col].values,
            color=clr, linewidth=1.2, label=col, alpha=0.88)
ax.axhline(0,    color=DIM,  linewidth=0.7, linestyle="--", alpha=0.6)
ax.axhline(0.05, color=TEAL, linewidth=0.5, linestyle=":",  alpha=0.40)
ax.legend(loc="upper right")
_style(ax, "ROLLING 63-DAY RANK IC  --  Per-Factor Predictive Power Over Time",
       ylabel="Spearman rho")
plt.tight_layout()
_save("09_rolling_factor_ic.png")

# ══════════════════════════════════════════════════════════════════════════════
# Chart 10 — Vol Regime Analysis
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 4.5))
fig.patch.set_facecolor(BG)
reg_list = reg_stats.index.tolist()
sharpes  = reg_stats["Sharpe"].values
colors10 = [GREEN if s > 0 else RED for s in sharpes]
bars = ax.barh(reg_list, sharpes, color=colors10, alpha=0.78, edgecolor=PANEL, height=0.5)
ax.axvline(0, color=DIM, linewidth=0.7, linestyle="--", alpha=0.6)
for bar, v, idx in zip(bars, sharpes, reg_stats.index):
    days = int(reg_stats.loc[idx, "Days"])
    ann_r = float(reg_stats.loc[idx, "Ann. Return"])
    sign  = 1 if v >= 0 else -1
    ax.text(v + sign*0.02, bar.get_y() + bar.get_height()/2,
            f"SR={v:.2f}  Ret={ann_r:+.1%}  n={days}d",
            va="center", ha="left" if v >= 0 else "right", fontsize=8.5, color=TEXT)
_style(ax, "SHARPE RATIO BY VOLATILITY REGIME  --  Does Market Calm Help?",
       xlabel="Annualized Sharpe Ratio")
plt.tight_layout()
_save("10_regime_analysis.png")

# ══════════════════════════════════════════════════════════════════════════════
# Chart 11 — Bootstrap Sharpe
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor(BG)
ax.hist(boot["samples"], bins=80, color=BLUE, alpha=0.62,
        edgecolor=PANEL, linewidth=0.3, density=True)
ax.axvline(boot["observed"], color=GREEN, linewidth=2.2, label=f"Observed SR = {boot['observed']:.2f}")
ax.axvline(boot["ci_lo"], color=GOLD, linewidth=1.4, linestyle="--",
           label=f"95% CI  [{boot['ci_lo']:.2f}, {boot['ci_hi']:.2f}]")
ax.axvline(boot["ci_hi"], color=GOLD, linewidth=1.4, linestyle="--")
ax.axvline(0, color=DIM, linewidth=0.8, linestyle=":", alpha=0.6)
clr = GREEN if boot["p_value"] < 0.05 else RED
ax.text(0.97, 0.95, f"p-value = {boot['p_value']:.3f}",
        transform=ax.transAxes, fontsize=9, family="monospace", va="top", ha="right", color=clr)
ax.legend(loc="upper left", fontsize=8.5)
_style(ax, "BOOTSTRAP SHARPE RATIO  (n=10,000 resamples)  --  Statistical Significance Test",
       ylabel="Density", xlabel="Annualized Sharpe Ratio")
plt.tight_layout()
_save("11_bootstrap_sharpe.png")

# ══════════════════════════════════════════════════════════════════════════════
# Chart 12 — IS vs OOS
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(13, 5))
fig.patch.set_facecolor(BG)
is_eq  = (1 + is_ret).cumprod() * 100
oos_eq = (1 + oos_ret).cumprod() * 100
ax.plot(is_eq.index,  is_eq.values,  color=BLUE, linewidth=2.0, label="In-Sample (2015-2018)")
ax.plot(oos_eq.index, oos_eq.values, color=GOLD, linewidth=2.0, label="Out-of-Sample (2019-2023)")
ax.fill_between(is_eq.index,  100, is_eq.values,  where=is_eq.values  >= 100, alpha=0.12, color=BLUE)
ax.fill_between(oos_eq.index, 100, oos_eq.values, where=oos_eq.values >= 100, alpha=0.12, color=GOLD)
ax.fill_between(oos_eq.index, 100, oos_eq.values, where=oos_eq.values  < 100, alpha=0.12, color=RED)
ax.axhline(100, color=DIM, linewidth=0.6, linestyle=":", alpha=0.5)
split = pd.Timestamp(OOS_START)
ax.axvline(split, color=DIM, linewidth=1.2, linestyle="--", alpha=0.7)
is_sr  = sharpe_ratio(is_ret);  is_r  = annualized_return(is_ret)
oos_sr = sharpe_ratio(oos_ret); oos_r = annualized_return(oos_ret)
ax.text(0.02, 0.97,
        f"In-Sample     Ann.Ret={is_r:+.2%}  Sharpe={is_sr:.2f}",
        transform=ax.transAxes, fontsize=9, family="monospace", va="top", color=BLUE)
ax.text(0.02, 0.89,
        f"Out-of-Sample Ann.Ret={oos_r:+.2%}  Sharpe={oos_sr:.2f}",
        transform=ax.transAxes, fontsize=9, family="monospace", va="top", color=GOLD)
ax.legend(loc="upper right")
_style(ax, "IN-SAMPLE vs OUT-OF-SAMPLE  --  Both Start at $100 (No Lookahead Bias)",
       ylabel="NAV ($)")
plt.tight_layout()
_save("12_oos_comparison.png")

print(f"\nAll 12 charts saved to the '{OUT}/' folder.")
