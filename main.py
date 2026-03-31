"""
main.py -- Multi-Factor Equity Alpha Model
           Pipeline: data -> factors -> backtest -> metrics -> analytics -> visualisation
"""
import pandas as pd

from config    import OOS_START
from data      import fetch_prices
from factors   import build_signal, get_factor_signals
from portfolio import run_backtest, equity_curve
from metrics   import summary
from regime    import classify_regimes, regime_stats
from analytics import ic_decay, bootstrap_sharpe, factor_correlation, oos_split
from attribution import factor_returns, rolling_factor_ic
from visualize import plot_full_report, plot_analytics_report, export_equity_json

_SEP = "  " + "-" * 62


def _header(n: int, total: int, label: str) -> None:
    print(f"\n  [{n}/{total}]  {label}")


def _print_metrics(stats: dict) -> None:
    groups = [
        ("Returns",      ["Annualized Return", "Total Return", "Annualized Volatility"]),
        ("Risk-Adj",     ["Sharpe Ratio", "Sortino Ratio", "Calmar Ratio"]),
        ("Drawdown",     ["Max Drawdown"]),
        ("Distribution", ["Win Rate", "Tail Ratio", "Return Skewness", "Excess Kurtosis"]),
        ("Tail Risk",    ["VaR 95% (daily)", "CVaR 95% (daily)"]),
    ]
    col_w = max(len(k) for k in stats)
    for group_name, keys in groups:
        print(f"\n       -- {group_name}")
        for k in keys:
            if k in stats:
                print(f"       {k:<{col_w}}  {stats[k]}")


def main() -> None:
    STEPS = 7
    print("\n" + _SEP)
    print("   MULTI-FACTOR EQUITY ALPHA MODEL")
    print("   Risk-Adjusted Momentum | Long/Short Equity | S&P 500 Universe")
    print(_SEP)

    # ── 1. Data ─────────────────────────────────────────────────────────────
    _header(1, STEPS, "Fetching price data...")
    prices = fetch_prices()
    print(f"         {len(prices.columns)} stocks  |  "
          f"{prices.index[0].date()} to {prices.index[-1].date()}")

    # ── 2. Signals ──────────────────────────────────────────────────────────
    _header(2, STEPS, "Computing factor signals...")
    signal      = build_signal(prices)
    factor_sigs = get_factor_signals(prices)
    coverage    = signal.notna().sum(axis=1).mean()
    print(f"         Composite: Risk-Adj Momentum | ST Reversal | Low Volatility")
    print(f"         Average cross-section coverage: {coverage:.0f} stocks/day")

    # ── 3. Backtest ─────────────────────────────────────────────────────────
    _header(3, STEPS, "Running backtest...")
    returns = run_backtest(prices, signal)
    print(f"         {len(returns)} trading days simulated  "
          f"({returns.index[0].date()} to {returns.index[-1].date()})")

    # ── 4. Performance metrics ───────────────────────────────────────────────
    _header(4, STEPS, "Performance summary")
    stats = summary(returns)
    _print_metrics(stats)

    # ── 5. Signal analytics ─────────────────────────────────────────────────
    _header(5, STEPS, "Running signal analytics...")

    print("         Computing IC decay curve...")
    ic_df = ic_decay(signal, prices)

    print("         Running bootstrap Sharpe (n=10,000)...")
    boot = bootstrap_sharpe(returns)
    print(f"         Observed Sharpe = {boot['observed']:.3f}  |  "
          f"95% CI = [{boot['ci_lo']:.3f}, {boot['ci_hi']:.3f}]  |  "
          f"p-value = {boot['p_value']:.3f}")

    print("         Computing factor correlations...")
    fcorr = factor_correlation(factor_sigs)

    print("         Computing per-factor rolling IC...")
    roll_ic_df = rolling_factor_ic(factor_sigs, prices, window=63)

    print("         Classifying volatility regimes...")
    regimes   = classify_regimes(returns, window=63)
    reg_stats = regime_stats(returns, regimes)
    print("\n       -- Vol-Regime Breakdown (Sharpe | Ann.Return | Days)")
    for regime, row in reg_stats.iterrows():
        print(f"       {regime:<12}  SR={row['Sharpe']:+.2f}  "
              f"Ret={row['Ann. Return']:+.2%}  n={int(row['Days'])}d")

    print("\n         Splitting IS / OOS at", OOS_START, "...")
    is_ret, oos_ret = oos_split(returns, OOS_START)
    is_sr  = (is_ret.mean() / is_ret.std()) * (252 ** 0.5)
    oos_sr = (oos_ret.mean() / oos_ret.std()) * (252 ** 0.5)
    print(f"         In-Sample     ({is_ret.index[0].year}-{is_ret.index[-1].year}):  "
          f"Sharpe={is_sr:.2f}  "
          f"Ann.Ret={is_ret.mean()*252:+.2%}")
    print(f"         Out-of-Sample ({oos_ret.index[0].year}-{oos_ret.index[-1].year}):  "
          f"Sharpe={oos_sr:.2f}  "
          f"Ann.Ret={oos_ret.mean()*252:+.2%}")

    # ── 6. Generate reports ──────────────────────────────────────────────────
    _header(6, STEPS, "Generating performance tear-sheet...")
    plot_full_report(returns, signal, prices, regimes=regimes)

    _header(7, STEPS, "Generating analytics report...")
    plot_analytics_report(
        ic_decay_df     = ic_df,
        factor_corr     = fcorr,
        bootstrap       = boot,
        reg_stats       = reg_stats,
        is_returns      = is_ret,
        oos_returns     = oos_ret,
        roll_factor_ic  = roll_ic_df,
        split_date      = pd.Timestamp(OOS_START),
    )
    export_equity_json(returns)

    eq = equity_curve(returns)
    eq.to_csv("equity_curve.csv", header=["equity"])
    print("         Saved -> equity_curve.csv")
    returns.to_csv("daily_returns.csv", header=["returns"])
    print("         Saved -> daily_returns.csv")

    print("\n" + _SEP)
    print("   Done.\n")


if __name__ == "__main__":
    main()
