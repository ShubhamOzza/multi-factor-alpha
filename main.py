import pandas as pd
from data       import fetch_prices, daily_returns
from factors    import build_signal
from portfolio  import run_backtest, equity_curve
from metrics    import summary
from visualize  import plot_full_report, export_equity_json


def main():
    print("\n Multi-Factor Equity Alpha Model")
    print(" ─" * 34)

    print("\n[1/5]  Fetching price data...")
    prices = fetch_prices()
    print(f"       {len(prices.columns)} stocks  |  {prices.index[0].date()} → {prices.index[-1].date()}")

    print("\n[2/5]  Computing factor signals...")
    signal = build_signal(prices)
    coverage = signal.notna().sum(axis=1).mean()
    print(f"       Average signal coverage: {coverage:.0f} stocks per day")

    print("\n[3/5]  Running backtest...")
    returns = run_backtest(prices, signal)
    print(f"       {len(returns)} trading days simulated")

    print("\n[4/5]  Performance summary:")
    stats = summary(returns)
    col_w = max(len(k) for k in stats)
    for k, v in stats.items():
        print(f"       {k:<{col_w}}  {v}")

    print("\n[5/5]  Generating outputs...")
    plot_full_report(returns, signal, prices)
    export_equity_json(returns)

    eq = equity_curve(returns)
    eq.to_csv("equity_curve.csv", header=["equity"])
    print("  Saved → equity_curve.csv")

    returns.to_csv("daily_returns.csv", header=["returns"])
    print("  Saved → daily_returns.csv")

    print("\n Done.\n")


if __name__ == "__main__":
    main()
