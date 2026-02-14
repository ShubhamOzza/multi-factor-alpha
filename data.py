import os
import pandas as pd
import yfinance as yf
from config import UNIVERSE, START, END

CACHE_FILE = "prices.parquet"


def fetch_prices() -> pd.DataFrame:
    if os.path.exists(CACHE_FILE):
        return pd.read_parquet(CACHE_FILE)

    raw = yf.download(UNIVERSE, start=START, end=END, auto_adjust=True, progress=False)
    prices = raw["Close"].dropna(axis=1, how="all").ffill()
    prices.to_parquet(CACHE_FILE)
    return prices


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change()
