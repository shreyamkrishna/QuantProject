import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple


def get_sp500_tickers():
    try:
        return pd.read_csv("sp500.csv")["Symbol"].tolist()
    except FileNotFoundError:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {"User-Agent": "Mozilla/5.0"}
        df = pd.read_html(url, storage_options=headers)[0]
        df.to_csv("sp500.csv", index=False)
        return df["Symbol"].tolist()


def load_prices(
    tickers: List[str],
    start: str = "2020-01-01",
    end: str = "2026-04-22",
    cache_path: str = "data/prices.parquet",
) -> pd.DataFrame:
    """Download adjusted close prices, with local parquet cache."""
    path = Path(cache_path)
    if path.exists():
        print("Loading prices from cache...")
        return pd.read_parquet(path)

    print(f"Downloading {len(tickers)} tickers from {start} to {end}...")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=True)
    prices = raw["Close"]
    path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(path)
    print(f"Saved to {cache_path}")
    return prices


def clean_prices(
    prices: pd.DataFrame,
    min_history_frac: float = 0.8,
    min_price: float = 1.0,
) -> pd.DataFrame:
    """
    Drop stocks with insufficient history or penny stock status.
    Forward-fill remaining NaNs (trading halts, etc).
    """
    T = len(prices)

    # Drop stocks missing more than (1 - min_history_frac) of dates
    prices = prices.dropna(thresh=int(min_history_frac * T), axis=1)

    # Drop penny stocks
    prices = prices.loc[:, prices.mean() >= min_price]

    # Forward-fill then drop any remaining leading NaNs
    prices = prices.ffill().dropna()

    print(f"Clean universe: {prices.shape[1]} stocks over {prices.shape[0]} trading days")
    return prices


def compute_returns(prices: pd.DataFrame, log: bool = True) -> pd.DataFrame:
    """
    Compute daily returns.
    Log returns preferred: additive across time, better statistical properties.
    """
    if log:
        returns = np.log(prices / prices.shift(1)).dropna()
    else:
        returns = prices.pct_change().dropna()
    return returns


def winsorize_returns(returns: pd.DataFrame, limit: float = 0.001) -> pd.DataFrame:
    """
    Clip extreme returns at 0.1th / 99.9th percentile per stock.
    Prevents earnings gaps and data errors from dominating covariance estimates.
    """
    lower = returns.quantile(limit)
    upper = returns.quantile(1 - limit)
    return returns.clip(lower=lower, upper=upper, axis=1)


def train_test_split(
    returns: pd.DataFrame,
    split_date: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Simple date-based train/test split."""
    train = returns.loc[:split_date]
    test = returns.loc[split_date:]
    print(f"Train: {len(train)} days | Test: {len(test)} days")
    return train, test
