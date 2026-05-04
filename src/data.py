"""
src/data.py — Data loading, cleaning and splitting utilities.

CHANGES FROM ORIGINAL:
    - Added get_spy() function to download the SPY ETF as the market benchmark.
      This is used by the new beta-neutralisation step (src/beta.py) to estimate
      each stock's sensitivity to the broad market before fitting PCA.
    - Everything else is unchanged.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple


# ── Ticker universe ────────────────────────────────────────────────────────────

def get_sp500_tickers():
    """
    Returns the list of S&P 500 ticker symbols.

    Tries to read from a local 'sp500.csv' file first (fast, no internet needed).
    If the file doesn't exist, scrapes the current constituents from Wikipedia
    and saves them to 'sp500.csv' so future runs are instant.

    The .replace('.', '-') handles tickers like BRK.B → BRK-B, which is what
    Yahoo Finance expects.
    """
    try:
        return pd.read_csv("sp500.csv")["Symbol"].tolist()
    except FileNotFoundError:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {"User-Agent": "Mozilla/5.0"}
        df = pd.read_html(url, storage_options=headers)[0]
        df.to_csv("sp500.csv", index=False)
        return df["Symbol"].tolist()


def get_spy(
    start: str = "2020-01-01",
    end:   str = "2026-04-29",
    cache_path: str = "data/spy_prices.parquet",
) -> pd.DataFrame:
    """
    [NEW] Download adjusted close prices for SPY (S&P 500 ETF).

    SPY is used as the market benchmark for beta estimation.  We download
    it separately from the universe so we always have a clean, single-column
    price series regardless of whether SPY happens to be in the constituent
    list or not.

    The result is cached to a parquet file so repeated runs don't re-download.

    Returns
    -------
    pd.DataFrame
        Single-column DataFrame with index=date, column='SPY', containing
        daily adjusted closing prices.

    Why a DataFrame and not a Series?
        Keeping it as a DataFrame makes it consistent with the multi-stock
        `prices` DataFrame from load_prices().  Callers that need a Series
        can do `.squeeze()` — but src/beta.py handles this internally.
    """
    path = Path(cache_path)

    if path.exists():
        print("Loading SPY prices from cache...")
        return pd.read_parquet(path)

    print(f"Downloading SPY from {start} to {end}...")
    raw = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)

    # .squeeze() collapses the ['Close'] DataFrame to a Series in newer yfinance
    # versions, which return a multi-level column index.  We then wrap it back
    # as a single-column DataFrame for consistency.
    prices = raw["Close"].squeeze().rename("SPY").to_frame()

    path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(path)
    print(f"SPY prices saved to {cache_path}")
    return prices


# ── Price download & cache ─────────────────────────────────────────────────────

def load_prices(
    tickers: List[str],
    start: str = "2020-01-01",
    end: str = "2026-04-22",
    cache_path: str = "data/prices.parquet",
) -> pd.DataFrame:
    """
    Download adjusted close prices for the full universe, with local parquet cache.

    The parquet cache is critical: downloading 490 tickers from yfinance takes
    ~3 minutes.  Once cached, it loads in under 1 second.

    UNCHANGED from original.
    """
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


# ── Cleaning ───────────────────────────────────────────────────────────────────

def clean_prices(
    prices: pd.DataFrame,
    min_history_frac: float = 0.8,
    min_price: float = 1.0,
) -> pd.DataFrame:
    """
    Drop stocks with insufficient history or penny-stock status, then forward-fill.

    min_history_frac = 0.8 means a stock must have prices for at least 80% of
    the date range.  Stocks that listed mid-period or were delisted early are
    excluded to avoid survivorship-bias artefacts in the covariance estimate.

    min_price = 1.0 excludes penny stocks whose large percentage swings
    would dominate the cross-section.

    Forward-fill handles trading halts (a stock that didn't trade one day
    keeps its last known price).

    UNCHANGED from original.
    """
    T = len(prices)

    # Drop stocks missing more than (1 - min_history_frac) of dates
    prices = prices.dropna(thresh=int(min_history_frac * T), axis=1)

    # Drop penny stocks (mean price below $1)
    prices = prices.loc[:, prices.mean() >= min_price]

    # Forward-fill then drop any remaining leading NaNs
    prices = prices.ffill().dropna()

    print(f"Clean universe: {prices.shape[1]} stocks over {prices.shape[0]} trading days")
    return prices


# ── Returns ────────────────────────────────────────────────────────────────────

def compute_returns(prices: pd.DataFrame, log: bool = True) -> pd.DataFrame:
    """
    Compute daily returns from prices.

    log=True  → log returns: ln(P_t / P_{t-1})
        Additive across time: weekly return = sum of daily log returns.
        Better statistical properties (more normally distributed).
        Standard choice for factor models.

    log=False → simple percentage returns: (P_t - P_{t-1}) / P_{t-1}
        Used when the downstream calculation requires simple returns
        (e.g. portfolio PnL is computed using simple returns, not log).

    UNCHANGED from original.
    """
    if log:
        returns = np.log(prices / prices.shift(1)).dropna()
    else:
        returns = prices.pct_change().dropna()
    return returns


# ── Winsorisation ──────────────────────────────────────────────────────────────

def winsorize_returns(returns: pd.DataFrame, limit: float = 0.001) -> pd.DataFrame:
    """
    Clip extreme returns at the 0.1th / 99.9th percentile per stock.

    Why winsorise?  A single earnings gap (+40% in one day) has an outsized
    effect on:
      - Covariance estimates  → the correlation matrix becomes noisy
      - Signal z-scores       → one stock dominates the cross-section
      - Portfolio PnL         → one bet drives the whole backtest

    Winsorising at 0.1%/99.9% keeps 99.8% of observations intact while
    capping the most extreme 0.2%.  More conservative than the 1%/99%
    clipping used in the notebook.

    UNCHANGED from original.
    """
    lower = returns.quantile(limit)
    upper = returns.quantile(1 - limit)
    return returns.clip(lower=lower, upper=upper, axis=1)


# ── Train / test split ─────────────────────────────────────────────────────────

def train_test_split(
    returns: pd.DataFrame,
    split_date: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simple date-based train/test split.

    Returns everything up to and including split_date as training data,
    and everything after as test data.

    Note: this is a strict temporal split.  No shuffling, no randomness.
    Shuffling would introduce lookahead bias — the model would have seen
    future data during training.

    UNCHANGED from original.
    """
    train = returns.loc[:split_date]
    test  = returns.loc[split_date:]
    print(f"Train: {len(train)} days | Test: {len(test)} days")
    return train, test