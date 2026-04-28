import numpy as np
import pandas as pd
from typing import Callable, Optional


# ── Signal construction ────────────────────────────────────────────────────────

def zscore_signal(idiosyncratic: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Rolling z-score of idiosyncratic returns, NEGATED for mean reversion.
    Recent outperformer (high z) → expect reversion down → short signal.
    """
    rolling_mean = idiosyncratic.rolling(lookback, min_periods=lookback // 2).mean()
    rolling_std  = idiosyncratic.rolling(lookback, min_periods=lookback // 2).std()
    z = (idiosyncratic - rolling_mean) / rolling_std.replace(0, np.nan)
    return -z


def rank_signal(signal: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional rank normalisation to [-1, +1]."""
    ranks = signal.rank(axis=1, pct=True)
    return 2 * ranks - 1


def decay_signal(signal: pd.DataFrame, halflife: int = 5) -> pd.DataFrame:
    """EWM smoothing to reduce turnover."""
    alpha = 1 - np.exp(-np.log(2) / halflife)
    return signal.ewm(alpha=alpha, min_periods=1).mean()


def build_signal(
    idiosyncratic: pd.DataFrame,
    lookback: int = 10,
    decay_halflife: int = 5,
) -> pd.DataFrame:
    """
    Mean-reversion signal pipeline (single decay pass only).
      1. Rolling z-score (negated)
      2. Cross-sectional rank
      3. One EWM smooth
      4. Re-rank to restore uniform distribution
    """
    z        = zscore_signal(idiosyncratic, lookback=lookback)
    ranked   = rank_signal(z)
    smoothed = decay_signal(ranked, halflife=decay_halflife)
    return rank_signal(smoothed.fillna(0))   # fillna preserves cross-section width


def momentum_signal(
    returns: pd.DataFrame,
    fast_window: int = 21,
    slow_window: int = 252,
    skip_days: int = 5,
    decay_halflife: int = 10,
) -> pd.DataFrame:
    """
    12-1 month cross-sectional momentum (industry standard).

    Cumulative return over [slow_window → skip_days] ago.
    skip_days avoids the 1-month short-term reversal effect that would
    contaminate the momentum signal with mean-reversion noise.

    Positive return over the window → uptrend → long signal.
    """
    cum_slow = returns.rolling(slow_window, min_periods=slow_window // 2).sum().shift(skip_days)
    cum_fast = returns.rolling(fast_window, min_periods=fast_window // 2).sum().shift(skip_days)
    raw      = cum_slow - cum_fast          # intermediate-term minus short-term
    ranked   = rank_signal(raw)
    smoothed = decay_signal(ranked, halflife=decay_halflife)
    return rank_signal(smoothed.fillna(0))


def combine_signals(
    mr: pd.DataFrame,
    mom: pd.DataFrame,
    mr_weight: float = 0.5,
) -> pd.DataFrame:
    """
    Blend mean-reversion and momentum with explicit weights.
    Re-rank the composite to restore uniform cross-sectional distribution.
    """
    mom_weight = 1.0 - mr_weight
    composite  = mr_weight * mr.fillna(0) + mom_weight * mom.fillna(0)
    return rank_signal(composite)


# ── Portfolio construction ─────────────────────────────────────────────────────

def long_short_portfolio(
    signal: pd.DataFrame,
    n_stocks: int = 50,
    dollar_neutral: bool = True,
    use_decile: bool = True,
    returns: Optional[pd.DataFrame] = None,   # if provided: inverse-vol sizing
    vol_window: int = 21,
) -> pd.DataFrame:
    """
    Long-short portfolio. Top decile = long, bottom decile = short.
    Optionally scales each position by 1/realised_vol (equalises risk
    contribution — prevents high-vol stocks dominating PnL variance).
    """
    weights = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)

    prev_weights = pd.Series(0.0, index=signal.columns)
    lambda_ = 0.3  # 0.2–0.5 is a good range

    for date in signal.index:
        row = signal.loc[date].dropna()

        if len(row) < 20:
            weights.loc[date] = prev_weights
            continue

        cutoff = max(1, len(row) // 10) if use_decile else n_stocks
        if len(row) < 2 * cutoff:
            cutoff = max(1, len(row) // 2)

        long_stocks  = row.nlargest(cutoff).index
        short_stocks = row.nsmallest(cutoff).index

        # --- target portfolio (what you WANT) ---
        target = pd.Series(0.0, index=signal.columns)
        target[long_stocks]  =  1.0 / cutoff
        target[short_stocks] = -1.0 / cutoff

        # --- NEW: partial rebalance ---
        new_weights = lambda_ * target + (1 - lambda_) * prev_weights

        weights.loc[date] = new_weights
        prev_weights = new_weights

    # Inverse-vol scaling: equalise risk contribution across positions
    if returns is not None:
        roll_vol   = returns.rolling(vol_window, min_periods=10).std()
        median_vol = roll_vol.median(axis=1)
        scale      = median_vol / roll_vol.replace(0, np.nan)
        scale      = scale.clip(upper=3.0).fillna(1.0)
        weights    = weights * scale

    if dollar_neutral:
        row_abs_sum = weights.abs().sum(axis=1).replace(0, np.nan)
        weights     = weights.div(row_abs_sum, axis=0).fillna(0)

    return weights


# ── Backtester ─────────────────────────────────────────────────────────────────

class Backtester:
    """
    1-day execution lag, proportional transaction costs on turnover.
    """

    def __init__(self, returns: pd.DataFrame, transaction_cost_bps: float = 7.0):
        self.returns = returns
        self.tc      = transaction_cost_bps / 10_000

    def run(self, weights: pd.DataFrame) -> pd.DataFrame:
        w = weights.shift(1).fillna(0)   # 1-day lag: signal today → trade tomorrow

        common    = w.columns.intersection(self.returns.columns)
        w         = w[common]
        r         = self.returns[common]

        gross_pnl = (w * r).sum(axis=1)
        turnover  = w.diff().abs().sum(axis=1) / 2
        costs     = turnover * self.tc
        net_pnl   = gross_pnl - costs

        return pd.DataFrame({
            "gross_pnl": gross_pnl,
            "costs":     costs,
            "net_pnl":   net_pnl,
            "turnover":  turnover,
            "cum_gross": (1 + gross_pnl).cumprod(),
            "cum_net":   (1 + net_pnl).cumprod(),
        })

    def walk_forward(
        self,
        signal_fn: Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame],
        train_window: int = 504,
        test_window:  int = 63,
        n_factors:    str = "rmt",
        verbose:      bool = True,
    ) -> pd.DataFrame:
        """
        Walk-forward OOS validation.

        signal_fn(idio_df, full_returns_up_to_test_end) -> signal DataFrame.
        Passing full_returns lets the momentum signal look back further than
        the idiosyncratic window (avoids NaN-filled momentum in early windows).
        """
        from src.covariance import ledoit_wolf_cov
        from src.factors    import EigenFactorModel

        all_results = []
        T           = len(self.returns)
        n_windows   = (T - train_window) // test_window
        print(f"Walk-forward: {n_windows} windows ({train_window}d train / {test_window}d test)")

        start = train_window
        win   = 0
        while start + test_window <= T:
            train = self.returns.iloc[start - train_window: start]
            test  = self.returns.iloc[start: start + test_window]

            valid_cols  = train.columns[train.notna().all()]
            train_clean = train[valid_cols]
            test_clean  = test[valid_cols]

            if len(valid_cols) < 20:
                start += test_window; win += 1; continue

            cov, _ = ledoit_wolf_cov(train_clean.values)
            model  = EigenFactorModel(n_factors=n_factors)
            model.fit(train_clean.values, cov)

            _, idio = model.transform(test_clean.values)
            idio_df = pd.DataFrame(idio, index=test_clean.index, columns=valid_cols)

            # Full return history up to end of test window (for momentum lookback)
            full_ret = self.returns.loc[:test.index[-1], valid_cols]

            signal  = signal_fn(idio_df, full_ret)
            weights = long_short_portfolio(
                signal, use_decile=True, returns=test_clean
            )

            window_bt      = Backtester(test_clean, transaction_cost_bps=self.tc * 10_000)
            window_results = window_bt.run(weights)
            all_results.append(window_results.loc[test.index])

            win += 1
            if verbose:
                end_date = test.index[-1].date()
                print(f"  Window {win:>3}/{n_windows}  ending {end_date}", end="\r")
            start += test_window

        print()
        return pd.concat(all_results)
