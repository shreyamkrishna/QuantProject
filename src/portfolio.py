"""
src/portfolio.py — Signal construction, portfolio weights and backtesting.

CHANGES FROM ORIGINAL:
    - Backtester.walk_forward() now accepts a `spy_returns` parameter.
      When provided, each walk-forward window:
        1. Estimates rolling beta on the training split (using src.beta.rolling_beta)
        2. Freezes the final training-window beta
        3. Applies the frozen beta to neutralise test-period returns
        4. Fits PCA on market-neutral (not raw) training returns
      This removes the implicit market factor from the PCA and from the signal.

    - All other functions (zscore_signal, rank_signal, decay_signal, build_signal,
      momentum_signal, combine_signals, long_short_portfolio, Backtester.run)
      are UNCHANGED from the original.
"""

import numpy as np
import pandas as pd
from typing import Callable, Optional


# ── Signal construction ────────────────────────────────────────────────────────

def zscore_signal(idiosyncratic: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Rolling z-score of idiosyncratic returns, NEGATED for mean reversion.

    Logic:
        z > 0  → stock's idio return has been above its own recent average
               → expect reversion downward → short signal (high z → sell)
        z < 0  → below recent average → expect upward reversion → long signal

    The negation (-z) converts "recent outperformer" into "sell signal".

    UNCHANGED from original.
    """
    rolling_mean = idiosyncratic.rolling(lookback, min_periods=lookback // 2).mean()
    rolling_std  = idiosyncratic.rolling(lookback, min_periods=lookback // 2).std()
    z = (idiosyncratic - rolling_mean) / rolling_std.replace(0, np.nan)
    return -z


def rank_signal(signal: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-sectional rank normalisation to [-1, +1].

    Each day, stocks are ranked 0 to 1 (percentile), then shifted to [-1, +1].
    This ensures the signal distribution is uniform regardless of the
    underlying signal's scale.  Prevents any single day's extreme z-scores
    from dominating portfolio construction.

    UNCHANGED from original.
    """
    ranks = signal.rank(axis=1, pct=True)
    return 2 * ranks - 1


def decay_signal(signal: pd.DataFrame, halflife: int = 5) -> pd.DataFrame:
    """
    EWM smoothing to reduce daily turnover.

    Without smoothing, the signal flips abruptly from day to day, causing
    high portfolio turnover (and therefore high transaction costs).
    EWM smoothing blends today's signal with recent history.

    halflife=5 means yesterday's signal contributes ~50% after 5 days.

    UNCHANGED from original.
    """
    alpha = 1 - np.exp(-np.log(2) / halflife)
    return signal.ewm(alpha=alpha, min_periods=1).mean()


def build_signal(
    idiosyncratic: pd.DataFrame,
    lookback: int = 10,
    decay_halflife: int = 5,
) -> pd.DataFrame:
    """
    Full mean-reversion signal pipeline.

    Steps:
      1. Rolling z-score (negated) — raw mean-reversion signal
      2. Cross-sectional rank → uniform [-1, +1] distribution
      3. EWM smooth → reduces turnover
      4. Re-rank → restores uniform distribution after smoothing

    The double ranking (steps 2 and 4) ensures the final signal always has
    a clean, well-behaved cross-sectional distribution.

    UNCHANGED from original.
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

    The signal = cumulative return over the past ~12 months,
    EXCLUDING the most recent skip_days.

    Why skip the most recent days?
        Stocks with strong very-recent returns often mean-revert in the short
        term (the "short-term reversal" effect).  By skipping the last 5 days,
        we avoid contaminating the momentum signal with this mean-reversion noise.

    Formula:
        raw = sum(returns, lag=5 to lag=252)
            = rolling_sum(252).shift(5) - rolling_sum(21).shift(5)

    Then rank and smooth for cleanliness.

    UNCHANGED from original.
    """
    cum_slow = returns.rolling(slow_window, min_periods=slow_window // 2).sum().shift(skip_days)
    cum_fast = returns.rolling(fast_window, min_periods=fast_window // 2).sum().shift(skip_days)
    raw      = cum_slow - cum_fast
    ranked   = rank_signal(raw)
    smoothed = decay_signal(ranked, halflife=decay_halflife)
    return rank_signal(smoothed.fillna(0))


def combine_signals(
    mr: pd.DataFrame,
    mom: pd.DataFrame,
    mr_weight: float = 0.5,
) -> pd.DataFrame:
    """
    Blend mean-reversion and momentum with fixed explicit weights.

    Note: main.py uses DYNAMIC weighting (weights proportional to recent
    signal strength) instead of this fixed-weight function.  This function
    is kept for backward compatibility and experimentation.

    UNCHANGED from original.
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
    returns: Optional[pd.DataFrame] = None,
    vol_window: int = 21,
) -> pd.DataFrame:
    """
    Long-short portfolio: top decile long, bottom decile short.

    use_decile=True (default):
        Long the top 10% of stocks, short the bottom 10%.
        Cutoff adjusts automatically to universe size.

    use_decile=False:
        Fixed n_stocks on each side.

    Inverse-vol sizing (when returns is provided):
        Position weight = 1/vol, normalised so total notional = 1.
        This equalises risk contribution: a high-vol stock gets a smaller
        weight so it contributes the same dollar vol as a low-vol stock.
        Without this, a few volatile names would dominate portfolio variance.

    dollar_neutral=True:
        Long weights sum to +1, short weights sum to -1 (net zero).
        This removes directional market exposure from the portfolio.

    UNCHANGED from original.
    """
    weights = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)

    for date in signal.index:
        row = signal.loc[date].dropna()
        if len(row) < 20:
            continue

        cutoff = max(1, len(row) // 10) if use_decile else n_stocks
        if len(row) < 2 * cutoff:
            cutoff = max(1, len(row) // 2)

        long_stocks  = row.nlargest(cutoff).index
        short_stocks = row.nsmallest(cutoff).index

        weights.loc[date, long_stocks]  =  1.0 / cutoff
        weights.loc[date, short_stocks] = -1.0 / cutoff

    # Inverse-vol scaling
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
    Simulates trading with:
      - 1-day execution lag (signal today → trade at tomorrow's open)
      - Proportional transaction costs on turnover (two-way, in bps)

    CHANGES FROM ORIGINAL:
        walk_forward() now accepts `spy_returns` for beta neutralisation.
        run() is unchanged.
    """

    def __init__(self, returns: pd.DataFrame, transaction_cost_bps: float = 7.0):
        self.returns = returns
        self.tc      = transaction_cost_bps / 10_000   # convert bps to decimal

    def run(self, weights: pd.DataFrame) -> pd.DataFrame:
        """
        Simulate the portfolio day by day.

        The 1-day lag (weights.shift(1)) means:
            - Signal computed after close on day t
            - Trades executed at open on day t+1
            - PnL realised using day t+1 returns

        Turnover = half the sum of absolute weight changes (two-way measure).
        Transaction cost = turnover × cost_per_unit.

        Returns a DataFrame with columns:
            gross_pnl   — PnL before transaction costs
            costs       — Transaction cost drag
            net_pnl     — PnL after costs
            turnover    — Two-way daily turnover
            cum_gross   — Cumulative gross return
            cum_net     — Cumulative net return

        UNCHANGED from original.
        """
        # Lag weights by 1 day: signal formed at close → executed next open
        w = weights.shift(1).fillna(0)

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
        signal_fn:   Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame],
        train_window: int = 504,
        test_window:  int = 63,
        n_factors:    str = "rmt",
        verbose:      bool = True,
        spy_returns:  Optional[pd.Series] = None,   # [NEW] market benchmark for beta
        beta_window:  int = 63,                      # [NEW] rolling beta window
    ) -> pd.DataFrame:
        """
        Walk-forward OOS validation.

        For each window:
            train  = returns[start - train_window : start]
            test   = returns[start             : start + test_window]

        signal_fn(idio_df, full_returns_up_to_test_end) → signal DataFrame

        The `full_returns_up_to_test_end` argument lets the momentum signal
        look back further than the idio window.  This avoids the momentum
        signal being all-NaN in early windows (momentum needs 252 days of
        history, but test windows are only 123 days).

        [NEW] Beta neutralisation (when spy_returns is provided):
        ──────────────────────────────────────────────────────────
        Step A: Estimate rolling beta on the TRAINING split.
                → Uses the vectorised rolling_beta() from src.beta.
        Step B: Freeze the LAST training-day beta.
                → This is the beta estimate available at the start of the test period.
                → Critically, no test data is used to estimate beta (no lookahead).
        Step C: Apply frozen beta to NEUTRALISE test returns.
                → r_i_neutral = r_i − β_i_frozen × r_SPY
        Step D: Fit PCA on MARKET-NEUTRAL training returns.
                → The first factor now captures sector/style, not market direction.
        Step E: Transform test returns to idiosyncratic using training eigenvectors.
                → Same eigenvectors B that were fitted on neutral training data.
        Step F: Call signal_fn on the idiosyncratic test returns.

        CHANGED from original: added steps A–E above.
        signal_fn interface is unchanged (same arguments).
        """
        # Late imports to avoid circular dependencies between src modules
        from src.covariance import ledoit_wolf_cov
        from src.factors    import EigenFactorModel

        # [NEW] Only import beta utilities when spy_returns is actually provided
        beta_neutralise = spy_returns is not None
        if beta_neutralise:
            from src.beta import rolling_beta, freeze_and_apply_beta
            spy = spy_returns.squeeze()   # ensure Series, not single-col DataFrame

        all_results = []
        T           = len(self.returns)
        n_windows   = (T - train_window) // test_window
        print(f"Walk-forward: {n_windows} windows ({train_window}d train / {test_window}d test)")

        start = train_window
        win   = 0

        while start + test_window <= T:
            # ── Slice raw return windows ───────────────────────────────────────
            train = self.returns.iloc[start - train_window: start]
            test  = self.returns.iloc[start: start + test_window]

            # Drop columns with any NaN in training (LedoitWolf requires complete data)
            valid_cols  = train.columns[train.notna().all()]
            train_clean = train[valid_cols]
            test_clean  = test[valid_cols]

            if len(valid_cols) < 20:
                start += test_window
                win   += 1
                continue

            # ── [NEW] Beta neutralisation ──────────────────────────────────────
            if beta_neutralise:
                # A. Estimate rolling beta on training data
                spy_train_window = spy.reindex(train_clean.index).ffill().fillna(0)
                train_betas = rolling_beta(train_clean, spy_train_window,
                                           window=beta_window)

                # B. Freeze last training-day beta (to be applied to test period)
                last_beta = train_betas.iloc[-1]   # shape (N,)

                # C. Market-neutral training returns (used for covariance + PCA)
                #    We also drop the first beta_window rows (warmup period where
                #    beta estimates are unreliable due to insufficient history).
                mn_train = train_clean - train_betas.mul(spy_train_window, axis=0)
                mn_train = mn_train.iloc[beta_window:]   # drop warmup rows

                # D. Market-neutral test returns (using frozen beta — no lookahead)
                spy_test_window = spy.reindex(test_clean.index).ffill().fillna(0)
                mn_test = freeze_and_apply_beta(test_clean, spy_test_window, last_beta)

                # Use market-neutral returns for covariance estimation and PCA
                cov_input   = mn_train.values
                idio_input  = mn_test.values
                idio_index  = test_clean.index
                idio_cols   = valid_cols
            else:
                # [ORIGINAL PATH] No beta neutralisation — use raw returns
                cov_input  = train_clean.values
                idio_input = test_clean.values
                idio_index = test_clean.index
                idio_cols  = valid_cols

            # ── Covariance estimation (Ledoit-Wolf) ────────────────────────────
            # Fitted on market-neutral training returns if beta_neutralise=True,
            # or on raw training returns otherwise.
            cov, _ = ledoit_wolf_cov(cov_input)

            # ── Factor model ───────────────────────────────────────────────────
            # EigenFactorModel.fit() selects factors via RMT threshold.
            # After market neutralisation, the dominant market factor is gone,
            # so selected factors should reflect sector/style effects.
            model = EigenFactorModel(n_factors=n_factors)
            model.fit(cov_input, cov)

            # ── Extract idiosyncratic returns for test period ───────────────────
            # We use the eigenvectors B fitted on training data — applying them
            # to test data ensures no test data influenced the factor structure.
            _, idio = model.transform(idio_input)
            idio_df = pd.DataFrame(idio, index=idio_index, columns=idio_cols)

            # ── Signal construction ────────────────────────────────────────────
            # full_ret = all returns up to end of test window.
            # Momentum needs 252 days of lookback, which extends before the
            # start of this test window.  Passing full history resolves this.
            full_ret = self.returns.loc[:test.index[-1], valid_cols]
            signal   = signal_fn(idio_df, full_ret)

            # ── Portfolio construction and backtesting ─────────────────────────
            weights = long_short_portfolio(
                signal, use_decile=True, returns=test_clean
            )
            window_bt      = Backtester(test_clean, transaction_cost_bps=self.tc * 10_000)
            window_results = window_bt.run(weights)
            all_results.append(window_results.loc[test.index])

            win += 1
            if verbose:
                end_date = test.index[-1].date()
                # \r overwrites the same line — keeps output clean for many windows
                print(f"  Window {win:>3}/{n_windows}  ending {end_date}", end="\r")

            start += test_window

        print()   # newline after the \r progress line
        return pd.concat(all_results)