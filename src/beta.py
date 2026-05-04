"""
src/beta.py — SPY Beta Estimation & Market-Neutral Return Construction.

[NEW MODULE — did not exist in the original project]

WHY THIS MODULE EXISTS
----------------------
In the original pipeline, PCA was applied directly to raw stock returns.
This caused a well-known problem: the first PCA factor (λ₁, ~31% of variance)
is essentially just "the market went up or down today".  It's not a useful
factor — it's just SPY wearing a disguise.

The fix is to explicitly regress out the market *before* PCA:
    r_i_neutral(t) = r_i(t) − β_i(t-1) × r_SPY(t)

After this step:
  1. The remaining PCA factors capture genuine sector/style effects, not market noise.
  2. Signals built on truly idiosyncratic returns are less regime-dependent.
  3. The portfolio naturally stays market-neutral without needing a separate SPY hedge.

HOW BETA IS ESTIMATED
---------------------
We use a rolling OLS beta over a 63-day (≈3 month) window:

    β_i(t) = Cov(r_i, r_SPY) / Var(r_SPY)

Using the identity:  Cov(X, Y) = E[XY] − E[X]·E[Y]
we compute this entirely with rolling means — no per-column Python loops,
so it runs on 490 stocks in about the same time as one stock.

CRITICAL: The beta used on day t is estimated using data up to day t-1
(beta is lagged by 1 day).  Using today's beta would be lookahead bias —
in live trading you wouldn't know today's beta until after market close.

FUNCTIONS
---------
rolling_beta(stock_ret, market_ret, window)
    → DataFrame of rolling betas, lagged 1 day.

market_neutral_returns(stock_ret, market_ret, beta_df)
    → DataFrame of beta-adjusted returns.

plot_beta_diagnostics(beta_df, raw_ret, mn_ret, spy_ret)
    → 3-panel diagnostic figure showing beta distribution,
      vol reduction, and SPY correlation before/after.

plot_beta_neutrality(port_ret, spy_ret)
    → 2-panel figure showing rolling portfolio beta and beta scatter.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Tuple


# ── Rolling beta estimation ────────────────────────────────────────────────────

def rolling_beta(
    stock_ret:  pd.DataFrame,
    market_ret: pd.Series,
    window:     int = 63,
) -> pd.DataFrame:
    """
    Compute rolling OLS beta for every stock vs. the market.

    Beta_i = Cov(r_i, r_mkt) / Var(r_mkt)

    Uses the algebraic identity to avoid a Python loop over 490 columns:
        Cov(X, Y) = E[XY] − E[X]·E[Y]

    This makes it ~490× faster than computing cov() per column.

    Parameters
    ----------
    stock_ret   : (T, N) DataFrame of daily stock returns
    market_ret  : (T,)  Series of daily market returns (e.g. SPY)
    window      : rolling window length in trading days (default 63 ≈ 3 months)

    Returns
    -------
    pd.DataFrame, same shape as stock_ret
        Rolling betas, SHIFTED FORWARD BY 1 DAY to prevent lookahead.
        Rows with insufficient history are filled with beta=1.0 (neutral assumption).

    Why fill NaN with 1.0?
        During the warmup period (first `window` days) there isn't enough data
        to estimate beta.  A beta of 1.0 means "assume this stock moves exactly
        with the market" — conservative and neutral, not zero (which would mean
        "this stock is completely uncorrelated with the market").
    """
    # ── 1. Align market returns to stock index ─────────────────────────────────
    # .squeeze() handles the case where market_ret is accidentally a DataFrame
    # (e.g. if yfinance returns a single-column DataFrame instead of a Series).
    mkt = market_ret.squeeze().reindex(stock_ret.index).ffill().fillna(0)
    assert isinstance(mkt, pd.Series), (
        "market_ret must be a Series after squeeze().  "
        "Got type: " + str(type(mkt))
    )

    # ── 2. Rolling variance of market returns: Var(r_mkt) ─────────────────────
    # This is the denominator in the beta formula.
    # min_periods = window//2 allows estimation to start after half the window
    # is filled (gives smoother startup at the cost of slightly noisier early betas).
    mkt_var = mkt.rolling(window, min_periods=window // 2).var()          # shape (T,)

    # ── 3. Rolling means needed for the covariance identity ───────────────────
    roll_mean_mkt   = mkt.rolling(window, min_periods=window // 2).mean() # shape (T,)
    roll_mean_stock = stock_ret.rolling(window, min_periods=window // 2).mean()  # (T, N)

    # E[r_i × r_mkt] — the rolling mean of the product of each stock with SPY
    roll_mean_prod  = (
        stock_ret                                                    # (T, N)
        .multiply(mkt, axis=0)                                       # (T, N): r_i * r_mkt
        .rolling(window, min_periods=window // 2)
        .mean()
    )                                                                 # shape (T, N)

    # ── 4. Covariance matrix: E[XY] − E[X]·E[Y] ──────────────────────────────
    # .mul(roll_mean_mkt, axis=0) broadcasts the (T,) series across all N columns
    cov_mat = roll_mean_prod.sub(
        roll_mean_stock.mul(roll_mean_mkt, axis=0)
    )                                                                 # shape (T, N)

    # ── 5. Beta = Cov / Var ───────────────────────────────────────────────────
    # Replace 0 variance (constant market return) with NaN to avoid division by zero.
    # fillna(1.0) applies the neutral-beta assumption during warmup.
    betas = (
        cov_mat
        .div(mkt_var.replace(0, np.nan), axis=0)
        .fillna(1.0)
    )                                                                 # shape (T, N)

    # ── 6. Lag by 1 day — NO LOOKAHEAD ────────────────────────────────────────
    # Beta estimated using data through day t−1 is used to neutralise day t.
    # This mirrors how a real trader would operate: yesterday's model → today's trade.
    return betas.shift(1).fillna(1.0)


# ── Market-neutral return construction ────────────────────────────────────────

def market_neutral_returns(
    stock_ret:  pd.DataFrame,
    market_ret: pd.Series,
    beta_df:    pd.DataFrame,
) -> pd.DataFrame:
    """
    Subtract the market component from each stock's return.

    r_i_neutral(t) = r_i(t) − β_i(t-1) × r_SPY(t)

    Parameters
    ----------
    stock_ret   : (T, N) DataFrame of raw stock returns
    market_ret  : (T,)  Series of market returns (SPY)
    beta_df     : (T, N) DataFrame of rolling betas (already lagged)

    Returns
    -------
    pd.DataFrame, same shape as stock_ret
        Market-neutral returns.  The market component has been removed,
        so remaining variation is stock-specific (idiosyncratic to the market).

    Why does this still leave some market exposure?
        Rolling beta is an estimate, not the true (unknown) beta.
        If the true beta is 1.3 but we estimate 1.2, there's a residual
        0.1 × r_SPY still in the "neutral" return.  Walk-forward validation
        (which re-estimates beta on each training window) helps here.
    """
    # Align market returns to the stock return index
    mkt = market_ret.squeeze().reindex(stock_ret.index).ffill().fillna(0)
    assert isinstance(mkt, pd.Series)

    # beta_df.reindex() handles any date gaps between beta_df and stock_ret
    beta_aligned = beta_df.reindex(stock_ret.index).fillna(1.0)

    # stock_ret − beta × mkt (broadcast mkt across all N stock columns)
    return stock_ret - beta_aligned.mul(mkt, axis=0)


def freeze_and_apply_beta(
    test_ret:       pd.DataFrame,
    test_market_ret: pd.Series,
    last_train_beta: pd.Series,
) -> pd.DataFrame:
    """
    Apply a FROZEN beta (estimated on training data) to the test period.

    Used in walk-forward validation and OOS backtest so that no test-period
    data influences the beta estimate.

    Parameters
    ----------
    test_ret         : (T_test, N) raw test returns
    test_market_ret  : (T_test,)  SPY returns for the test period
    last_train_beta  : (N,)       beta from the last day of training

    Returns
    -------
    pd.DataFrame  market-neutral test returns

    Why freeze rather than re-estimate on test data?
        Re-estimating beta using test observations would mean the "neutralisation"
        uses future information.  In live trading, the beta model is updated nightly
        and the next day's positions use yesterday's estimate.  Freezing the final
        training-window beta is the conservative, clean way to simulate this.
    """
    mkt = test_market_ret.squeeze().reindex(test_ret.index).ffill().fillna(0)

    # Broadcast last_train_beta (N,) across all test rows:
    # np.outer(mkt_values, beta_values) → (T_test, N) matrix of beta*mkt_return
    beta_contribution = pd.DataFrame(
        np.outer(mkt.values, last_train_beta.values),
        index=test_ret.index,
        columns=test_ret.columns,
    )
    return test_ret - beta_contribution


# ── Diagnostic plots ───────────────────────────────────────────────────────────

def plot_beta_diagnostics(
    beta_df:    pd.DataFrame,
    raw_ret:    pd.DataFrame,
    mn_ret:     pd.DataFrame,
    spy_ret:    pd.Series,
) -> plt.Figure:
    """
    3-panel diagnostic figure to verify beta estimation worked correctly.

    Panel 1 — Beta distribution across the universe.
        Expected: centred around 1.0, most stocks between 0.5 and 1.8.
        High-beta stocks (growth, tech) cluster above 1; low-beta (utilities,
        consumer staples) cluster below 1.

    Panel 2 — Scatter: raw vol vs market-neutral vol (annualised).
        Expected: most dots fall BELOW the y = x diagonal, meaning
        market-neutral returns have lower volatility than raw returns.
        The market component has been stripped out.

    Panel 3 — Distribution of correlations with SPY, before and after.
        Expected: raw returns are highly correlated with SPY (~0.5–0.7).
        Market-neutral returns should cluster near 0.
        A remaining non-zero mean indicates residual market exposure
        (beta estimation error), which is normal and acceptable.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Colour palette (matches notebook aesthetic)
    BLUE, GREEN, AMBER, RED = "#58a6ff", "#3fb950", "#d29922", "#f85149"

    # ── Panel 1: Beta distribution ─────────────────────────────────────────────
    ax = axes[0]
    avg_betas = beta_df.mean()                    # time-average beta per stock
    ax.hist(avg_betas, bins=60, color=BLUE, alpha=0.8, edgecolor="none")
    ax.axvline(1.0, color=AMBER, lw=2, ls="--",
               label="β = 1 (market neutral)")
    ax.axvline(avg_betas.mean(), color=GREEN, lw=2, ls="-",
               label=f"Mean β = {avg_betas.mean():.2f}")
    ax.set_title("Cross-Sectional Beta Distribution")
    ax.set_xlabel("Average Rolling Beta (vs SPY)")
    ax.set_ylabel("Number of stocks")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Panel 2: Vol comparison (raw vs market-neutral) ────────────────────────
    ax2 = axes[1]
    raw_vols = (raw_ret.std() * np.sqrt(252)).values          # annualised
    mn_vols  = (mn_ret.std()  * np.sqrt(252)).values          # annualised
    ax2.scatter(raw_vols, mn_vols, alpha=0.35, s=12, color=BLUE)
    lim = max(raw_vols.max(), mn_vols.max()) * 1.05
    ax2.plot([0, lim], [0, lim], color=AMBER, lw=1, ls="--",
             label="y = x  (no vol change)")
    ax2.set_title("Volatility: Raw vs Market-Neutral")
    ax2.set_xlabel("Raw return vol (annualised)")
    ax2.set_ylabel("Market-neutral vol (annualised)")
    # Vol reduction percentage as annotation
    vol_reduction = 1 - mn_vols.mean() / raw_vols.mean()
    ax2.text(0.05, 0.92, f"Avg vol reduction: {vol_reduction:.1%}",
             transform=ax2.transAxes, color=GREEN, fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: SPY correlation before and after ──────────────────────────────
    ax3 = axes[2]
    spy_aligned = spy_ret.squeeze().reindex(raw_ret.index).ffill().fillna(0)
    corr_raw = raw_ret.corrwith(spy_aligned)     # correlation of each stock with SPY
    corr_mn  = mn_ret.corrwith(spy_aligned)      # same, after market-neutralisation
    ax3.hist(corr_raw, bins=50, color=RED,   alpha=0.6, label="Raw returns")
    ax3.hist(corr_mn,  bins=50, color=GREEN, alpha=0.6, label="Market-neutral")
    ax3.axvline(corr_raw.mean(), color=RED,   lw=2, ls="--",
                label=f"Mean raw corr = {corr_raw.mean():.2f}")
    ax3.axvline(corr_mn.mean(),  color=GREEN, lw=2, ls="--",
                label=f"Mean MN corr = {corr_mn.mean():.2f}")
    ax3.set_title("Distribution of Correlations with SPY")
    ax3.set_xlabel("Correlation with SPY")
    ax3.set_ylabel("Number of stocks")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    fig.suptitle("SPY Beta Estimation — Market Neutralisation Diagnostics",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_beta_neutrality(
    port_ret: pd.Series,
    spy_ret:  pd.Series,
    window:   int = 63,
) -> plt.Figure:
    """
    2-panel figure to verify that the PORTFOLIO (not just individual stocks)
    is market-neutral after beta-neutralising the returns.

    Panel 1 — Rolling 63-day portfolio beta vs SPY.
        Target: rolling beta should hover around 0.
        Occasional spikes are normal; persistent non-zero beta means the
        strategy is inadvertently taking a market view.

    Panel 2 — Scatter: portfolio daily return vs SPY daily return.
        Target: flat regression line (slope ≈ 0).
        The y-intercept of the regression line (daily alpha) should be positive.
        A steeply sloping line means significant market beta remains.
    """
    BLUE, GREEN, AMBER, RED = "#58a6ff", "#3fb950", "#d29922", "#f85149"

    # Align SPY to portfolio dates
    spy = spy_ret.squeeze().reindex(port_ret.index).ffill().fillna(0)

    # ── Rolling portfolio beta ─────────────────────────────────────────────────
    mkt_var  = spy.rolling(window, min_periods=window // 2).var()
    cov_pm   = port_ret.rolling(window, min_periods=window // 2).cov(spy)
    port_beta_roll = (cov_pm / mkt_var.replace(0, np.nan))

    # ── Overall (full-period) beta and alpha via linear regression ─────────────
    mask = spy.notna() & port_ret.notna()
    beta_overall, alpha_overall = np.polyfit(spy[mask], port_ret[mask], deg=1)
    alpha_ann = alpha_overall * 252

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Panel 1: Rolling beta over time
    ax = axes[0]
    ax.plot(port_beta_roll, color=BLUE, lw=1.5,
            label=f"{window}d rolling portfolio beta vs SPY")
    ax.axhline(0,            color="white", lw=1.0, ls="--", alpha=0.5,
               label="β = 0  (target)")
    ax.axhline(beta_overall, color=AMBER,  lw=1.0, ls=":",
               label=f"Full-period β = {beta_overall:.3f}")
    # Shade the ±0.1 "acceptable" band in faint green
    ax.fill_between(port_beta_roll.index, -0.1, 0.1, color=GREEN, alpha=0.05)
    ax.set_title(f"Rolling Portfolio Beta vs SPY  "
                 f"(full-period β={beta_overall:.3f}, α={alpha_ann:.2%}/yr)")
    ax.set_ylabel("Beta")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: Beta scatter
    ax2 = axes[1]
    ax2.scatter(spy[mask], port_ret[mask], alpha=0.15, s=5, color=BLUE)
    x_line = np.linspace(spy[mask].min(), spy[mask].max(), 100)
    ax2.plot(x_line, beta_overall * x_line + alpha_overall,
             color=RED, lw=2,
             label=f"β = {beta_overall:.3f}   α = {alpha_ann:.2%}/yr")
    ax2.axhline(0, color="white", lw=0.5, alpha=0.3)
    ax2.axvline(0, color="white", lw=0.5, alpha=0.3)
    ax2.set_title("Portfolio Returns vs SPY — Beta Scatter")
    ax2.set_xlabel("SPY daily return")
    ax2.set_ylabel("Portfolio daily return")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig