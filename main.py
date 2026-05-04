"""
main.py — Eigenvalue Factor Model: Full Beta-Neutral Pipeline

CHANGES FROM ORIGINAL main.py
──────────────────────────────
NEW Step 2  : SPY beta estimation + market-neutral return construction
              Uses src/beta.py (new module).

Step 3 (was 2): Train/test split — unchanged, but now applied AFTER beta neutralisation
                so that the market-neutral returns are split alongside raw returns.

Step 4 (was 3): Covariance estimation — now fitted on market-neutral training returns.
                The dominant market eigenvalue is no longer artificially inflated.

Step 5 (was 4): Factor model — fitted on market-neutral covariance.
                RMT selects fewer/different factors now that λ₁ (market) is removed.

Step 6 (was 5): Idiosyncratic extraction — from market-neutral returns.
                These idio returns are genuinely stock-specific.

Steps 7–10     : Signal construction, IC decay, IS/OOS backtest — unchanged logic,
                 but inputs are now market-neutral idiosyncratic returns.

Step 10 (OOS)  : Uses frozen beta (last training beta) applied to test period.
                 No test data influences the beta estimate.

Step 11 (WF)   : Walk-forward now passes spy_returns to Backtester.walk_forward(),
                 which handles beta neutralisation inside each window.

NEW Step 12    : Beta neutrality diagnostics — plots that verify the portfolio
                 actually came out market-neutral.

All original step numbering is preserved in comments for easy cross-referencing.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

# ── Imports from src modules ───────────────────────────────────────────────────
from src.data import (
    get_sp500_tickers,
    get_spy,                    # [NEW] SPY market benchmark loader
    load_prices,
    clean_prices,
    compute_returns,
    winsorize_returns,
    train_test_split,
)
from src.beta import (          # [NEW] entire module
    rolling_beta,
    market_neutral_returns,
    freeze_and_apply_beta,
    plot_beta_diagnostics,
    plot_beta_neutrality,
)
from src.covariance import ledoit_wolf_cov, compare_condition_numbers
from src.factors    import EigenFactorModel
from src.portfolio  import (
    build_signal,
    momentum_signal,
    combine_signals,
    long_short_portfolio,
    Backtester,
)
from src.evaluate   import (
    full_report,
    ic_decay,
    plot_performance,
    plot_ic_decay,
    plot_momentum_diagnostics,
    plot_signal_comparison,
)

# ── Config ─────────────────────────────────────────────────────────────────────
# All tunable parameters in one place.  Mirrors original main.py exactly.
CACHE_PATH   = "data/prices.parquet"
START_DATE   = "2020-01-01"
END_DATE     = "2026-04-29"
SPLIT_DATE   = "2024-12-31"
MR_LOOKBACK  = 20      # mean-reversion z-score window (days)
MR_DECAY     = 10      # mean-reversion EWM halflife
MOM_FAST     = 21      # momentum: 1-month fast window
MOM_SLOW     = 252     # momentum: 12-month slow window
MR_WEIGHT    = 0.3     # fixed fallback weight (dynamic weighting used in practice)
TC_BPS       = 7.0     # round-trip transaction cost in basis points
TRAIN_WINDOW = 504     # walk-forward train window (~2 years, keeps T > N≈490)
TEST_WINDOW  = 123     # walk-forward test window (~1 quarter)
WARMUP       = 60      # signal warmup period before trading begins (days)
BETA_WINDOW  = 63      # [NEW] rolling beta estimation window (~3 months)

Path("plots").mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Data loading
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1: Loading data")
print("=" * 60)

# Load S&P 500 tickers (from local CSV or Wikipedia)
tickers = get_sp500_tickers()

# Download/load stock prices (auto-cached to data/prices.parquet)
prices  = load_prices(tickers, start=START_DATE, end=END_DATE, cache_path=CACHE_PATH)
prices  = clean_prices(prices)

# Compute simple percentage returns (not log) for portfolio PnL compatibility
returns = compute_returns(prices, log=False)

# Winsorise at 0.1/99.9 percentile to cap earnings-gap outliers
returns = winsorize_returns(returns)

print(f"Final dataset : {returns.shape[0]} days x {returns.shape[1]} stocks\n")

# [NEW] Download SPY as the market benchmark
# SPY prices are used to estimate rolling beta for each stock.
# Cached separately to data/spy_prices.parquet.
spy_prices = get_spy(start=START_DATE, end=END_DATE)
spy_returns = winsorize_returns(
    compute_returns(spy_prices, log=False)
)
# .squeeze() ensures we have a Series, not a single-column DataFrame.
# This matters because rolling_beta() and market_neutral_returns() both
# call .mul(mkt, axis=0), which requires mkt to be a Series.
spy_returns = spy_returns.squeeze()

# Align SPY dates to the stock return index
# (SPY and stocks may occasionally have different trading calendars due to
#  data provider quirks — reindex + ffill handles any gaps)
spy_returns = spy_returns.reindex(returns.index).ffill()

print(f"SPY returns loaded: {len(spy_returns)} days")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — [NEW] SPY beta estimation & market-neutral returns
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2: Beta estimation & market neutralisation [NEW]")
print("=" * 60)
print(f"  Beta window: {BETA_WINDOW}d  |  Warmup to drop: {BETA_WINDOW} rows")

# Estimate rolling beta for every stock vs SPY over the full date range.
# Beta is LAGGED by 1 day inside rolling_beta(), so no lookahead.
#
# betas_full shape: (T, N)  where T = total trading days, N = stocks
# Each cell betas_full[t, i] = β_i estimated using data up to day t-1.
print("\nEstimating rolling betas (vectorised — all stocks simultaneously)...")
betas_full = rolling_beta(returns, spy_returns, window=BETA_WINDOW)

# Compute market-neutral returns: r_neutral = r_stock − β × r_SPY
# Shape: (T, N) — same as returns
mn_returns = market_neutral_returns(returns, spy_returns, betas_full)

# Sanity checks
assert mn_returns.shape == returns.shape, "Shape mismatch after neutralisation"
assert not mn_returns.isnull().all().any(), \
    f"All-NaN columns in mn_returns: {mn_returns.isnull().all().sum()}"

print(f"\nBeta estimation complete.")
print(f"  Mean beta across universe   : {betas_full.mean().mean():.3f}  (expected ~1)")
print(f"  Beta std across universe    : {betas_full.std().mean():.3f}")
raw_vol_mean = returns.std().mean()
mn_vol_mean  = mn_returns.std().mean()
print(f"  Raw daily vol (mean)        : {raw_vol_mean:.5f}")
print(f"  Market-neutral daily vol    : {mn_vol_mean:.5f}")
print(f"  Vol reduction from beta     : {1 - mn_vol_mean / raw_vol_mean:.1%}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Train / test split
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3: Train/test split")
print("=" * 60)

# Split the RAW returns on the date boundary.
# We keep the raw returns split for:
#   - Momentum signal (computed on raw returns, full history)
#   - Portfolio PnL calculation (must use raw returns, not neutralised)
train_returns, test_returns = train_test_split(returns, SPLIT_DATE)

# [NEW] Also split the market-neutral returns and the betas.
# These are used for covariance estimation and factor model fitting.
mn_train = mn_returns.loc[mn_returns.index <= SPLIT_DATE]
mn_test  = mn_returns.loc[mn_returns.index >  SPLIT_DATE]

beta_train = betas_full.loc[betas_full.index <= SPLIT_DATE]
beta_test  = betas_full.loc[betas_full.index >  SPLIT_DATE]

spy_train = spy_returns.loc[spy_returns.index <= SPLIT_DATE]
spy_test  = spy_returns.loc[spy_returns.index >  SPLIT_DATE]

# [NEW] Drop warmup rows from the TRAINING market-neutral returns.
# The first BETA_WINDOW rows have noisy beta estimates (insufficient history),
# so we exclude them from covariance fitting and factor model training.
# We keep the raw train_returns intact for momentum signal lookback.
warmup_cutoff     = mn_train.index[BETA_WINDOW]
mn_train_clean    = mn_train.loc[warmup_cutoff:]         # used for cov + PCA
train_ret_aligned = train_returns.loc[warmup_cutoff:]    # aligned raw returns (for signals)
beta_train_clean  = beta_train.loc[warmup_cutoff:]       # aligned betas (for diagnostics)

print(f"\nAfter dropping {BETA_WINDOW}-day beta warmup:")
print(f"  Clean train rows  : {len(mn_train_clean)}")
print(f"  Test rows         : {len(mn_test)}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Covariance estimation
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 4: Covariance estimation")
print("=" * 60)

# [NEW] Fitted on MARKET-NEUTRAL training returns instead of raw returns.
# Key difference: the dominant market factor (λ₁, previously ~31% of variance)
# has been removed, so the covariance matrix reflects sector/style co-movement.
print("\nCondition numbers (market-neutral returns):")
compare_condition_numbers(mn_train_clean.values)

# Ledoit-Wolf shrinkage on market-neutral returns.
# Returns (covariance_matrix, shrinkage_intensity).
cov, _ = ledoit_wolf_cov(mn_train_clean.values)

print("\nNote: Covariance fitted on market-neutral returns.")
print("      λ₁ should be much smaller than in the original (market factor removed).")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Eigenvalue factor model
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 5: Eigenvalue factor model")
print("=" * 60)

# [NEW] Fitted on market-neutral covariance.
# The factors now capture sector/style effects rather than a mix of
# market direction + sector + noise.
model = EigenFactorModel(n_factors="rmt")
model.fit(mn_train_clean.values, cov)

print(f"\nTop-5 eigenvalues (market-neutral covariance):")
for i, ev in enumerate(model.eigenvalues_[:5], 1):
    print(f"  λ_{i} = {ev:.4f}  ({model.explained_variance_ratio_[i-1]:.1%})")

print("\nFactor interpretability (top loadings per factor):")
interp = model.factor_interpretability(mn_train_clean.columns.tolist(), top_n=5)
print(interp.to_string())

fig = model.plot_eigenvalue_spectrum()
fig.savefig("plots/eigenvalue_spectrum.png", dpi=150, bbox_inches="tight")
print("\nSaved: plots/eigenvalue_spectrum.png")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Idiosyncratic return extraction
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 6: Extracting idiosyncratic returns")
print("=" * 60)

# [NEW] model.transform() is applied to MARKET-NEUTRAL training returns.
# Idiosyncratic returns = market-neutral returns − systematic factor component.
# This is the second layer of stripping:
#   raw returns
#   → minus market (beta × SPY) = market-neutral returns
#   → minus PCA systematic factors = idiosyncratic returns
#
# The resulting idio_train_df captures only stock-specific information.
factor_ret_train, idio_train = model.transform(mn_train_clean.values)
idio_train_df = pd.DataFrame(
    idio_train,
    index=mn_train_clean.index,
    columns=mn_train_clean.columns,
)

# Vol statistics — should be lower than the original because we've now
# stripped out both the market factor AND PCA systematic factors
total_vol = mn_train_clean.std().mean()
idio_vol  = idio_train_df.std().mean()
print(f"Average market-neutral daily vol : {total_vol:.4f}")
print(f"Average idiosyncratic daily vol  : {idio_vol:.4f}")
print(f"Additional vol reduction (PCA)   : {1 - idio_vol/total_vol:.1%}")

fc = model.factor_correlation_matrix(factor_ret_train)
print(f"\nFactor correlation matrix (top-left 5x5):")
print(fc.iloc[:5, :5].to_string())


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — Signal construction
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 7: Signal construction")
print("=" * 60)
print(f"  MR lookback: {MR_LOOKBACK}d | MR decay: {MR_DECAY}d | "
      f"Dynamic blend (not fixed {MR_WEIGHT:.0%}/{1-MR_WEIGHT:.0%})")

# ── Mean-reversion signal ──────────────────────────────────────────────────────
# Built on IDIOSYNCRATIC returns from market-neutral data.
# These are the cleanest possible returns: market removed by beta,
# then sector/style factors removed by PCA.
mr_train = build_signal(idio_train_df, lookback=MR_LOOKBACK, decay_halflife=MR_DECAY)

# ── Momentum signal ────────────────────────────────────────────────────────────
# Built on RAW total returns (not idiosyncratic, not market-neutral).
# Reason: momentum captures persistent price trends, including sector trends.
# Removing the market or sector component would strip out the very trends
# the signal is trying to capture.
mom_train = momentum_signal(
    idio_train_df,   # <-- CHANGE HERE
    fast_window=MOM_FAST,
    slow_window=MOM_SLOW,
    skip_days=5,
)

# ── Dynamic blending ───────────────────────────────────────────────────────────
# Instead of the fixed 30/70 blend in combine_signals(), we dynamically
# weight each signal proportional to its recent absolute signal strength.
#
# w_MR(t)  = strength_MR(t)  / (strength_MR(t)  + strength_Mom(t))
# w_Mom(t) = strength_Mom(t) / (strength_MR(t)  + strength_Mom(t))
#
# Where strength = rolling 60-day mean of |signal|.
# When MR is "louder" (large absolute values), it gets more weight.
# When momentum is "louder", momentum gets more weight.
# In volatile regimes, MR tends to dominate; in trending markets, momentum does.
mr_strength  = mr_train.abs().rolling(60, min_periods=10).mean()
mom_strength = mom_train.abs().rolling(60, min_periods=10).mean()

total = mr_strength + mom_strength

w_mr  = (mr_strength / total).fillna(0.5)   # fallback to 50/50 if no history
w_mom = (mom_strength / total).fillna(0.5)

signal_train = w_mr * mr_train + w_mom * mom_train

print(f"Signal stats:")
print(f"  Non-null observations : {signal_train.notna().sum().sum():,}")
print(f"  Cross-sectional mean  : {signal_train.mean().mean():.4f}  (should be ~0)")
print(f"  Cross-sectional std   : {signal_train.std().mean():.4f}  (should be ~0.58)")
print(f"\n  Mean-reversion only:")
print(f"    CS std: {mr_train.std().mean():.4f}")
print(f"  Momentum only:")
print(f"    CS std: {mom_train.std().mean():.4f}")
print(f"\n  Avg MR weight   : {w_mr.mean().mean():.2f}")
print(f"  Avg Mom weight  : {w_mom.mean().mean():.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — IC decay analysis (in-sample)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 8: IC decay analysis (in-sample)")
print("=" * 60)

# Information Coefficient (IC) = Spearman rank correlation between
# today's signal and future returns.  IC > 0.05 is considered meaningful.
# ICIR = mean(IC) / std(IC) = consistency-adjusted IC.
for name, sig in [("Mean Reversion", mr_train),
                   ("Momentum",       mom_train),
                   ("Combined",       signal_train)]:
    tbl = ic_decay(sig, train_ret_aligned, horizons=[1, 2, 5, 10, 21, 42, 63])
    print(f"\n  {name}:")
    print(tbl.round(4).to_string())

# MR IC decay plot
fig_ic = plot_ic_decay(ic_decay(mr_train, train_ret_aligned))
fig_ic.savefig("plots/ic_decay_mean_reversion.png", dpi=150, bbox_inches="tight")
print("\nSaved: plots/ic_decay_mean_reversion.png")

# Momentum 6-panel diagnostic dashboard
fig_mom = plot_momentum_diagnostics(mom_train, train_ret_aligned, label="Momentum")
fig_mom.savefig("plots/momentum_diagnostics.png", dpi=150, bbox_inches="tight")
print("Saved: plots/momentum_diagnostics.png")

# Side-by-side IC comparison
fig_cmp = plot_signal_comparison(
    {"Mean Reversion": mr_train, "Momentum": mom_train, "Combined": signal_train},
    train_ret_aligned,
    horizons=[1, 2, 5, 10, 21, 42, 63],
)
fig_cmp.savefig("plots/signal_comparison_ic.png", dpi=150, bbox_inches="tight")
print("Saved: plots/signal_comparison_ic.png")

# [NEW] Beta diagnostics plot — verify neutralisation worked
fig_beta_diag = plot_beta_diagnostics(
    beta_df  = beta_train_clean,
    raw_ret  = train_ret_aligned,
    mn_ret   = mn_train_clean,
    spy_ret  = spy_train,
)
fig_beta_diag.savefig("plots/beta_diagnostics.png", dpi=150, bbox_inches="tight")
print("Saved: plots/beta_diagnostics.png")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 9 — In-sample backtest
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 9: In-sample backtest")
print("=" * 60)

# Portfolio weights from the combined signal (top/bottom decile, equal weight)
# PnL is computed against RAW returns (not market-neutral).
# This is correct: the portfolio's actual dollar return uses real price moves.
weights_train = long_short_portfolio(
    signal_train, use_decile=True, returns=train_ret_aligned
)
bt = Backtester(train_ret_aligned, transaction_cost_bps=TC_BPS)
results_train = bt.run(weights_train)
full_report(results_train, label="(In-Sample)")
fig_is = plot_performance(results_train, title="Eigenvalue Factor Model — In-Sample")
fig_is.savefig("plots/is_performance.png", dpi=150, bbox_inches="tight")
print("Saved: plots/is_performance.png")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 10 — Out-of-sample backtest
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 10: Out-of-sample backtest")
print("=" * 60)

# [NEW] Use FROZEN beta for the test period.
# The frozen beta is the last beta from the training period.
# This is the beta available at the start of the test period —
# using any test-period beta would be lookahead.
last_train_beta = beta_train_clean.iloc[-1]   # shape: (N,)

# Apply frozen beta to market-neutralise test returns
mn_test_frozen = freeze_and_apply_beta(test_returns, spy_test, last_train_beta)

# Extract idiosyncratic returns using the SAME eigenvectors from training.
# model.transform() projects test returns onto training-era factors.
# Using training eigenvectors = no information from test period in the model.
_, idio_test = model.transform(mn_test_frozen.values)
idio_test_df = pd.DataFrame(
    idio_test,
    index=test_returns.index,
    columns=test_returns.columns,
)

# MR signal on test-period idiosyncratic returns
mr_test = build_signal(idio_test_df, lookback=MR_LOOKBACK, decay_halflife=MR_DECAY)

# Momentum signal: computed on FULL history (raw returns), sliced to test dates.
# KEY: we use the full `returns` DataFrame (all dates), then slice.
# Using only test_returns would give all-NaN because MOM_SLOW=252d > 123d test window.
mom_test = momentum_signal(idio_test_df, fast_window=MOM_FAST,
                            slow_window=MOM_SLOW, skip_days=5)
mom_test = mom_test.loc[test_returns.index]   # slice to test dates

# Dynamic blending for test period
mr_strength_t  = mr_test.abs().rolling(60, min_periods=10).mean()
mom_strength_t = mom_test.abs().rolling(60, min_periods=10).mean()
total_t        = mr_strength_t + mom_strength_t
w_mr_t         = (mr_strength_t / total_t).fillna(0.5)
w_mom_t        = (mom_strength_t / total_t).fillna(0.5)
signal_test    = w_mr_t * mr_test + w_mom_t * mom_test

# Portfolio and backtest against RAW test returns
weights_test = long_short_portfolio(
    signal_test, use_decile=True, returns=test_returns
)
bt_test      = Backtester(test_returns, transaction_cost_bps=TC_BPS)
results_test = bt_test.run(weights_test)
full_report(results_test, label="(Out-of-Sample)")

fig_oos = plot_performance(results_test, title="Eigenvalue Factor Model — Out-of-Sample")
fig_oos.savefig("plots/oos_performance.png", dpi=150, bbox_inches="tight")
print("Saved: plots/oos_performance.png")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 11 — Walk-forward validation
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 11: Walk-forward validation")
print("=" * 60)
print(f"  Train window: {TRAIN_WINDOW}d | Test window: {TEST_WINDOW}d")


def make_signal(idio_df: pd.DataFrame, full_ret: pd.DataFrame) -> pd.DataFrame:
    """
    Signal construction function called once per walk-forward window.

    Arguments provided by Backtester.walk_forward():
        idio_df  : idiosyncratic returns for the TEST window only.
                   These are already market-neutral + PCA-neutralised
                   (the neutralisation happens inside walk_forward()
                   when spy_returns is provided).
        full_ret : raw returns for all dates up to end of this test window.
                   Used for momentum's 252-day lookback.

    Returns
    -------
    pd.DataFrame  — combined signal, indexed to test window dates.

    Why pass full_ret for momentum?
        Momentum needs 252 trading days of history (MOM_SLOW=252).
        Each test window is only 123 days.  If we built momentum only on
        the test window data, the signal would be all-NaN for the first
        252 rows — meaning the portfolio would hold nothing.
        Using full_ret gives momentum the lookback it needs.
    """
    # MR signal built on truly idiosyncratic returns (already stripped of
    # both market and PCA systematic components)
    mr = build_signal(idio_df, lookback=MR_LOOKBACK, decay_halflife=MR_DECAY)

    # Momentum built on full raw returns history, then sliced to test window
    mom = momentum_signal(
        idio_df,
        fast_window=MOM_FAST,
        slow_window=MOM_SLOW,
        skip_days=5,
    )  # align to test window

    # Dynamic weighting (same logic as Steps 7 and 10)
    mr_strength  = mr.abs().rolling(60, min_periods=10).mean()
    mom_strength = mom.abs().rolling(60, min_periods=10).mean()
    total        = mr_strength + mom_strength
    w_mr         = (mr_strength / total).fillna(0.5)
    w_mom        = (mom_strength / total).fillna(0.5)

    signal = w_mr * mr + w_mom * mom
    return signal.fillna(0)


# [NEW] Pass spy_returns to walk_forward().
# This tells the backtester to apply beta neutralisation inside each window:
#   - Estimate rolling beta on each training split
#   - Freeze last training beta
#   - Neutralise test returns
#   - Fit PCA on market-neutral training returns
# All of this happens automatically inside Backtester.walk_forward() when
# spy_returns is provided (see src/portfolio.py for the implementation).
bt_full    = Backtester(returns, transaction_cost_bps=TC_BPS)
wf_results = bt_full.walk_forward(
    signal_fn    = make_signal,
    train_window = TRAIN_WINDOW,
    test_window  = TEST_WINDOW,
    n_factors    = "rmt",
    verbose      = True,
    spy_returns  = spy_returns,    # [NEW] enables beta neutralisation per window
    beta_window  = BETA_WINDOW,    # [NEW] rolling window for beta estimation
)

full_report(wf_results, label="(Walk-Forward OOS)")

fig_wf = plot_performance(
    wf_results, title="Eigenvalue Factor Model — Walk-Forward OOS"
)
fig_wf.savefig("plots/walk_forward_performance.png", dpi=150, bbox_inches="tight")
print("Saved: plots/walk_forward_performance.png")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 12 — [NEW] Beta neutrality diagnostics
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 12: Beta neutrality diagnostics [NEW]")
print("=" * 60)

# Combine IS and OOS portfolio returns into a single series
port_ret_full = pd.concat([results_train["net_pnl"], results_test["net_pnl"]])

# Compute and print portfolio-level beta statistics
spy_port_aligned = spy_returns.reindex(port_ret_full.index).fillna(0)
mask = spy_port_aligned.notna() & port_ret_full.notna()
beta_port, alpha_port = np.polyfit(spy_port_aligned[mask], port_ret_full[mask], 1)

print(f"\nPortfolio market-exposure check:")
print(f"  Full-period portfolio beta  : {beta_port:.4f}  (target ~0)")
print(f"  Annualised portfolio alpha  : {alpha_port * 252:.2%}")
print(f"  SPY correlation             : {port_ret_full.corr(spy_port_aligned):.3f}")
print(f"\n  A beta close to 0 confirms the beta-neutralisation is working.")
print(f"  A positive alpha means the strategy earns returns INDEPENDENT of market.")

# Plot portfolio-level beta diagnostics
fig_bn = plot_beta_neutrality(port_ret_full, spy_returns)
fig_bn.savefig("plots/beta_neutrality.png", dpi=150, bbox_inches="tight")
print("\nSaved: plots/beta_neutrality.png")


# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("COMPLETE")
print("=" * 60)
print("  plots/beta_diagnostics.png        — [NEW] Beta distribution, vol reduction, SPY corr")
print("  plots/eigenvalue_spectrum.png     — RMT factor count (market-neutral covariance)")
print("  plots/ic_decay_mean_reversion.png — MR signal IC by horizon")
print("  plots/momentum_diagnostics.png    — 6-panel momentum dashboard:")
print("                                         IC decay (short+long horizons)")
print("                                         Daily IC time series")
print("                                         Top/bottom decile spread")
print("                                         Signal autocorrelation")
print("                                         Cross-sectional dispersion")
print("                                         Rolling ICIR")
print("  plots/signal_comparison_ic.png    — MR vs Momentum vs Combined IC")
print("  plots/is_performance.png          — In-sample 4-panel dashboard")
print("  plots/oos_performance.png         — Out-of-sample 4-panel dashboard")
print("  plots/walk_forward_performance.png— Walk-forward OOS dashboard")
print("  plots/beta_neutrality.png         — [NEW] Portfolio beta scatter and rolling beta")