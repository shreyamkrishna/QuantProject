"""
Eigenvalue Factor Model — Full Pipeline
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

from src.data import (
    get_sp500_tickers, load_prices, clean_prices,
    compute_returns, winsorize_returns, train_test_split,
)
from src.covariance import ledoit_wolf_cov, compare_condition_numbers
from src.factors    import EigenFactorModel
from src.portfolio  import (
    build_signal, momentum_signal, combine_signals,
    long_short_portfolio, Backtester,
)
from src.evaluate   import (
    full_report, ic_decay, plot_performance, plot_ic_decay,
    plot_momentum_diagnostics, plot_signal_comparison,
)

# ── Config ─────────────────────────────────────────────────────────────────────
CACHE_PATH   = "data/prices.parquet"
START_DATE   = "2020-01-01"
END_DATE     = "2026-04-22"
SPLIT_DATE   = "2024-12-31"
MR_LOOKBACK  = 10      # mean-reversion z-score window (days)
MR_DECAY     = 5       # mean-reversion EWM halflife
MOM_FAST     = 21      # momentum: 1-month fast window
MOM_SLOW     = 252     # momentum: 12-month slow window
MR_WEIGHT    = 0.5     # 50% MR / 50% momentum blend
TC_BPS       = 7.0
TRAIN_WINDOW = 504     # walk-forward: 2yr train (keeps T > N=490)
TEST_WINDOW  = 63      # walk-forward: 1 quarter (gives ~12 folds)

Path("plots").mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)

# ── 1. Data ────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading data")
print("=" * 60)

tickers = get_sp500_tickers()
prices  = load_prices(tickers, start=START_DATE, end=END_DATE, cache_path=CACHE_PATH)
prices  = clean_prices(prices)
returns = compute_returns(prices, log=False)
returns = winsorize_returns(returns)
print(f"Final dataset : {returns.shape[0]} days x {returns.shape[1]} stocks\n")

# ── 2. Train / test split ──────────────────────────────────────────────────────
print("=" * 60)
print("STEP 2: Train/test split")
print("=" * 60)

train_returns, test_returns = train_test_split(returns, SPLIT_DATE)

# ── 3. Covariance ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Covariance estimation")
print("=" * 60)

print("\nCondition numbers:")
compare_condition_numbers(train_returns.values)
cov, _ = ledoit_wolf_cov(train_returns.values)

# ── 4. Factor model ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Eigenvalue factor model")
print("=" * 60)

model = EigenFactorModel(n_factors="rmt")
model.fit(train_returns.values, cov)

print(f"\nTop-5 eigenvalues:")
for i, ev in enumerate(model.eigenvalues_[:5], 1):
    print(f"  λ_{i} = {ev:.4f}  ({model.explained_variance_ratio_[i-1]:.1%})")

print("\nFactor interpretability (top loadings per factor):")
interp = model.factor_interpretability(train_returns.columns.tolist(), top_n=5)
print(interp.to_string())

fig = model.plot_eigenvalue_spectrum()
fig.savefig("plots/eigenvalue_spectrum.png", dpi=150, bbox_inches="tight")
print("\nSaved: plots/eigenvalue_spectrum.png")

# ── 5. Idiosyncratic returns ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Extracting idiosyncratic returns")
print("=" * 60)

factor_ret_train, idio_train = model.transform(train_returns.values)
idio_train_df = pd.DataFrame(idio_train, index=train_returns.index,
                              columns=train_returns.columns)

total_vol = train_returns.std().mean()
idio_vol  = idio_train_df.std().mean()
print(f"Average total daily vol        : {total_vol:.4f}")
print(f"Average idiosyncratic daily vol: {idio_vol:.4f}")
print(f"Vol reduction                  : {1 - idio_vol/total_vol:.1%}")

fc = model.factor_correlation_matrix(factor_ret_train)
print(f"\nFactor correlation matrix (top-left 5x5):")
print(fc.iloc[:5, :5].to_string())

# ── 6. Signal construction ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Signal construction")
print("=" * 60)
print(f"  MR lookback: {MR_LOOKBACK}d | MR decay: {MR_DECAY}d | "
      f"Blend: {MR_WEIGHT:.0%} MR / {1-MR_WEIGHT:.0%} Momentum")

# Mean reversion on idiosyncratic returns
mr_train  = build_signal(idio_train_df, lookback=MR_LOOKBACK, decay_halflife=MR_DECAY)

# 12-1 month momentum on total returns
mom_train = momentum_signal(train_returns, fast_window=MOM_FAST,
                             slow_window=MOM_SLOW, skip_days=5)

# Blended signal
signal_train = combine_signals(mr_train, mom_train, mr_weight=MR_WEIGHT)

print(f"Signal stats:")
print(f"  Non-null observations : {signal_train.notna().sum().sum():,}")
print(f"  Cross-sectional mean  : {signal_train.mean().mean():.4f}  (should be ~0)")
print(f"  Cross-sectional std   : {signal_train.std().mean():.4f}  (should be ~0.58)")

print(f"\n  Mean-reversion only:")
print(f"    CS std: {mr_train.std().mean():.4f}")
print(f"  Momentum only:")
print(f"    CS std: {mom_train.std().mean():.4f}")

# ── 7. IC decay ────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: IC decay analysis (in-sample)")
print("=" * 60)

for name, sig in [("Mean Reversion", mr_train),
                   ("Momentum",       mom_train),
                   ("Combined",       signal_train)]:
    tbl = ic_decay(sig, train_returns, horizons=[1, 2, 5, 10, 21, 42, 63])
    print(f"\n  {name}:")
    print(tbl.round(4).to_string())

# ── Eigenvalue / MR IC decay (existing plot) ──────────────────────────────────
fig_ic = plot_ic_decay(ic_decay(mr_train, train_returns))
fig_ic.savefig("plots/ic_decay_mean_reversion.png", dpi=150, bbox_inches="tight")
print("\nSaved: plots/ic_decay_mean_reversion.png")

# ── Momentum 6-panel diagnostic dashboard ────────────────────────────────────
fig_mom = plot_momentum_diagnostics(mom_train, train_returns, label="Momentum")
fig_mom.savefig("plots/momentum_diagnostics.png", dpi=150, bbox_inches="tight")
print("Saved: plots/momentum_diagnostics.png")

# ── Side-by-side IC comparison across all three signals ──────────────────────
fig_cmp = plot_signal_comparison(
    {"Mean Reversion": mr_train, "Momentum": mom_train, "Combined": signal_train},
    train_returns,
    horizons=[1, 2, 5, 10, 21, 42, 63],
)
fig_cmp.savefig("plots/signal_comparison_ic.png", dpi=150, bbox_inches="tight")
print("Saved: plots/signal_comparison_ic.png")

# ── 8. In-sample backtest ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8: In-sample backtest")
print("=" * 60)

weights_train = long_short_portfolio(
    signal_train, use_decile=True, returns=train_returns
)
bt = Backtester(train_returns, transaction_cost_bps=TC_BPS)
results_train = bt.run(weights_train)
full_report(results_train, label="(In-Sample)")

# ── 9. Out-of-sample backtest ──────────────────────────────────────────────────
print("=" * 60)
print("STEP 9: Out-of-sample backtest")
print("=" * 60)

_, idio_test = model.transform(test_returns.values)
idio_test_df = pd.DataFrame(idio_test, index=test_returns.index,
                             columns=test_returns.columns)

mr_test      = build_signal(idio_test_df, lookback=MR_LOOKBACK, decay_halflife=MR_DECAY)
mom_test     = momentum_signal(returns, fast_window=MOM_FAST,
                                slow_window=MOM_SLOW, skip_days=5)
mom_test     = mom_test.loc[test_returns.index]   # align to test period
signal_test  = combine_signals(mr_test, mom_test, mr_weight=MR_WEIGHT)

weights_test = long_short_portfolio(
    signal_test, use_decile=True, returns=test_returns
)
bt_test      = Backtester(test_returns, transaction_cost_bps=TC_BPS)
results_test = bt_test.run(weights_test)
full_report(results_test, label="(Out-of-Sample)")

fig_oos = plot_performance(results_test, title="Eigenvalue Factor Model — Out-of-Sample")
fig_oos.savefig("plots/oos_performance.png", dpi=150, bbox_inches="tight")
print("Saved: plots/oos_performance.png")

# ── 10. Walk-forward validation ────────────────────────────────────────────────
print("=" * 60)
print("STEP 10: Walk-forward validation")
print("=" * 60)
print(f"  Train window: {TRAIN_WINDOW}d | Test window: {TEST_WINDOW}d")

def make_signal(idio_df: pd.DataFrame, full_ret: pd.DataFrame) -> pd.DataFrame:
    """
    Called per walk-forward window.
    idio_df  : idiosyncratic returns for the test window only
    full_ret : all returns up to end of test window (for momentum lookback)
    No extra shift here — the backtester already applies a 1-day lag.
    """
    mr  = build_signal(idio_df, lookback=MR_LOOKBACK, decay_halflife=MR_DECAY)
    mom = momentum_signal(full_ret, fast_window=MOM_FAST,
                           slow_window=MOM_SLOW, skip_days=5)
    mom = mom.loc[idio_df.index]   # trim to test window
    return combine_signals(mr, mom, mr_weight=MR_WEIGHT)

bt_full    = Backtester(returns, transaction_cost_bps=TC_BPS)
wf_results = bt_full.walk_forward(
    signal_fn    = make_signal,
    train_window = TRAIN_WINDOW,
    test_window  = TEST_WINDOW,
    n_factors    = "rmt",
    verbose      = True,
)

full_report(wf_results, label="(Walk-Forward OOS)")

fig_wf = plot_performance(
    wf_results, title="Eigenvalue Factor Model — Walk-Forward OOS"
)
fig_wf.savefig("plots/walk_forward_performance.png", dpi=150, bbox_inches="tight")
print("Saved: plots/walk_forward_performance.png")

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("COMPLETE")
print("=" * 60)
print("  plots/eigenvalue_spectrum.png        — RMT factor count diagnostics")
print("  plots/ic_decay_mean_reversion.png    — MR signal IC by horizon")
print("  plots/momentum_diagnostics.png       — 6-panel momentum dashboard:")
print("                                           IC decay (short+long horizons)")
print("                                           Daily IC time series")
print("                                           Top/bottom decile spread")
print("                                           Signal autocorrelation")
print("                                           Cross-sectional dispersion")
print("                                           Rolling ICIR")
print("  plots/signal_comparison_ic.png       — MR vs Momentum vs Combined IC")
print("  plots/oos_performance.png            — OOS 4-panel dashboard")
print("  plots/walk_forward_performance.png   — Walk-forward OOS")
