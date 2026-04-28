import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List, Optional


# ── Core metrics ───────────────────────────────────────────────────────────────

def sharpe_ratio(returns: pd.Series, annualise: bool = True) -> float:
    """Annualised Sharpe ratio (252 trading days)."""
    if returns.std() == 0:
        return np.nan
    sr = returns.mean() / returns.std()
    return sr * np.sqrt(252) if annualise else sr


def sortino_ratio(returns: pd.Series, annualise: bool = True) -> float:
    """Sortino ratio — penalises only downside volatility."""
    downside = returns[returns < 0].std()
    if downside == 0:
        return np.nan
    sr = returns.mean() / downside
    return sr * np.sqrt(252) if annualise else sr


def max_drawdown(cum_returns: pd.Series) -> float:
    """Maximum peak-to-trough drawdown (negative number)."""
    rolling_max = cum_returns.cummax()
    dd = (cum_returns - rolling_max) / rolling_max
    return float(dd.min())


def calmar_ratio(returns: pd.Series, cum_returns: pd.Series) -> float:
    """Annualised return / |max drawdown|."""
    ann_return = returns.mean() * 252
    mdd = abs(max_drawdown(cum_returns))
    return ann_return / mdd if mdd > 0 else np.nan


def hit_rate(returns: pd.Series) -> float:
    """Fraction of days with positive PnL."""
    return float((returns > 0).mean())


# ── Information Coefficient ────────────────────────────────────────────────────

def information_coefficient(
    signal: pd.DataFrame,
    forward_returns: pd.DataFrame,
    horizon: int = 1,
) -> pd.Series:
    """
    Daily IC = Spearman rank correlation between signal and forward returns.

    IC > 0.05 is considered meaningful in industry.
    IC > 0.10 is strong.
    IC < 0.02 is noise.

    Parameters
    ----------
    signal          : cross-sectional signal DataFrame (T x N)
    forward_returns : raw returns DataFrame (T x N)
    horizon         : forward return horizon in days
    """
    fwd = forward_returns.shift(-horizon)

    daily_ic = []
    for date in signal.index:
        s = signal.loc[date].dropna()
        f = fwd.loc[date].dropna()
        common = s.index.intersection(f.index)
        if len(common) < 10:
            daily_ic.append(np.nan)
            continue
        ic = s[common].rank().corr(f[common].rank(), method="spearman")
        daily_ic.append(ic)

    return pd.Series(daily_ic, index=signal.index, name=f"IC_h{horizon}")


def ic_decay(
    signal: pd.DataFrame,
    returns: pd.DataFrame,
    horizons: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    IC at multiple horizons — shows signal half-life.

    IC should drop as horizon increases.
    The horizon where IC crosses zero is the signal's effective half-life.
    """
    if horizons is None:
        horizons = [1, 2, 5, 10, 21]

    rows = []
    for h in horizons:
        ic = information_coefficient(signal, returns, horizon=h)
        rows.append({
            "horizon_days": h,
            "mean_ic": ic.mean(),
            "ic_std": ic.std(),
            "icir": ic.mean() / ic.std() if ic.std() > 0 else np.nan,
            "frac_positive": float((ic > 0).mean()),
        })
    return pd.DataFrame(rows).set_index("horizon_days")


# ── Full performance report ────────────────────────────────────────────────────

def full_report(results: pd.DataFrame, label: str = "") -> Dict[str, str]:
    """
    Print and return a full performance summary.
    """
    r = results["net_pnl"]
    cum = results["cum_net"]

    metrics = {
        "Annualised Return": f"{r.mean() * 252:.2%}",
        "Annualised Volatility": f"{r.std() * np.sqrt(252):.2%}",
        "Sharpe Ratio": f"{sharpe_ratio(r):.2f}",
        "Sortino Ratio": f"{sortino_ratio(r):.2f}",
        "Max Drawdown": f"{max_drawdown(cum):.2%}",
        "Calmar Ratio": f"{calmar_ratio(r, cum):.2f}",
        "Win Rate": f"{hit_rate(r):.2%}",
        "Avg Daily Turnover": f"{results['turnover'].mean():.2%}",
        "Avg Daily Cost (bps)": f"{results['costs'].mean() * 10_000:.2f}",
        "Cost Drag (ann)": f"{results['costs'].mean() * 252:.2%}",
    }

    title = f"── Performance Report {label} ──"
    print(f"\n{title}")
    print("─" * len(title))
    for k, v in metrics.items():
        print(f"  {k:<28} {v}")
    print()

    return metrics


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_performance(
    results: pd.DataFrame,
    title: str = "Eigenvalue Factor Model Strategy",
    benchmark: Optional[pd.Series] = None,
) -> plt.Figure:
    """
    4-panel performance dashboard:
      1. Cumulative PnL (gross vs net, optional benchmark)
      2. Drawdown
      3. Rolling 63-day Sharpe
      4. Daily PnL distribution
    """
    fig = plt.figure(figsize=(14, 11))
    gs = gridspec.GridSpec(4, 1, hspace=0.45)

    # ── Panel 1: Cumulative PnL ────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    results["cum_gross"].plot(ax=ax1, color="#4a90d9", linewidth=1.5,
                              label="Gross PnL", alpha=0.7)
    results["cum_net"].plot(ax=ax1, color="#e8533a", linewidth=2,
                            label="Net of costs")
    if benchmark is not None:
        (1 + benchmark).cumprod().reindex(results.index).plot(
            ax=ax1, color="grey", linewidth=1.2, linestyle="--",
            label="Benchmark"
        )
    ax1.axhline(1, color="black", linewidth=0.6, linestyle=":")
    ax1.set_ylabel("Cumulative Return")
    ax1.set_title(title, fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: Drawdown ──────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    cum = results["cum_net"]
    dd = (cum - cum.cummax()) / cum.cummax()
    dd.plot(ax=ax2, color="#c0392b", linewidth=0.9)
    ax2.fill_between(dd.index, dd, 0, alpha=0.25, color="#c0392b")
    ax2.axhline(0, color="black", linewidth=0.6)
    ax2.set_ylabel("Drawdown")
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: Rolling Sharpe ────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    roll_sharpe = (
        results["net_pnl"].rolling(63).mean()
        / results["net_pnl"].rolling(63).std()
    ) * np.sqrt(252)
    roll_sharpe.plot(ax=ax3, color="#27ae60", linewidth=1.2)
    ax3.axhline(0, color="black", linewidth=0.6)
    ax3.axhline(1, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
    ax3.set_ylabel("Rolling Sharpe (63d)")
    ax3.grid(True, alpha=0.3)

    # ── Panel 4: PnL distribution ──────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[3])
    pnl = results["net_pnl"].dropna()
    ax4.hist(pnl, bins=80, color="#4a90d9", alpha=0.7, edgecolor="none",
             density=True)
    ax4.axvline(pnl.mean(), color="#e8533a", linewidth=1.5,
                label=f"Mean = {pnl.mean()*100:.3f}%")
    ax4.axvline(0, color="black", linewidth=0.8)
    ax4.set_ylabel("Density")
    ax4.set_xlabel("Daily PnL")
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    return fig


def plot_ic_decay(ic_table: pd.DataFrame) -> plt.Figure:
    """Plot IC vs. horizon with error bars."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    horizons = ic_table.index.tolist()

    # IC mean with ±1 std bands
    ax = axes[0]
    ax.bar(horizons, ic_table["mean_ic"],
           color=["steelblue" if v > 0 else "#c0392b"
                  for v in ic_table["mean_ic"]],
           alpha=0.75, width=0.8)
    ax.errorbar(horizons, ic_table["mean_ic"],
                yerr=ic_table["ic_std"] / np.sqrt(len(horizons)),
                fmt="none", color="black", capsize=4)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.axhline(0.05, color="green", linewidth=0.8, linestyle="--",
               label="IC = 0.05 (meaningful)")
    ax.set_xlabel("Forward horizon (days)")
    ax.set_ylabel("Mean IC")
    ax.set_title("IC Decay")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ICIR
    ax = axes[1]
    ax.bar(horizons, ic_table["icir"],
           color=["steelblue" if v > 0 else "#c0392b"
                  for v in ic_table["icir"]],
           alpha=0.75, width=0.8)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Forward horizon (days)")
    ax.set_ylabel("IC Information Ratio")
    ax.set_title("IC Information Ratio by Horizon")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ── Momentum-specific diagnostics ─────────────────────────────────────────────

def plot_momentum_diagnostics(
    mom_signal: pd.DataFrame,
    returns: pd.DataFrame,
    label: str = "Momentum",
) -> plt.Figure:
    """
    6-panel diagnostic figure for the momentum signal.

    Panel 1 — IC decay at short AND long horizons (1..126 days).
              Momentum IC should be low at h=1, peak around h=21-63,
              and slowly decay. Flat or rising IC curve = genuine trend.

    Panel 2 — Daily IC time series with 63-day rolling mean.
              Shows whether momentum alpha is consistent across regimes
              or concentrated in a few episodes.

    Panel 3 — Decile spread: average return of top decile minus bottom decile.
              The raw return spread is the gross alpha before any
              position sizing. Should be positive and stable.

    Panel 4 — Signal autocorrelation (lag 1..30 days).
              Momentum should have HIGH autocorrelation (signal persists).
              Mean reversion has near-zero or negative autocorrelation.
              This confirms the signal is genuinely slow-moving.

    Panel 5 — Cross-sectional signal distribution over time (violin-style heatmap).
              Checks that the signal stays spread out and doesn't collapse.

    Panel 6 — Rolling 63-day ICIR.
              ICIR = IC mean / IC std. Periods with ICIR > 0.3 are
              reliably profitable. Negative ICIR = signal breakdown.
    """
    import matplotlib.colors as mcolors

    # ── Pre-compute all ingredients ────────────────────────────────────────────
    horizons_long = [1, 5, 10, 21, 42, 63, 126]
    ic_rows = []
    daily_ic_h1 = information_coefficient(mom_signal, returns, horizon=1)
    daily_ic_h21 = information_coefficient(mom_signal, returns, horizon=21)

    for h in horizons_long:
        ic = information_coefficient(mom_signal, returns, horizon=h)
        ic_rows.append({
            "horizon": h,
            "mean_ic": ic.mean(),
            "ic_std":  ic.std(),
            "icir":    ic.mean() / ic.std() if ic.std() > 0 else np.nan,
        })
    ic_long = pd.DataFrame(ic_rows).set_index("horizon")

    # Decile spread: actual returns of top vs bottom decile
    fwd1 = returns.shift(-1)
    top_ret    = []
    bot_ret    = []
    spread_ret = []
    for date in mom_signal.index[:-1]:
        row = mom_signal.loc[date].dropna()
        if len(row) < 20:
            top_ret.append(np.nan); bot_ret.append(np.nan); spread_ret.append(np.nan)
            continue
        cutoff   = max(1, len(row) // 10)
        top_tkrs = row.nlargest(cutoff).index
        bot_tkrs = row.nsmallest(cutoff).index
        fwd_row  = fwd1.loc[date].dropna()
        t = fwd_row.reindex(top_tkrs).mean()
        b = fwd_row.reindex(bot_tkrs).mean()
        top_ret.append(t); bot_ret.append(b); spread_ret.append(t - b)

    spread_s = pd.Series(spread_ret, index=mom_signal.index[:-1])
    top_s    = pd.Series(top_ret,    index=mom_signal.index[:-1])
    bot_s    = pd.Series(bot_ret,    index=mom_signal.index[:-1])

    # Signal autocorrelation (mean across stocks)
    max_lag = 30
    auto_vals = []
    for lag in range(1, max_lag + 1):
        per_stock = [
            mom_signal[col].autocorr(lag=lag)
            for col in mom_signal.columns
            if mom_signal[col].notna().sum() > lag + 10
        ]
        auto_vals.append(np.nanmean(per_stock))
    acf = pd.Series(auto_vals, index=range(1, max_lag + 1))

    # Rolling ICIR (63-day window on h=21 IC)
    roll_icir = (
        daily_ic_h21.rolling(63).mean()
        / daily_ic_h21.rolling(63).std()
    )

    # ── Build figure ───────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 18))
    gs  = gridspec.GridSpec(3, 2, hspace=0.50, wspace=0.35)

    BLUE   = "#4a90d9"
    RED    = "#e8533a"
    GREEN  = "#27ae60"
    ORANGE = "#f39c12"

    # ── Panel 1: Long-horizon IC decay ────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    colors_bar = [GREEN if v > 0 else RED for v in ic_long["mean_ic"]]
    ax1.bar(ic_long.index, ic_long["mean_ic"], color=colors_bar,
            alpha=0.75, width=3)
    ax1.errorbar(ic_long.index, ic_long["mean_ic"],
                 yerr=ic_long["ic_std"] / np.sqrt(len(horizons_long)),
                 fmt="none", color="black", capsize=4)
    ax1.axhline(0,    color="black", linewidth=0.7)
    ax1.axhline(0.05, color=GREEN,  linewidth=0.9, linestyle="--",
                label="IC = 0.05 (meaningful)")
    ax1.set_xlabel("Forward horizon (days)")
    ax1.set_ylabel("Mean IC")
    ax1.set_title(f"{label} — IC Decay (short + long horizons)\n"
                  "Momentum IC should peak at 21–63 days", fontsize=10)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(ic_long.index)

    # ── Panel 2: Daily IC time series (h=1 and h=21) ─────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    daily_ic_h1.plot(ax=ax2, color=BLUE, alpha=0.25, linewidth=0.6,
                     label="Daily IC (h=1)")
    daily_ic_h1.rolling(63).mean().plot(ax=ax2, color=BLUE, linewidth=1.8,
                                         label="63d rolling mean (h=1)")
    daily_ic_h21.rolling(63).mean().plot(ax=ax2, color=ORANGE, linewidth=1.8,
                                          linestyle="--",
                                          label="63d rolling mean (h=21)")
    ax2.axhline(0, color="black", linewidth=0.7)
    ax2.set_ylabel("IC")
    ax2.set_title(f"{label} — Daily IC Time Series\n"
                  "Flat rolling mean = consistent alpha across regimes", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: Decile spread ────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    cum_top    = (1 + top_s.fillna(0)).cumprod()
    cum_bot    = (1 + bot_s.fillna(0)).cumprod()
    cum_spread = (1 + spread_s.fillna(0)).cumprod()
    cum_top.plot(ax=ax3, color=GREEN, linewidth=1.5, label="Top decile (long)")
    cum_bot.plot(ax=ax3, color=RED,   linewidth=1.5, label="Bot decile (short)")
    cum_spread.plot(ax=ax3, color="black", linewidth=2.0, linestyle="--",
                    label="Spread (long − short)")
    ax3.axhline(1, color="black", linewidth=0.6, linestyle=":")
    ax3.set_ylabel("Cumulative gross return")
    ax3.set_title(f"{label} — Decile Spread (gross, 1d forward)\n"
                  "Top should outperform bottom if signal has predictive power", fontsize=10)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ── Panel 4: Signal autocorrelation ──────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.bar(acf.index, acf.values, color=BLUE, alpha=0.75, width=0.8)
    ax4.axhline(0, color="black", linewidth=0.7)
    # 95% confidence band for white noise
    n = mom_signal.notna().all(axis=1).sum()
    ci = 1.96 / np.sqrt(n) if n > 0 else 0.1
    ax4.axhline( ci, color=GREEN, linewidth=0.9, linestyle="--",
                label=f"95% CI (±{ci:.3f})")
    ax4.axhline(-ci, color=GREEN, linewidth=0.9, linestyle="--")
    ax4.set_xlabel("Lag (days)")
    ax4.set_ylabel("Mean autocorrelation across stocks")
    ax4.set_title(f"{label} — Signal Autocorrelation\n"
                  "High ACF = signal persists (momentum). Near-zero = mean reversion",
                  fontsize=10)
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # ── Panel 5: Cross-sectional signal std over time ─────────────────────────
    ax5 = fig.add_subplot(gs[2, 0])
    cs_std  = mom_signal.std(axis=1)
    cs_mean = mom_signal.mean(axis=1)
    cs_std.plot(ax=ax5, color=BLUE, linewidth=1.0, label="CS std")
    ax5.fill_between(cs_std.index,
                     cs_mean - cs_std, cs_mean + cs_std,
                     alpha=0.15, color=BLUE, label="±1 CS std band")
    ax5.axhline(cs_std.mean(), color=ORANGE, linewidth=1.2, linestyle="--",
                label=f"Mean std = {cs_std.mean():.3f}")
    ax5.set_ylabel("Cross-sectional std of signal")
    ax5.set_title(f"{label} — Signal Cross-Sectional Dispersion\n"
                  "Collapse toward 0 = signal has lost discriminating power", fontsize=10)
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # ── Panel 6: Rolling ICIR ─────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 1])
    roll_icir.plot(ax=ax6, color=ORANGE, linewidth=1.4)
    ax6.axhline(0,   color="black", linewidth=0.7)
    ax6.axhline(0.3, color=GREEN,  linewidth=0.9, linestyle="--",
                label="ICIR = 0.3 (consistently profitable)")
    ax6.axhline(-0.3, color=RED,   linewidth=0.9, linestyle="--",
                label="ICIR = −0.3 (signal breakdown)")
    ax6.fill_between(roll_icir.index, roll_icir, 0,
                     where=roll_icir > 0,  alpha=0.15, color=GREEN)
    ax6.fill_between(roll_icir.index, roll_icir, 0,
                     where=roll_icir <= 0, alpha=0.15, color=RED)
    ax6.set_ylabel("Rolling ICIR (63d, h=21)")
    ax6.set_title(f"{label} — Rolling ICIR\n"
                  "Green = regime where momentum is working", fontsize=10)
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)

    fig.suptitle(f"{label} Signal — Full Diagnostic Dashboard",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    return fig


def plot_signal_comparison(
    signals: dict,          # {"Mean Reversion": df, "Momentum": df, "Combined": df}
    returns: pd.DataFrame,
    horizons: list = None,
) -> plt.Figure:
    """
    Side-by-side IC decay for all signals on one chart.
    Makes it easy to see which signal dominates at which horizon.
    """
    if horizons is None:
        horizons = [1, 2, 5, 10, 21, 42, 63]

    colors = ["#4a90d9", "#e8533a", "#27ae60", "#9b59b6"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for i, (name, sig) in enumerate(signals.items()):
        ic_vals, icir_vals = [], []
        for h in horizons:
            ic = information_coefficient(sig, returns, horizon=h)
            ic_vals.append(ic.mean())
            icir_vals.append(ic.mean() / ic.std() if ic.std() > 0 else np.nan)

        color = colors[i % len(colors)]
        axes[0].plot(horizons, ic_vals,  "o-", color=color, linewidth=2,
                     markersize=5, label=name)
        axes[1].plot(horizons, icir_vals, "o-", color=color, linewidth=2,
                     markersize=5, label=name)

    for ax, ylabel, title in zip(
        axes,
        ["Mean IC", "ICIR"],
        ["IC by Horizon — All Signals\n(shows each signal's predictive range)",
         "ICIR by Horizon — All Signals\n(accounts for IC consistency, not just level)"],
    ):
        ax.axhline(0,    color="black", linewidth=0.7)
        ax.axhline(0.05, color="grey",  linewidth=0.8, linestyle="--", alpha=0.7,
                   label="IC = 0.05")
        ax.set_xlabel("Forward horizon (days)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(horizons)

    plt.tight_layout()
    return fig
