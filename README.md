# Eigenvalue Factor Model

Eigenvalue-based equity factor model applying Random Matrix Theory to extract
idiosyncratic alpha signals from S&P 500 return data.

Methodology directly derived from PhD work on signal extraction in high-noise
high-dimensional datasets.

## Structure

```
factor_model/
├── src/
│   ├── data.py        # price download, cleaning, return computation
│   ├── covariance.py  # Ledoit-Wolf shrinkage, condition number analysis
│   ├── factors.py     # eigendecomposition, Marchenko-Pastur RMT
│   ├── portfolio.py   # signal construction, long-short portfolio, backtester
│   └── evaluate.py    # Sharpe, IC decay, performance plots
├── data/              # cached price parquet (auto-created)
├── plots/             # output charts (auto-created)
├── main.py            # full pipeline
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
python main.py
```

First run downloads ~500 tickers from Yahoo Finance and caches to `data/prices.parquet`.
Subsequent runs load from cache.

## Key Methodology

| Step | Method | Analogy to PhD |
|------|--------|----------------|
| Covariance estimation | Ledoit-Wolf shrinkage | Regularised least-squares, avoids ill-conditioned matrix inversion |
| Factor count selection | Marchenko-Pastur RMT | Signal vs noise threshold from random matrix theory |
| Factor decomposition | Eigenvalue decomposition | fPCA on time-series, energy concentration in leading eigencomponents |
| Signal extraction | Idiosyncratic return z-score | Residual after projecting out systematic components |
| Validation | Walk-forward OOS | Out-of-sample testing across multiple configurations |

## Output
  plots/beta_diagnostics.png        — Beta distribution, vol reduction, SPY corr
  
  plots/eigenvalue_spectrum.png     — RMT factor count (market-neutral covariance)
  
  plots/ic_decay_mean_reversion.png — MR signal IC by horizon
  
  plots/momentum_diagnostics.png    — 6-panel momentum dashboard: IC decay (short+long horizons),Daily IC                                          time series, Top/bottom decile spread, Signal autocorrelation,                                               Cross-sectional dispersion, Rolling ICIR
  
  plots/signal_comparison_ic.png    — MR vs Momentum vs Combined IC
  
  plots/is_performance.png          — In-sample 4-panel dashboard
  
  plots/oos_performance.png         — Out-of-sample 4-panel dashboard
  
  plots/walk_forward_performance.png— Walk-forward OOS dashboard
  
  plots/beta_neutrality.png         — Portfolio beta scatter and rolling beta
  
## Requirements

Python 3.9+
