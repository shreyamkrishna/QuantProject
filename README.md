# Eigenvalue Factor Model

Eigenvalue-based equity factor model applying Random Matrix Theory to extract
idiosyncratic alpha signals from S&P 500 return data.

Methodology directly derived from PhD work on signal extraction in high-noise
high-dimensional datasets (radio telescope data → equity returns).

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

- `plots/eigenvalue_spectrum.png` — scree plot + Marchenko-Pastur overlay
- `plots/ic_decay.png` — IC vs forward horizon (signal half-life)
- `plots/oos_performance.png` — simple OOS performance
- `plots/walk_forward_performance.png` — walk-forward validated performance

## Requirements

Python 3.9+
