import numpy as np
from typing import Dict, Tuple
from sklearn.covariance import LedoitWolf, OAS


def sample_covariance(returns: np.ndarray) -> np.ndarray:
    """
    Naive sample covariance.
    Ill-conditioned when N is close to T — included for comparison only.
    """
    return np.cov(returns.T)


def ledoit_wolf_cov(returns: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Ledoit-Wolf analytical shrinkage estimator.
    Shrinks sample covariance toward scaled identity matrix.
    Optimal shrinkage intensity chosen analytically (no cross-validation needed).

    Directly analogous to the regularised least-squares approach in the PhD:
    avoids inversion of ill-conditioned covariance matrices.
    """
    lw = LedoitWolf()
    lw.fit(returns)
    return lw.covariance_, lw.shrinkage_


def oas_cov(returns: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Oracle Approximating Shrinkage (OAS) estimator.
    Often outperforms Ledoit-Wolf when N >> T.
    """
    oas = OAS()
    oas.fit(returns)
    return oas.covariance_, oas.shrinkage_


def exponential_weighted_cov(
    returns: np.ndarray,
    halflife: int = 63,  # ~3 months in trading days
) -> np.ndarray:
    """
    Exponentially weighted covariance — more weight on recent data.
    Better captures non-stationary volatility and changing regimes.
    """
    T, N = returns.shape
    decay = np.exp(-np.log(2) / halflife)
    weights = np.array([decay ** i for i in range(T - 1, -1, -1)])
    weights /= weights.sum()

    mu = (returns * weights[:, None]).sum(axis=0)
    demeaned = returns - mu

    cov = (demeaned * weights[:, None]).T @ demeaned
    return cov


def compare_condition_numbers(returns: np.ndarray) -> Dict[str, float]:
    """
    Condition number = max eigenvalue / min eigenvalue.
    High condition number = ill-conditioned = numerically unstable inversions.

    This directly motivates the use of shrinkage estimation.
    """
    sample = sample_covariance(returns)
    lw, shrinkage = ledoit_wolf_cov(returns)
    oasCOV, oasShrinkage = oas_cov(returns)
    ewCOV = exponential_weighted_cov(returns)

    def condition_number(cov: np.ndarray) -> float:
        evals = np.linalg.eigvalsh(cov)
        #evals = evals[evals > 0]  # guard against numerical negatives
        return float(evals.max() / evals.min())

    result = {
        "sample_cov": condition_number(sample),
        "ledoit_wolf": condition_number(lw),
        "oas_cov": condition_number(oasCOV),
        "ew_cov": condition_number(ewCOV),
        "lw_shrinkage_intensity": shrinkage,
        "oas_shrinkage_intensity": oasShrinkage
    }

    print(f"  Sample covariance condition number      : {result['sample_cov']:>15,.0f}")
    print(f"  Ledoit-Wolf condition number            : {result['ledoit_wolf']:>15,.0f}")
    print(f"  OAS condition number                    : {result['oas_cov']:>15,.0f}")
    print(f"  Exponentially weighted condition number : {result['ew_cov']:>15,.0f}")
    print(f"  LW Shrinkage intensity                  : {result['lw_shrinkage_intensity']:>15.4f}")
    print(f"  OAS Shrinkage intensity                  : {result['oas_shrinkage_intensity']:>15.4f}")

    return result
