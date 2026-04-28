import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Union


# ── Marchenko-Pastur distribution ─────────────────────────────────────────────

def marchenko_pastur_bounds(
    T: int, N: int, sigma2: float = 1.0
) -> Tuple[float, float]:
    """
    Under the null that returns are i.i.d. Gaussian, eigenvalues of the
    sample correlation matrix fall within [lambda_min, lambda_max].
    q = T/N must be > 1.
    """
    q = T / N
    assert q > 1, f"Need T > N for Marchenko-Pastur. Got T={T}, N={N}."
    lambda_max = sigma2 * (1 + 1 / np.sqrt(q)) ** 2
    lambda_min = sigma2 * (1 - 1 / np.sqrt(q)) ** 2
    return lambda_min, lambda_max


def marchenko_pastur_pdf(
    lambda_: np.ndarray, T: int, N: int, sigma2: float = 1.0
) -> np.ndarray:
    q = T / N
    lmin, lmax = marchenko_pastur_bounds(T, N, sigma2)
    pdf = np.zeros_like(lambda_, dtype=float)
    mask = (lambda_ >= lmin) & (lambda_ <= lmax)
    l = lambda_[mask]
    pdf[mask] = (q / (2 * np.pi * sigma2 * l)) * np.sqrt(
        (lmax - l) * (l - lmin)
    )
    return pdf


def rmt_factor_count(T: int, N: int, corr: np.ndarray) -> int:
    """
    Factor count via Marchenko-Pastur on the correlation matrix.
    Falls back gracefully to a broken-stick heuristic when T <= N.
    """
    evals_corr = np.linalg.eigvalsh(corr)

    if T > N:
        _, lambda_max = marchenko_pastur_bounds(T, N, sigma2=1.0)
        k = int(np.sum(evals_corr > lambda_max))
    else:
        # Underdetermined (T <= N): broken-stick fallback
        evals_sorted = np.sort(evals_corr)[::-1]
        total = evals_sorted.sum()
        n = len(evals_sorted)
        broken_stick = np.array([
            sum(1.0 / j for j in range(i + 1, n + 1)) / n
            for i in range(n)
        ])
        actual_prop = evals_sorted / total
        k = int(np.sum(actual_prop > broken_stick))
        k = max(3, min(k, T // 10))

    return max(1, k)


# ── Core Factor Model ──────────────────────────────────────────────────────────

class EigenFactorModel:
    """
    Eigenvalue-based factor model.  Decomposes returns as:
        r_t = B @ f_t + epsilon_t
    where epsilon_t is the idiosyncratic (tradeable) component.
    """

    def __init__(self, n_factors: Union[str, int] = "rmt"):
        self.n_factors = n_factors
        self.eigenvalues_: Optional[np.ndarray] = None
        self.eigenvectors_: Optional[np.ndarray] = None
        self.n_factors_selected_: Optional[int] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None
        self._T: Optional[int] = None
        self._N: Optional[int] = None

    def fit(self, returns: np.ndarray, cov: np.ndarray) -> "EigenFactorModel":
        T, N = returns.shape
        self._T = T
        self._N = N

        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        self.eigenvalues_ = eigenvalues[idx]
        self.eigenvectors_ = eigenvectors[:, idx]
        self.explained_variance_ratio_ = self.eigenvalues_ / self.eigenvalues_.sum()

        if self.n_factors == "rmt":
            std = np.sqrt(np.diag(cov))
            corr = cov / np.outer(std, std)
            k = rmt_factor_count(T, N, corr)
            self.n_factors_selected_ = k
        elif self.n_factors == "kaiser":
            self.n_factors_selected_ = max(1, int(np.sum(self.eigenvalues_ > 1.0)))
        else:
            self.n_factors_selected_ = int(self.n_factors)

        k = self.n_factors_selected_
        var_explained = float(self.explained_variance_ratio_[:k].sum())
        print(f"Factors selected : {k} / {N}")
        print(f"Variance explained by factors : {var_explained:.1%}")
        return self

    def transform(self, returns: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        assert self.eigenvectors_ is not None, "Call fit() first."
        k = self.n_factors_selected_
        B = self.eigenvectors_[:, :k]
        factor_returns = returns @ B
        systematic = factor_returns @ B.T
        idiosyncratic = returns - systematic
        return factor_returns, idiosyncratic

    def plot_eigenvalue_spectrum(self, figsize: Tuple[int, int] = (13, 5)) -> plt.Figure:
        assert self.eigenvalues_ is not None, "Call fit() first."
        T, N = self._T, self._N
        k = self.n_factors_selected_

        fig, axes = plt.subplots(1, 2, figsize=figsize)
        ax = axes[0]
        n_show = min(60, N)
        colors = ["steelblue" if i < k else "lightgrey" for i in range(n_show)]
        ax.bar(range(1, n_show + 1), self.explained_variance_ratio_[:n_show],
               color=colors, edgecolor="none")
        ax2 = ax.twinx()
        cumvar = np.cumsum(self.explained_variance_ratio_[:n_show])
        ax2.plot(range(1, n_show + 1), cumvar, "r-", linewidth=2, label="Cumulative")
        ax.axvline(x=k + 0.5, color="black", linestyle="--", linewidth=1.2,
                   label=f"K = {k} factors")
        ax.set_xlabel("Eigenvalue rank")
        ax.set_ylabel("Fraction of variance")
        ax2.set_ylabel("Cumulative variance")
        ax.set_title("Scree Plot")
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, fontsize=9)

        ax = axes[1]
        sigma2 = float(self.eigenvalues_.mean())
        if T > N:
            lmin, lmax = marchenko_pastur_bounds(T, N, sigma2=sigma2)
            bulk = self.eigenvalues_[self.eigenvalues_ < lmax * 4]
            ax.hist(bulk, bins=60, density=True, alpha=0.55, color="steelblue",
                    label="Empirical eigenvalues")
            x = np.linspace(max(lmin * 0.5, 1e-6), lmax * 1.3, 400)
            mp = marchenko_pastur_pdf(x, T, N, sigma2=sigma2)
            ax.plot(x, mp, "r-", linewidth=2, label="Marchenko-Pastur")
            ax.axvline(x=lmax, color="black", linestyle="--", linewidth=1.2,
                       label=f"λ_max = {lmax:.3f}")
        else:
            ax.hist(self.eigenvalues_, bins=60, density=True, alpha=0.55,
                    color="steelblue", label="Empirical eigenvalues")
        ax.set_xlabel("Eigenvalue")
        ax.set_ylabel("Density")
        ax.set_title("Empirical Eigenvalues vs Marchenko-Pastur")
        ax.legend(fontsize=9)

        fig.suptitle("Eigenvalue Factor Model — Spectrum Analysis", fontsize=13,
                     fontweight="bold")
        plt.tight_layout()
        return fig

    def factor_correlation_matrix(self, factor_returns: np.ndarray) -> pd.DataFrame:
        corr = np.corrcoef(factor_returns.T)
        return pd.DataFrame(corr).round(3)

    def factor_interpretability(self, tickers: list, top_n: int = 10) -> pd.DataFrame:
        assert self.eigenvectors_ is not None, "Call fit() first."
        k = self.n_factors_selected_
        B = self.eigenvectors_[:, :k]
        rows = []
        for i in range(k):
            loadings = pd.Series(B[:, i], index=tickers)
            top_pos = loadings.nlargest(top_n).index.tolist()
            top_neg = loadings.nsmallest(top_n).index.tolist()
            rows.append({
                "factor": i + 1,
                "top_positive_loadings": ", ".join(top_pos),
                "top_negative_loadings": ", ".join(top_neg),
                "loading_std": float(loadings.std()),
            })
        return pd.DataFrame(rows).set_index("factor")
