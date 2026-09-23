import numpy as np


def covariance_matrix(X_centered):
    """Compute the sample covariance matrix."""

    T = X_centered.shape[0]

    return (
        X_centered.T
        @ X_centered
        / (T - 1)
    )


def fit_pca(X, k):
    """Fit PCA using covariance eigendecomposition."""

    if X.ndim != 2:
        raise ValueError(
            "X must be a 2D matrix."
        )

    _, N = X.shape

    if not 1 <= k <= N:
        raise ValueError(
            f"k must satisfy 1 <= k <= {N}"
        )

    mean = X.mean(axis=0)

    X_centered = X - mean

    Sigma = covariance_matrix(
        X_centered
    )

    eigenvalues, eigenvectors = (
        np.linalg.eigh(Sigma)
    )

    order = np.argsort(
        eigenvalues
    )[::-1]

    eigenvalues = (
        eigenvalues[order]
    )

    eigenvectors = (
        eigenvectors[:, order]
    )

    explained_variance_ratio = (
        eigenvalues
        / eigenvalues.sum()
    )

    Q_k = eigenvectors[:, :k]

    return (
        mean,
        eigenvalues,
        eigenvectors,
        explained_variance_ratio,
        Q_k
    )


def transform(X, mean, Q_k):
    """Project returns into PCA coordinates."""

    X_centered = X - mean

    return X_centered @ Q_k


def reconstruct(X, mean, Q_k):
    """Reconstruct returns from the retained PCs."""

    X_centered = X - mean

    return (
        X_centered
        @ Q_k
        @ Q_k.T
    )