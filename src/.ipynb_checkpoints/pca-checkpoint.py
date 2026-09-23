import numpy as np


def covariance_matrix(X_centered):
    """
    Sample covariance matrix:

        Sigma = X^T X / (T - 1)

    X_centered shape:
        (T, N)
    """

    T = X_centered.shape[0]

    return X_centered.T @ X_centered / (T - 1)


def fit_pca(X, k):
    """
    Fit PCA manually using eigendecomposition
    of the covariance matrix.

    Parameters
    ----------
    X : np.ndarray
        Shape (T, N).
        Rows = days.
        Columns = stocks.

    k : int
        Number of principal components to keep.

    Returns
    -------
    mean : np.ndarray, shape (N,)
    eigenvalues : np.ndarray, shape (N,)
    eigenvectors : np.ndarray, shape (N, N)
    explained_variance_ratio : np.ndarray, shape (N,)
    Q_k : np.ndarray, shape (N, k)
    """

    if X.ndim != 2:
        raise ValueError("X must be a 2D matrix.")

    T, N = X.shape

    if not 1 <= k <= N:
        raise ValueError(f"k must satisfy 1 <= k <= {N}")

    # Mean return of every stock.
    mean = X.mean(axis=0)

    # Center the data.
    X_centered = X - mean

    # Covariance matrix.
    Sigma = covariance_matrix(X_centered)

    # Sigma is symmetric, so use eigh.
    eigenvalues, eigenvectors = np.linalg.eigh(Sigma)

    # np.linalg.eigh returns eigenvalues
    # from smallest to largest.
    order = np.argsort(eigenvalues)[::-1]

    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Fraction of total variance explained by each PC.
    explained_variance_ratio = (
        eigenvalues / eigenvalues.sum()
    )

    # First k eigenvectors.
    Q_k = eigenvectors[:, :k]

    return (
        mean,
        eigenvalues,
        eigenvectors,
        explained_variance_ratio,
        Q_k
    )


def transform(X, mean, Q_k):
    """
    Convert stock coordinates to PCA coordinates.

        C_k = (X - mean) Q_k

    Output shape:
        (T, k)
    """

    X_centered = X - mean

    return X_centered @ Q_k


def reconstruct(X, mean, Q_k):
    """
    Reconstruct the centered component explained
    by the first k principal components.

        X_hat = (X - mean) Q_k Q_k^T

    Note:
    X_hat is still in centered-return space.
    """

    X_centered = X - mean

    return X_centered @ Q_k @ Q_k.T