import numpy as np
import pandas as pd

from .pca import fit_pca


def compute_residual(x, mean, Q_k):
    """
    Compute PCA residual for one day.

    x shape:
        (N,)

    Mathematics:

        z = x - mean

        x_hat = Q_k Q_k^T z

        e = z - x_hat
    """

    x_centered = x - mean

    factor_component = Q_k @ (Q_k.T @ x_centered)

    residual = x_centered - factor_component

    return residual


def rolling_pca_residuals(
    stock_returns,
    pca_window=252,
    k=6
):
    """
    Compute out-of-sample PCA residuals.

    For every day t:

        1. Fit PCA using days [t-L, ..., t-1]
        2. Observe return x_t
        3. Compute residual e_t

    Future data is never used.

    Parameters
    ----------
    stock_returns : pd.DataFrame
        Rows = dates
        Columns = stocks

    pca_window : int
        Number of past days used to estimate PCA.

    k : int
        Number of PCs retained.

    Returns
    -------
    residuals : pd.DataFrame
        Same columns as stock_returns.
        First pca_window rows are NaN.
    """

    residuals = pd.DataFrame(
        np.nan,
        index=stock_returns.index,
        columns=stock_returns.columns
    )

    for t in range(pca_window, len(stock_returns)):

        # Only past data:
        train = stock_returns.iloc[
            t - pca_window:t
        ].to_numpy()

        # Today's return:
        x_t = stock_returns.iloc[t].to_numpy()

        mean, _, _, _, Q_k = fit_pca(
            train,
            k=k
        )

        e_t = compute_residual(
            x=x_t,
            mean=mean,
            Q_k=Q_k
        )

        residuals.iloc[t] = e_t

    return residuals


def rolling_residual_spread(
    residuals,
    window=20
):
    """
    Rolling H-day cumulative residual:

        s_t = e_t + e_{t-1} + ... + e_{t-H+1}
    """

    spread = residuals.rolling(
        window=window,
        min_periods=window
    ).sum()

    return spread