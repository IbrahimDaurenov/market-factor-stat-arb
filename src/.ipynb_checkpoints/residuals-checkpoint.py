import numpy as np
import pandas as pd

from .pca import fit_pca


def compute_residual(x, mean, Q_k):
    """Compute the PCA residual for one observation."""

    x_centered = x - mean

    factor_component = (
        Q_k
        @ (Q_k.T @ x_centered)
    )

    return (
        x_centered
        - factor_component
    )


def rolling_pca_residuals(
    stock_returns,
    pca_window=252,
    k=6
):
    """Compute rolling out-of-sample PCA residuals."""

    residuals = pd.DataFrame(
        np.nan,
        index=stock_returns.index,
        columns=stock_returns.columns
    )

    for t in range(
        pca_window,
        len(stock_returns)
    ):
        train = stock_returns.iloc[
            t - pca_window:t
        ].to_numpy()

        x_t = (
            stock_returns
            .iloc[t]
            .to_numpy()
        )

        mean, _, _, _, Q_k = fit_pca(
            train,
            k=k
        )

        residuals.iloc[t] = compute_residual(
            x=x_t,
            mean=mean,
            Q_k=Q_k
        )

    return residuals


def rolling_residual_spread(
    residuals,
    window=20
):
    """Compute rolling cumulative PCA residuals."""

    return residuals.rolling(
        window=window,
        min_periods=window
    ).sum()