import numpy as np
import pandas as pd


def build_raw_stock_weights(
    positions,
    position_size=0.05
):
    """
    Raw alpha portfolio before factor neutralization.

    long  -> +position_size
    short -> -position_size
    """

    return positions.astype(float) * position_size


def pca_neutralize_weights(
    raw_weights,
    Q_k,
    max_gross=0.50
):
    """
    Remove exposure to the first k PCA factors.

        w_neutral = (I - Q_k Q_k^T) w_raw

    Therefore:

        Q_k^T w_neutral = 0

    Parameters
    ----------
    raw_weights : pd.Series, shape (N,)
    Q_k : np.ndarray, shape (N, k)
    max_gross : float
        Maximum total stock gross exposure.

    Returns
    -------
    pd.Series
    """

    w_raw = raw_weights.to_numpy()

    # Projection onto residual subspace.
    projection = Q_k @ Q_k.T

    w_neutral = (
        np.eye(len(w_raw)) - projection
    ) @ w_raw

    weights = pd.Series(
        w_neutral,
        index=raw_weights.index
    )

    # Projection can create small positions
    # in many stocks. Keep gross exposure bounded.
    gross = weights.abs().sum()

    if gross > max_gross:
        weights *= max_gross / gross

    return weights


def factor_exposure(weights, Q_k):
    """
    Exposure to retained PCA factors:

        Q_k^T w

    Should be approximately zero
    after neutralization.
    """

    return Q_k.T @ weights.to_numpy()


def net_exposure(weights):
    return weights.sum()


def gross_exposure(weights):
    return weights.abs().sum()


def calculate_turnover(
    new_weights,
    old_weights
):
    """
    Gross traded notional:

        turnover = sum_i |w_t,i - w_{t-1,i}|
    """

    all_assets = new_weights.index.union(
        old_weights.index
    )

    new_weights = new_weights.reindex(
        all_assets,
        fill_value=0.0
    )

    old_weights = old_weights.reindex(
        all_assets,
        fill_value=0.0
    )

    return (
        new_weights - old_weights
    ).abs().sum()