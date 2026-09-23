import numpy as np
import pandas as pd


def build_raw_stock_weights(
    positions,
    position_size=0.05
):
    """Convert position states into raw portfolio weights."""

    return (
        positions.astype(float)
        * position_size
    )


def pca_neutralize_weights(
    raw_weights,
    Q_k,
    max_gross=0.50
):
    """Remove exposure to the retained PCA factors."""

    w_raw = raw_weights.to_numpy()

    projection = Q_k @ Q_k.T

    w_neutral = (
        np.eye(len(w_raw)) - projection
    ) @ w_raw

    weights = pd.Series(
        w_neutral,
        index=raw_weights.index
    )

    gross = weights.abs().sum()

    if gross > max_gross:
        weights *= max_gross / gross

    return weights


def factor_exposure(weights, Q_k):
    """Compute exposure to the retained PCA factors."""

    return (
        Q_k.T
        @ weights.to_numpy()
    )


def net_exposure(weights):
    """Compute net portfolio exposure."""

    return weights.sum()


def gross_exposure(weights):
    """Compute gross portfolio exposure."""

    return weights.abs().sum()


def calculate_turnover(
    new_weights,
    old_weights
):
    """Compute gross traded notional as a fraction of equity."""

    all_assets = (
        new_weights.index
        .union(old_weights.index)
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