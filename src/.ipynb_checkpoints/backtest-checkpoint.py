import numpy as np
import pandas as pd

from .residuals import (
    rolling_pca_residuals,
    rolling_residual_spread
)

from .signals import (
    rolling_zscore,
    rolling_empirical_quantiles,
    rolling_ar1_parameters,
    update_positions
)

from .portfolio import (
    build_raw_stock_weights,
    pca_neutralize_weights,
    factor_exposure,
    calculate_turnover,
    net_exposure,
    gross_exposure
)

from .pca import fit_pca


def rolling_pc1_share(
    stock_returns,
    pca_window=252
):
    """
    Rolling strength of the first PCA factor.

    For day t, PCA uses ONLY:

        t-pca_window, ..., t-1

    PC1 share:

        lambda_1 / sum(lambda_j)
    """

    pc1_share = pd.Series(
        np.nan,
        index=stock_returns.index,
        dtype=float
    )

    for t in range(
        pca_window,
        len(stock_returns)
    ):

        train = stock_returns.iloc[
            t - pca_window:t
        ].to_numpy()

        _, _, _, evr, _ = fit_pca(
            train,
            k=1
        )

        pc1_share.iloc[t] = evr[0]

    return pc1_share


def run_backtest(
    stock_returns,

    # ==================================================
    # PCA
    # ==================================================

    pca_window=252,
    k=6,

    # ==================================================
    # RESIDUAL SIGNAL
    # ==================================================

    spread_window=40,

    signal_mode="zscore",

    z_window=252,
    entry_z=2.5,

    lower_q=0.025,
    upper_q=0.975,

    # ==================================================
    # OPTIONAL AR(1)
    # ==================================================

    use_ar1_filter=False,
    ar_window=252,
    phi_min=0.0,
    phi_max=1.0,

    # ==================================================
    # PC1 REGIME FILTER
    # ==================================================

    use_pc1_regime_filter=True,

    regime_window=252,
    regime_quantile=0.50,

    # ==================================================
    # TRADING
    # ==================================================

    max_holding=20,
    position_size=0.10,
    max_positions=5,

    # ==================================================
    # PORTFOLIO
    # ==================================================

    max_gross=0.50,
    cost_bps=10,
    rebalance_interval=20
):
    """
    Rolling PCA residual mean-reversion strategy.

    Main signal:
        extreme PCA residual spread.

    Optional regime filter:
        new trades are allowed only when current
        PC1 explained-variance share is above its
        past-only rolling historical quantile.

    Existing positions are NOT closed automatically
    when the market enters a low-PC1 regime.
    """

    # ==================================================
    # 1. PCA RESIDUALS
    # ==================================================

    residuals = rolling_pca_residuals(
        stock_returns,
        pca_window=pca_window,
        k=k
    )

    # ==================================================
    # 2. H-DAY RESIDUAL SPREAD
    # ==================================================

    spreads = rolling_residual_spread(
        residuals,
        window=spread_window
    )

    # ==================================================
    # 3. SIGNAL STATISTICS
    # ==================================================

    zscores = rolling_zscore(
        spreads,
        window=z_window
    )

    (
        lower_threshold,
        median_threshold,
        upper_threshold
    ) = rolling_empirical_quantiles(
        spreads,
        window=z_window,
        lower_q=lower_q,
        upper_q=upper_q
    )

    # ==================================================
    # 4. OPTIONAL AR(1)
    # ==================================================

    (
        ar1_alpha,
        ar1_phi,
        ar1_half_life
    ) = rolling_ar1_parameters(
        residuals,
        window=ar_window
    )

    # ==================================================
    # 5. PC1 REGIME
    # ==================================================

    pc1_share = rolling_pc1_share(
        stock_returns,
        pca_window=pca_window
    )

    # IMPORTANT:
    # today's threshold does not contain today's
    # PC1 share.
    pc1_threshold = (
        pc1_share
        .shift(1)
        .rolling(
            window=regime_window,
            min_periods=regime_window
        )
        .quantile(regime_quantile)
    )

    high_pc1_regime = (
        pc1_share > pc1_threshold
    )

    # ==================================================
    # 6. INITIAL PORTFOLIO STATE
    # ==================================================

    tickers = stock_returns.columns

    positions = pd.Series(
        0,
        index=tickers,
        dtype=int
    )

    holding_days = pd.Series(
        0,
        index=tickers,
        dtype=int
    )

    previous_raw_weights = pd.Series(
        0.0,
        index=tickers
    )

    previous_weights = pd.Series(
        0.0,
        index=tickers
    )

    position_history = []
    weight_history = []
    performance_history = []

    cost_rate = (
        cost_bps / 10000
    )

    days_since_rebalance = (
        rebalance_interval
    )

    # ==================================================
    # 7. FIND FIRST VALID DATE
    # ==================================================

    if signal_mode == "zscore":

        valid_signal = (
            zscores
            .notna()
            .any(axis=1)
        )

    elif signal_mode == "quantile":

        valid_signal = (
            lower_threshold.notna().any(axis=1)
            &
            median_threshold.notna().any(axis=1)
            &
            upper_threshold.notna().any(axis=1)
        )

    else:

        raise ValueError(
            "signal_mode must be "
            "'zscore' or 'quantile'"
        )

    if use_ar1_filter:

        valid_signal = (
            valid_signal
            &
            ar1_phi.notna().any(axis=1)
        )

    if use_pc1_regime_filter:

        valid_signal = (
            valid_signal
            &
            pc1_threshold.notna()
        )

    if not valid_signal.any():

        raise ValueError(
            "No valid trading dates."
        )

    first_signal_date = (
        valid_signal[
            valid_signal
        ].index[0]
    )

    start_t = (
        stock_returns.index.get_loc(
            first_signal_date
        )
    )

    # ==================================================
    # 8. WALK FORWARD
    # ==================================================

    for t in range(
        start_t,
        len(stock_returns) - 1
    ):

        date = stock_returns.index[t]
        next_date = stock_returns.index[t + 1]

        # ------------------------------------------------
        # Today's signal information
        # ------------------------------------------------

        z_today = zscores.iloc[t]

        spread_today = spreads.iloc[t]

        lower_today = (
            lower_threshold.iloc[t]
        )

        median_today = (
            median_threshold.iloc[t]
        )

        upper_today = (
            upper_threshold.iloc[t]
        )

        if use_ar1_filter:
            phi_today = ar1_phi.iloc[t]
        else:
            phi_today = None

        # ------------------------------------------------
        # Today's PC1 regime
        # ------------------------------------------------

        if use_pc1_regime_filter:

            regime_high = bool(
                high_pc1_regime.iloc[t]
            )

        else:

            regime_high = True

        # ==================================================
        # A. UPDATE POSITION STATES
        # ==================================================

        old_positions = positions.copy()

        # In a LOW-PC1 regime:
        #
        # max_positions = 0
        #
        # Existing positions are still processed
        # for exits, but no new positions can open.

        allowed_max_positions = (
            max_positions
            if regime_high
            else 0
        )

        positions, holding_days = update_positions(
            positions=positions,
            holding_days=holding_days,

            signal_mode=signal_mode,

            z_today=z_today,
            entry_z=entry_z,

            spread_today=spread_today,
            lower_today=lower_today,
            median_today=median_today,
            upper_today=upper_today,

            max_holding=max_holding,
            max_positions=allowed_max_positions,

            phi_today=phi_today,
            phi_min=phi_min,
            phi_max=phi_max
        )

        positions_changed = (
            not positions.equals(
                old_positions
            )
        )

        # ==================================================
        # B. TODAY'S PAST-ONLY PCA MODEL
        # ==================================================

        train = stock_returns.iloc[
            t - pca_window:t
        ].to_numpy()

        _, _, _, _, Q_k_today = fit_pca(
            train,
            k=k
        )

        # ==================================================
        # C. RAW ALPHA WEIGHTS
        # ==================================================

        raw_weights = (
            build_raw_stock_weights(
                positions=positions,
                position_size=position_size
            )
        )

        raw_turnover = calculate_turnover(
            new_weights=raw_weights,
            old_weights=previous_raw_weights
        )

        # ==================================================
        # D. PCA FACTOR NEUTRALIZATION
        # ==================================================

        days_since_rebalance += 1

        scheduled_rebalance = (
            days_since_rebalance
            >= rebalance_interval
        )

        should_rebalance = (
            positions_changed
            or scheduled_rebalance
        )

        if should_rebalance:

            weights = (
                pca_neutralize_weights(
                    raw_weights=raw_weights,
                    Q_k=Q_k_today,
                    max_gross=max_gross
                )
            )

            days_since_rebalance = 0

        else:

            weights = (
                previous_weights.copy()
            )

        # ==================================================
        # E. FACTOR EXPOSURE
        # ==================================================

        factor_exp = factor_exposure(
            weights,
            Q_k_today
        )

        max_factor_exp = np.max(
            np.abs(factor_exp)
        )

        # ==================================================
        # F. TURNOVER + COSTS
        # ==================================================

        turnover = calculate_turnover(
            new_weights=weights,
            old_weights=previous_weights
        )

        transaction_cost = (
            cost_rate * turnover
        )

        extra_pca_turnover = (
            turnover - raw_turnover
        )

        # ==================================================
        # G. NEXT-DAY RETURN
        # ==================================================

        next_stock_returns = (
            stock_returns.iloc[t + 1]
        )

        gross_return = (
            weights
            * next_stock_returns
        ).sum()

        net_return = (
            gross_return
            - transaction_cost
        )

        # ==================================================
        # H. SAVE HISTORY
        # ==================================================

        position_row = positions.copy()
        position_row.name = date

        weight_row = weights.copy()
        weight_row.name = date

        position_history.append(
            position_row
        )

        weight_history.append(
            weight_row
        )

        performance_history.append({

            "date":
                next_date,

            "gross_return":
                gross_return,

            "transaction_cost":
                transaction_cost,

            "net_return":
                net_return,

            "turnover":
                turnover,

            "raw_turnover":
                raw_turnover,

            "extra_pca_turnover":
                extra_pca_turnover,

            "net_exposure":
                net_exposure(weights),

            "gross_exposure":
                gross_exposure(weights),

            "active_positions":
                (positions != 0).sum(),

            "max_factor_exposure":
                max_factor_exp,

            "rebalanced":
                should_rebalance,

            # Regime diagnostics
            "pc1_share":
                pc1_share.iloc[t],

            "pc1_threshold":
                pc1_threshold.iloc[t],

            "high_pc1_regime":
                regime_high,

            "new_entries_allowed":
                regime_high
        })

        previous_raw_weights = (
            raw_weights.copy()
        )

        previous_weights = (
            weights.copy()
        )

    # ==================================================
    # 9. OUTPUT
    # ==================================================

    positions_df = pd.DataFrame(
        position_history
    )

    weights_df = pd.DataFrame(
        weight_history
    )

    performance_df = pd.DataFrame(
        performance_history
    ).set_index("date")

    return {

        "residuals":
            residuals,

        "spreads":
            spreads,

        "zscores":
            zscores,

        "lower_threshold":
            lower_threshold,

        "median_threshold":
            median_threshold,

        "upper_threshold":
            upper_threshold,

        "ar1_alpha":
            ar1_alpha,

        "ar1_phi":
            ar1_phi,

        "ar1_half_life":
            ar1_half_life,

        "pc1_share":
            pc1_share,

        "pc1_threshold":
            pc1_threshold,

        "high_pc1_regime":
            high_pc1_regime,

        "positions":
            positions_df,

        "weights":
            weights_df,

        "performance":
            performance_df
    }