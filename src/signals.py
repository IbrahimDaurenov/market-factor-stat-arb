import numpy as np
import pandas as pd


def rolling_zscore(spreads, window=252):
    """
    Past-only rolling z-score.

        z_t = (s_t - mu_t) / sigma_t

    Mean and std use only past values:
        s_{t-window}, ..., s_{t-1}
    """

    past_spreads = spreads.shift(1)

    rolling_mean = past_spreads.rolling(
        window=window,
        min_periods=window
    ).mean()

    rolling_std = past_spreads.rolling(
        window=window,
        min_periods=window
    ).std(ddof=1)

    return (
        spreads - rolling_mean
    ) / rolling_std


def rolling_empirical_quantiles(
    spreads,
    window=252,
    lower_q=0.025,
    upper_q=0.975
):
    """
    Past-only empirical thresholds.

    For each day t and stock:

        lower_t = historical lower quantile
        median_t = historical median
        upper_t = historical upper quantile

    using only:
        s_{t-window}, ..., s_{t-1}
    """

    past_spreads = spreads.shift(1)

    lower = past_spreads.rolling(
        window=window,
        min_periods=window
    ).quantile(lower_q)

    median = past_spreads.rolling(
        window=window,
        min_periods=window
    ).quantile(0.50)

    upper = past_spreads.rolling(
        window=window,
        min_periods=window
    ).quantile(upper_q)

    return lower, median, upper


def rolling_ar1_parameters(
    residuals,
    window=252
):
    """
    Rolling AR(1) diagnostic.

        S_{t+1} = alpha + phi S_t + error

    where S_t is cumulative residual return.

    Kept as an optional diagnostic/filter.
    """

    residual_level = residuals.cumsum()

    x = residual_level.shift(1)
    y = residual_level

    rolling_cov = x.rolling(
        window=window,
        min_periods=window
    ).cov(y)

    rolling_var = x.rolling(
        window=window,
        min_periods=window
    ).var(ddof=1)

    phi_raw = rolling_cov / rolling_var

    x_mean = x.rolling(
        window=window,
        min_periods=window
    ).mean()

    y_mean = y.rolling(
        window=window,
        min_periods=window
    ).mean()

    alpha_raw = (
        y_mean - phi_raw * x_mean
    )

    # Past-only parameters.
    phi = phi_raw.shift(1)
    alpha = alpha_raw.shift(1)

    half_life = (
        np.log(0.5) / np.log(phi)
    )

    half_life = half_life.where(
        (phi > 0) & (phi < 1)
    )

    return alpha, phi, half_life


def update_positions(
    positions,
    holding_days,

    # Signal type
    signal_mode="zscore",

    # z-score version
    z_today=None,
    entry_z=2.5,

    # empirical quantile version
    spread_today=None,
    lower_today=None,
    median_today=None,
    upper_today=None,

    # position management
    max_holding=20,
    max_positions=5,

    # optional AR filter
    phi_today=None,
    phi_min=0.0,
    phi_max=1.0
):
    """
    Update persistent position states.

    Position:
         1 = long
        -1 = short
         0 = no position

    signal_mode="zscore":

        entry:
            z < -entry_z -> long
            z > +entry_z -> short

        exit:
            long  -> z >= 0
            short -> z <= 0


    signal_mode="quantile":

        entry:
            spread < lower quantile -> long
            spread > upper quantile -> short

        exit:
            long  -> spread >= median
            short -> spread <= median
    """

    positions = positions.copy()
    holding_days = holding_days.copy()

    exited_today = set()

    # ==================================================
    # 1. EXIT / HOLD EXISTING POSITIONS
    # ==================================================

    for ticker in positions.index:

        position = positions[ticker]

        if position == 0:
            continue

        holding_days[ticker] += 1

        exit_trade = False

        # ----------------------------------------------
        # Z-SCORE EXIT
        # ----------------------------------------------

        if signal_mode == "zscore":

            z = z_today[ticker]

            if not np.isnan(z):

                if position == 1 and z >= 0:
                    exit_trade = True

                elif position == -1 and z <= 0:
                    exit_trade = True

        # ----------------------------------------------
        # EMPIRICAL-QUANTILE EXIT
        # ----------------------------------------------

        elif signal_mode == "quantile":

            spread = spread_today[ticker]
            median = median_today[ticker]

            if (
                not np.isnan(spread)
                and not np.isnan(median)
            ):

                if position == 1 and spread >= median:
                    exit_trade = True

                elif position == -1 and spread <= median:
                    exit_trade = True

        else:
            raise ValueError(
                "signal_mode must be "
                "'zscore' or 'quantile'"
            )

        # Maximum holding period.
        if holding_days[ticker] >= max_holding:
            exit_trade = True

        if exit_trade:

            positions[ticker] = 0
            holding_days[ticker] = 0

            exited_today.add(ticker)

    # ==================================================
    # 2. HOW MANY NEW POSITIONS CAN WE OPEN?
    # ==================================================

    active_positions = (
        positions != 0
    ).sum()

    available_slots = (
        max_positions - active_positions
    )

    if available_slots <= 0:
        return positions, holding_days

    # ==================================================
    # 3. ENTRY CANDIDATES
    # ==================================================

    candidates = []

    for ticker in positions.index:

        if positions[ticker] != 0:
            continue

        if ticker in exited_today:
            continue

        # ----------------------------------------------
        # OPTIONAL AR FILTER
        # ----------------------------------------------

        if phi_today is not None:

            phi = phi_today[ticker]

            if np.isnan(phi):
                continue

            if not (
                phi_min < phi < phi_max
            ):
                continue

        # ----------------------------------------------
        # Z-SCORE SIGNAL
        # ----------------------------------------------

        if signal_mode == "zscore":

            z = z_today[ticker]

            if np.isnan(z):
                continue

            if z < -entry_z:

                candidates.append(
                    (
                        ticker,
                        1,
                        abs(z)
                    )
                )

            elif z > entry_z:

                candidates.append(
                    (
                        ticker,
                        -1,
                        abs(z)
                    )
                )

        # ----------------------------------------------
        # EMPIRICAL-QUANTILE SIGNAL
        # ----------------------------------------------

        elif signal_mode == "quantile":

            spread = spread_today[ticker]
            lower = lower_today[ticker]
            median = median_today[ticker]
            upper = upper_today[ticker]

            if (
                np.isnan(spread)
                or np.isnan(lower)
                or np.isnan(median)
                or np.isnan(upper)
            ):
                continue

            # Scale used only to rank simultaneous
            # candidate signals by extremeness.
            scale = upper - lower

            if scale <= 0:
                continue

            strength = abs(
                spread - median
            ) / scale

            if spread < lower:

                candidates.append(
                    (
                        ticker,
                        1,
                        strength
                    )
                )

            elif spread > upper:

                candidates.append(
                    (
                        ticker,
                        -1,
                        strength
                    )
                )

    # Strongest signals first.
    candidates.sort(
        key=lambda x: x[2],
        reverse=True
    )

    # ==================================================
    # 4. OPEN POSITIONS
    # ==================================================

    for ticker, direction, _ in candidates[
        :available_slots
    ]:

        positions[ticker] = direction
        holding_days[ticker] = 1

    return positions, holding_days