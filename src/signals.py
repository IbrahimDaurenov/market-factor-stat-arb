import numpy as np


def rolling_zscore(spreads, window=252):
    """Compute past-only rolling z-scores."""

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
    """Compute past-only rolling empirical thresholds."""

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
    """Estimate rolling AR(1) parameters for cumulative residuals."""

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

    phi_raw = (
        rolling_cov
        / rolling_var
    )

    x_mean = x.rolling(
        window=window,
        min_periods=window
    ).mean()

    y_mean = y.rolling(
        window=window,
        min_periods=window
    ).mean()

    alpha_raw = (
        y_mean
        - phi_raw * x_mean
    )

    phi = phi_raw.shift(1)
    alpha = alpha_raw.shift(1)

    half_life = (
        np.log(0.5)
        / np.log(phi)
    )

    half_life = half_life.where(
        (phi > 0)
        & (phi < 1)
    )

    return alpha, phi, half_life


def update_positions(
    positions,
    holding_days,
    signal_mode="zscore",
    z_today=None,
    entry_z=2.5,
    spread_today=None,
    lower_today=None,
    median_today=None,
    upper_today=None,
    max_holding=20,
    max_positions=5,
    phi_today=None,
    phi_min=0.0,
    phi_max=1.0
):
    """Update persistent long, short, and flat position states."""

    if signal_mode not in {
        "zscore",
        "quantile"
    }:
        raise ValueError(
            "signal_mode must be "
            "'zscore' or 'quantile'"
        )

    positions = positions.copy()
    holding_days = holding_days.copy()

    exited_today = set()

    for ticker in positions.index:
        position = positions[ticker]

        if position == 0:
            continue

        holding_days[ticker] += 1
        exit_trade = False

        if signal_mode == "zscore":
            z = z_today[ticker]

            if not np.isnan(z):
                if (
                    position == 1
                    and z >= 0
                ):
                    exit_trade = True

                elif (
                    position == -1
                    and z <= 0
                ):
                    exit_trade = True

        else:
            spread = spread_today[ticker]
            median = median_today[ticker]

            if (
                not np.isnan(spread)
                and not np.isnan(median)
            ):
                if (
                    position == 1
                    and spread >= median
                ):
                    exit_trade = True

                elif (
                    position == -1
                    and spread <= median
                ):
                    exit_trade = True

        if (
            holding_days[ticker]
            >= max_holding
        ):
            exit_trade = True

        if exit_trade:
            positions[ticker] = 0
            holding_days[ticker] = 0
            exited_today.add(ticker)

    active_positions = (
        positions != 0
    ).sum()

    available_slots = (
        max_positions
        - active_positions
    )

    if available_slots <= 0:
        return (
            positions,
            holding_days
        )

    candidates = []

    for ticker in positions.index:
        if positions[ticker] != 0:
            continue

        if ticker in exited_today:
            continue

        if phi_today is not None:
            phi = phi_today[ticker]

            if np.isnan(phi):
                continue

            if not (
                phi_min < phi < phi_max
            ):
                continue

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

        else:
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

            scale = upper - lower

            if scale <= 0:
                continue

            strength = (
                abs(spread - median)
                / scale
            )

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

    candidates.sort(
        key=lambda x: x[2],
        reverse=True
    )

    for ticker, direction, _ in (
        candidates[:available_slots]
    ):
        positions[ticker] = direction
        holding_days[ticker] = 1

    return positions, holding_days