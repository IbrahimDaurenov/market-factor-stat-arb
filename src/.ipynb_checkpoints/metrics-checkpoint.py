import numpy as np
import pandas as pd


def equity_curve(returns, initial_capital=100_000):
    """Build the portfolio equity curve."""

    growth = (1 + returns).cumprod()

    return initial_capital * growth


def cumulative_return(returns):
    """Compute total compounded return."""

    return (1 + returns).prod() - 1


def annualized_volatility(
    returns,
    periods_per_year=252
):
    """Compute annualized return volatility."""

    daily_vol = returns.std(ddof=1)

    return (
        daily_vol
        * np.sqrt(periods_per_year)
    )


def sharpe_ratio(
    returns,
    risk_free_rate=0.0,
    periods_per_year=252
):
    """Compute annualized Sharpe ratio."""

    daily_rf = (
        risk_free_rate
        / periods_per_year
    )

    excess_returns = (
        returns - daily_rf
    )

    volatility = excess_returns.std(
        ddof=1
    )

    if volatility == 0:
        return np.nan

    return (
        np.sqrt(periods_per_year)
        * excess_returns.mean()
        / volatility
    )


def drawdown_series(equity):
    """Compute drawdown from the running peak."""

    running_max = equity.cummax()

    return (
        equity - running_max
    ) / running_max


def max_drawdown(equity):
    """Return the maximum historical drawdown."""

    return drawdown_series(
        equity
    ).min()


def performance_summary(
    performance,
    initial_capital=100_000
):
    """Summarize backtest performance."""

    returns = (
        performance["net_return"]
        .dropna()
    )

    equity = equity_curve(
        returns,
        initial_capital=initial_capital
    )

    summary = {
        "Cumulative Return":
            cumulative_return(returns),

        "Annualized Volatility":
            annualized_volatility(returns),

        "Sharpe Ratio":
            sharpe_ratio(returns),

        "Max Drawdown":
            max_drawdown(equity),

        "Average Daily Turnover":
            performance["turnover"].mean(),

        "Total Transaction Costs":
            performance[
                "transaction_cost"
            ].sum(),

        "Average Gross Exposure":
            performance[
                "gross_exposure"
            ].mean(),

        "Average Active Positions":
            performance[
                "active_positions"
            ].mean()
    }

    return pd.Series(summary)