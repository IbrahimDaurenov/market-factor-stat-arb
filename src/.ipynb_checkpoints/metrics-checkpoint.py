import numpy as np
import pandas as pd


def equity_curve(returns, initial_capital=100_000):
    """
    Build equity curve from periodic returns.

        V_t = V_{t-1} (1 + R_t)
    """

    growth = (1 + returns).cumprod()

    return initial_capital * growth


def cumulative_return(returns):
    """
    Total compounded return:

        prod(1 + R_t) - 1
    """

    return (1 + returns).prod() - 1


def annualized_volatility(returns, periods_per_year=252):
    """
    Annualized volatility.
    """

    daily_vol = returns.std(ddof=1)

    return daily_vol * np.sqrt(periods_per_year)


def sharpe_ratio(
    returns,
    risk_free_rate=0.0,
    periods_per_year=252
):
    """
    Annualized Sharpe ratio.

    Assumes risk_free_rate is annualized.
    """

    if returns.std(ddof=1) == 0:
        return np.nan

    daily_rf = risk_free_rate / periods_per_year

    excess_returns = returns - daily_rf

    return (
        np.sqrt(periods_per_year)
        * excess_returns.mean()
        / excess_returns.std(ddof=1)
    )


def drawdown_series(equity):
    """
    Drawdown from previous running peak.

        D_t = (V_t - M_t) / M_t
    """

    running_max = equity.cummax()

    drawdown = (
        equity - running_max
    ) / running_max

    return drawdown


def max_drawdown(equity):
    """
    Worst historical drawdown.
    """

    drawdown = drawdown_series(equity)

    return drawdown.min()


def performance_summary(
    performance,
    initial_capital=100_000
):
    """
    Summarize backtest performance.

    Expects performance DataFrame returned
    by run_backtest().
    """

    returns = performance["net_return"].dropna()

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
            performance["transaction_cost"].sum(),

        "Average Gross Exposure":
            performance["gross_exposure"].mean(),

        "Average Active Positions":
            performance["active_positions"].mean()
    }

    return pd.Series(summary)