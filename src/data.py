import pandas as pd
import yfinance as yf


DEFAULT_TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META",
    "JPM", "BAC", "GS",
    "XOM", "CVX",
    "JNJ", "PFE",
    "WMT", "COST",
    "BA", "CAT",
    "KO", "PEP",
    "DIS", "NFLX"
]

HEDGE_TICKER = "SPY"


def download_prices(
    tickers=DEFAULT_TICKERS,
    hedge_ticker=HEDGE_TICKER,
    start="2019-01-01",
    end=None
):
    """
    Download adjusted closing prices for stocks + hedge ETF.
    """

    symbols = list(tickers)

    if hedge_ticker not in symbols:
        symbols.append(hedge_ticker)

    data = yf.download(
        symbols,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False
    )

    prices = data["Close"].copy()

    # Remove dates where data is missing.
    prices = prices.dropna(how="any")

    return prices


def compute_returns(prices):
    """
    Compute simple daily returns:
    
    """

    returns = prices.pct_change(fill_method=None)

    return returns.dropna(how="any")


def load_market_data(
    tickers=DEFAULT_TICKERS,
    hedge_ticker=HEDGE_TICKER,
    start="2019-01-01",
    end=None
):

    prices = download_prices(
        tickers=tickers,
        hedge_ticker=hedge_ticker,
        start=start,
        end=end
    )

    returns = compute_returns(prices)

    stock_returns = returns[list(tickers)].copy()

    hedge_returns = returns[hedge_ticker].copy()
    hedge_returns.name = hedge_ticker

    return prices, stock_returns, hedge_returns