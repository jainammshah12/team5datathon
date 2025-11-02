"""Module for fetching stock data from yfinance API."""

import yfinance as yf
import pandas as pd
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import time
from tqdm import tqdm


def fetch_daily_stock_data(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    period: str = "1y",
) -> pd.DataFrame:
    """
    Fetch daily stock price data for a given ticker.

    Args:
        ticker: Stock symbol (e.g., 'AAPL', 'MSFT')
        start_date: Start date in YYYY-MM-DD format (optional)
        end_date: End date in YYYY-MM-DD format (optional)
        period: Period to fetch (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)

    Returns:
        DataFrame with daily price data (Open, High, Low, Close, Volume, Dividends, Stock Splits)
    """
    try:
        stock = yf.Ticker(ticker)

        if start_date and end_date:
            hist = stock.history(start=start_date, end=end_date)
        else:
            hist = stock.history(period=period)

        if hist.empty:
            print(f"Warning: No data found for {ticker}")
            return pd.DataFrame()

        # Reset index to make Date a column
        hist.reset_index(inplace=True)
        hist["Date"] = pd.to_datetime(hist["Date"]).dt.date

        # Add ticker column
        hist["Ticker"] = ticker

        return hist
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()


def fetch_stock_info(ticker: str) -> Dict:
    """
    Fetch company information and financial metrics.

    Args:
        ticker: Stock symbol

    Returns:
        Dictionary with company info, financials, and key metrics
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # Extract key metrics
        metrics = {
            "Ticker": ticker,
            "Company Name": info.get("longName", ""),
            "Sector": info.get("sector", ""),
            "Industry": info.get("industry", ""),
            "Market Cap": info.get("marketCap", None),
            "Current Price": info.get("currentPrice", None),
            "EPS": info.get("trailingEps", None),
            "PE Ratio": info.get("trailingPE", None),
            "Revenue": info.get("totalRevenue", None),
            "Revenue Growth": info.get("revenueGrowth", None),
            "Operating Margin": info.get("operatingMargins", None),
            "Profit Margin": info.get("profitMargins", None),
            "Free Cash Flow": info.get("freeCashflow", None),
            "Debt to Equity": info.get("debtToEquity", None),
            "ROE": info.get("returnOnEquity", None),
            "ROA": info.get("returnOnAssets", None),
            "52 Week High": info.get("fiftyTwoWeekHigh", None),
            "52 Week Low": info.get("fiftyTwoWeekLow", None),
            "Dividend Yield": info.get("dividendYield", None),
            "Beta": info.get("beta", None),
        }

        return metrics
    except Exception as e:
        print(f"Error fetching info for {ticker}: {e}")
        return {"Ticker": ticker}


def fetch_financials(ticker: str) -> Dict[str, pd.DataFrame]:
    """
    Fetch financial statements (income statement, balance sheet, cash flow).

    Args:
        ticker: Stock symbol

    Returns:
        Dictionary with 'income_statement', 'balance_sheet', 'cash_flow'
    """
    try:
        stock = yf.Ticker(ticker)

        return {
            "income_statement": stock.financials,
            "balance_sheet": stock.balance_sheet,
            "cash_flow": stock.cashflow,
        }
    except Exception as e:
        print(f"Error fetching financials for {ticker}: {e}")
        return {}


def create_daily_performance_dataset(
    tickers: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    include_financials: bool = True,
    delay: float = 0.1,
) -> pd.DataFrame:
    """
    Create a comprehensive daily performance dataset for multiple stocks.

    Args:
        tickers: List of stock symbols
        start_date: Start date in YYYY-MM-DD format (optional)
        end_date: End date in YYYY-MM-DD format (optional)
        include_financials: Whether to include company info and financials
        delay: Delay between API calls (seconds) to avoid rate limiting

    Returns:
        DataFrame with daily performance data for all stocks
    """
    all_data = []
    all_info = []

    print(f"Fetching data for {len(tickers)} stocks...")

    for ticker in tqdm(tickers, desc="Processing stocks"):
        # Fetch daily price data
        daily_data = fetch_daily_stock_data(ticker, start_date, end_date)

        if not daily_data.empty:
            all_data.append(daily_data)

        # Fetch company info and financials if requested
        if include_financials:
            info = fetch_stock_info(ticker)
            all_info.append(info)

        # Rate limiting
        time.sleep(delay)

    # Combine all daily data
    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
    else:
        print("Warning: No daily data fetched")
        return pd.DataFrame()

    # Add company info to daily data if available
    if all_info:
        info_df = pd.DataFrame(all_info)
        combined_data = combined_data.merge(
            info_df[
                [
                    "Ticker",
                    "Company Name",
                    "Sector",
                    "Industry",
                    "Market Cap",
                    "EPS",
                    "Revenue",
                    "Revenue Growth",
                    "PE Ratio",
                    "Free Cash Flow",
                    "ROE",
                    "Beta",
                ]
            ],
            on="Ticker",
            how="left",
        )

    # Calculate daily performance metrics
    combined_data["Daily Return"] = combined_data.groupby("Ticker")[
        "Close"
    ].pct_change()
    combined_data["Price Change"] = combined_data.groupby("Ticker")["Close"].diff()
    combined_data["Price Change %"] = (
        combined_data["Price Change"] / combined_data["Close"].shift(1) * 100
    )

    # Sort by date and ticker
    combined_data = combined_data.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    return combined_data


def get_sp500_tickers() -> List[str]:
    """
    Get list of S&P 500 tickers from local CSV file.

    Returns:
        List of ticker symbols
    """
    try:
        df = pd.read_csv("2025-08-15_composition_sp500.csv")
        # Extract ticker from Symbol column
        tickers = df["Symbol"].tolist()
        return [ticker.strip() for ticker in tickers if pd.notna(ticker)]
    except Exception as e:
        print(f"Error reading S&P 500 tickers: {e}")
        return []


def query_stock_data(
    ticker: str, start_date: str, end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Query function to get stock data for a specific ticker and date range.
    This is the main API function for querying stocks by date.

    Args:
        ticker: Stock symbol
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format (defaults to today if None)

    Returns:
        DataFrame with stock data for the specified period
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    data = fetch_daily_stock_data(ticker, start_date, end_date)

    if not data.empty:
        # Add calculated metrics
        data["Daily Return"] = data["Close"].pct_change()
        data["Price Change"] = data["Close"].diff()
        data["Volume Change"] = data["Volume"].diff()

        # Get company info
        info = fetch_stock_info(ticker)
        for key, value in info.items():
            if key != "Ticker":
                data[key] = value

    return data


def query_multiple_stocks(
    tickers: List[str], start_date: str, end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Query multiple stocks for a date range.

    Args:
        tickers: List of stock symbols
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format (defaults to today if None)

    Returns:
        DataFrame with data for all specified stocks
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    all_data = []

    for ticker in tickers:
        data = query_stock_data(ticker, start_date, end_date)
        if not data.empty:
            all_data.append(data)
        time.sleep(0.1)  # Rate limiting

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()
