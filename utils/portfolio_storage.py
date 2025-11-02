"""Portfolio storage utilities for saving/loading portfolios from S3."""

import pandas as pd
import json
import os
from typing import Dict, List, Optional, Tuple
from utils.s3_utils import read_csv_from_s3, upload_file_to_s3, list_files_in_s3
from io import StringIO


PORTFOLIO_S3_PREFIX = "data/portfolios/"
PORTFOLIO_FILENAME = "user_portfolio.csv"


def get_portfolio_key(filename: str = None) -> str:
    """
    Get S3 key for portfolio file.

    Args:
        filename: Optional custom filename (defaults to user_portfolio.csv)

    Returns:
        S3 key path
    """
    filename = filename or PORTFOLIO_FILENAME
    return f"{PORTFOLIO_S3_PREFIX}{filename}"


def load_portfolio_from_s3(
    filename: str = None,
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Load portfolio from S3.

    Args:
        filename: Portfolio filename (defaults to user_portfolio.csv)

    Returns:
        Tuple of (DataFrame, error_message)
    """
    try:
        key = get_portfolio_key(filename)
        df = read_csv_from_s3(key)
        return df, None
    except Exception as e:
        # File might not exist yet
        return None, None  # Return None instead of error for non-existent files


def save_portfolio_to_s3(
    portfolio_df: pd.DataFrame, filename: str = None
) -> Tuple[bool, Optional[str]]:
    """
    Save portfolio DataFrame to S3 (creates or updates existing file).

    Args:
        portfolio_df: Portfolio DataFrame with columns: Ticker, Price, Quantity, Date_Bought
        filename: Portfolio filename (defaults to user_portfolio.csv - only 1 file per user)

    Returns:
        Tuple of (success, error_message)
    """
    try:
        # Use default filename for single user
        if filename is None:
            filename = PORTFOLIO_FILENAME

        # Validate required columns
        required_cols = ["Ticker", "Price", "Quantity", "Date_Bought"]
        if not all(col in portfolio_df.columns for col in required_cols):
            return False, f"Missing required columns. Need: {required_cols}"

        # Ensure columns are in correct order
        portfolio_df = portfolio_df[required_cols].copy()

        # Convert to CSV
        csv_buffer = StringIO()
        portfolio_df.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue().encode("utf-8")

        # Upload to S3 (will overwrite if exists)
        key = get_portfolio_key(filename)
        upload_file_to_s3(csv_content, key)

        return True, None
    except Exception as e:
        return False, str(e)


def list_portfolios_in_s3() -> List[str]:
    """
    List all portfolio files in S3.

    Returns:
        List of portfolio filenames
    """
    try:
        files = list_files_in_s3(PORTFOLIO_S3_PREFIX)
        # Extract just the filenames
        filenames = [
            f.replace(PORTFOLIO_S3_PREFIX, "") for f in files if f.endswith(".csv")
        ]
        return filenames
    except Exception as e:
        return []


def add_portfolio_entry(
    ticker: str,
    price: float,
    quantity: int,
    date_bought: str,
    existing_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Add a new entry to portfolio DataFrame.

    Args:
        ticker: Stock symbol
        price: Purchase price per share
        quantity: Number of shares
        date_bought: Purchase date (YYYY-MM-DD)
        existing_df: Existing portfolio DataFrame (None for new portfolio)

    Returns:
        Updated DataFrame
    """
    new_entry = pd.DataFrame(
        {
            "Ticker": [ticker.upper()],
            "Price": [float(price)],
            "Quantity": [int(quantity)],
            "Date_Bought": [date_bought],
        }
    )

    if existing_df is not None and len(existing_df) > 0:
        # Append to existing
        return pd.concat([existing_df, new_entry], ignore_index=True)
    else:
        # New portfolio
        return new_entry


def calculate_portfolio_value(
    portfolio_df: pd.DataFrame, current_prices: Dict[str, float] = None
) -> pd.DataFrame:
    """
    Calculate portfolio value and performance metrics.

    Args:
        portfolio_df: Portfolio DataFrame
        current_prices: Optional dict of {ticker: current_price}

    Returns:
        DataFrame with additional columns: Total_Cost, Current_Value, P&L, Return_Pct
    """
    df = portfolio_df.copy()

    # Calculate total cost
    df["Total_Cost"] = df["Price"] * df["Quantity"]

    # Calculate current value if prices provided
    if current_prices:
        df["Current_Price"] = df["Ticker"].map(current_prices).fillna(df["Price"])
        df["Current_Value"] = df["Current_Price"] * df["Quantity"]
        df["P&L"] = df["Current_Value"] - df["Total_Cost"]
        df["Return_Pct"] = (df["P&L"] / df["Total_Cost"] * 100).round(2)

    return df
